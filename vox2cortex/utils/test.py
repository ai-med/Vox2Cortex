
""" Test procedure """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import os
import logging
import json

import torch

from utils.logging import init_logging, get_log_dir
from utils.utils import string_dict, dict_to_lower_dict, update_dict
from utils.modes import ExecModes
from utils.evaluate import ModelEvaluator
from data.dataset_split_handler import dataset_split_handler
from models.model_handler import ModelHandler
from utils.params import DATASET_PARAMS, DATASET_SPLIT_PARAMS
from utils.model_names import (
    INTERMEDIATE_MODEL_NAME,
    BEST_MODEL_NAME,
    FINAL_MODEL_NAME
)

def _get_test_dataset_params(hps, training_hps):
    """ Get test split: All parameters equal to the training parameters but the
    dataset can potentially be different
    """
    if (hps['DATASET'] == training_hps['DATASET'] and
        (any(hps[k] != training_hps[k] for k in DATASET_SPLIT_PARAMS))):
        raise RuntimeError(
            "Cannot test on the same dataset but potentially different test"
            " split than defined during training!"
        )
    test_dataset_params = {
        k: hps[k] for k in (DATASET_PARAMS + DATASET_SPLIT_PARAMS)
    }
    test_dataset_params = update_dict(training_hps, test_dataset_params)

    return test_dataset_params

def write_test_results(results: dict, model_name: str, experiment_dir: str):
    res_file = os.path.join(experiment_dir, "test_results.txt")
    with open(res_file, 'a') as f:
        f.write(f"Results of {model_name}:\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

        f.write("\n\n")

    logging.getLogger(ExecModes.TEST.name).info("Test result written to %s",
                                                res_file)

def test_routine(hps: dict, experiment_name, loglevel='INFO', resume=False):
    """ A full testing routine for a trained model

    :param dict hps: Hyperparameters to use.
    :param str experiment_name: The experiment id where the trained models can
    be found.
    :param loglevel: The loglevel of the standard logger to use.
    :param resume: Only for compatibility with training but single test routine
    cannot be resumed.
    """
    experiment_base_dir = hps['EXPERIMENT_BASE_DIR']
    if experiment_name is None:
        print("Please specify experiment name for testing with --exp_name.")
        return

    experiment_dir = os.path.join(experiment_base_dir, experiment_name)
    log_dir = get_log_dir(experiment_dir, create=True)
    hps_to_write = string_dict(hps)

    init_logging(logger_name=ExecModes.TEST.name,
                 exp_name=experiment_name,
                 log_dir=log_dir,
                 loglevel=loglevel,
                 mode=ExecModes.TEST,
                 proj_name=hps['PROJ_NAME'],
                 group_name=hps['GROUP_NAME'],
                 params=hps_to_write,
                 time_logging=hps['TIME_LOGGING'])

    testLogger = logging.getLogger(ExecModes.TEST.name)
    if resume:
        testLogger.warning("Test routine cannot be resumed, ignoring this"\
                           " parameter.")
    experiment_dir = os.path.join(experiment_base_dir, experiment_name)
    # Directoy where test results are written to
    test_dir = os.path.join(
        experiment_dir,
        "test_template_"
        + str(hps['N_TEMPLATE_VERTICES_TEST'])
        + f"_{hps['DATASET']}"
    )
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    testLogger.info("Testing %s...", experiment_name)

    param_file = os.path.join(experiment_dir, "params.json")
    with open(param_file, 'r') as f:
        training_hps = json.load(f)

    # Choose template according to the number of template vertices specified
    # (which may be different than during training)
    if hps['N_TEMPLATE_VERTICES'] != hps['N_TEMPLATE_VERTICES_TEST']:
        hps['MODEL_CONFIG']['MESH_TEMPLATE'] =\
            hps['MODEL_CONFIG']['MESH_TEMPLATE'].replace(
                str(hps['N_TEMPLATE_VERTICES']), str(hps['N_TEMPLATE_VERTICES_TEST'])
            )
    testLogger.info("Using template %s", hps['MODEL_CONFIG']['MESH_TEMPLATE'])

    # Lower case param names as input to constructors/functions
    training_hps_lower = dict_to_lower_dict(training_hps)
    hps_lower = dict_to_lower_dict(hps)
    model_config = hps_lower['model_config']
    # Check if model configs are equal (besides template)
    for k, v in string_dict(model_config).items():
        v_train = training_hps_lower['model_config'][k]
        if v_train != v and k != 'mesh_template':
            raise RuntimeError(f"Hyperparameter {k.upper()} is not equal to the"\
                               " model that should be tested. Values are "\
                               f" {v_train} and {v}.")

    # Load test dataset
    test_dataset_params = _get_test_dataset_params(hps, training_hps)
    testLogger.info("Loading dataset %s...", test_dataset_params['DATASET'])
    _, _, test_set = dataset_split_handler[test_dataset_params['DATASET']](
        save_dir=test_dir,
        load_only='test',
        **dict_to_lower_dict(test_dataset_params)
    )
    testLogger.info("%d test files.", len(test_set))

    # Use current hps for testing. In particular, the evaluation metrics may be
    # different than during training.
    evaluator = ModelEvaluator(eval_dataset=test_set, save_dir=test_dir,
                               **hps_lower)

    # Test models
    model = ModelHandler[training_hps['ARCHITECTURE']].value(
        ndims=training_hps['NDIMS'],
        n_v_classes=training_hps['N_V_CLASSES'],
        n_m_classes=training_hps['N_M_CLASSES'],
        patch_shape=training_hps['PATCH_SIZE'],
        **model_config
    ).float()

    # Select best and last model by default or model of a certain epoch
    if hps['TEST_MODEL_EPOCH'] > 0:
        model_names = ["epoch_" + str(hps['TEST_MODEL_EPOCH']) + ".model"]
    else:
        model_names = [fn for fn in os.listdir(experiment_dir) if (
            BEST_MODEL_NAME in fn or
            INTERMEDIATE_MODEL_NAME in fn or
            FINAL_MODEL_NAME in fn
            )
        ]

    epochs_file = os.path.join(experiment_dir, "models_to_epochs.json")
    try:
        with open(epochs_file, 'r') as f:
            models_to_epochs = json.load(f)
    except FileNotFoundError:
        testLogger.warning("No models-to-epochs file found, don't know epochs"\
                           " of stored models.")
        models_to_epochs = {}
        for mn in model_names:
            models_to_epochs[mn] = -1 # -1 = unknown

    epochs_tested = []

    for mn in model_names:
        model_path = os.path.join(experiment_dir, mn)
        epoch = models_to_epochs.get(mn, int(hps['TEST_MODEL_EPOCH']))

        # Test each epoch that has been stored
        if epoch not in epochs_tested or epoch == -1:
            testLogger.info("Test model %s stored in training epoch %d",
                            model_path, epoch)

            # Avoid problem of cuda out of memory by first loading to cpu, see
            # https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/3
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.cuda()
            model.eval()

            results = evaluator.evaluate(
                model, epoch, save_meshes=len(test_set),
                remove_previous_meshes=False,
                store_in_orig_coords=True
            )

            write_test_results(results, mn, test_dir)

            epochs_tested.append(epoch)
