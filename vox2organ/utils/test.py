
""" Test procedure """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import re
import os
import sys
import logging
import json
from copy import deepcopy

import wandb
import torch
import numpy as np

import logger
from data.dataset_split_handler import dataset_split_handler
from models.model_handler import ModelHandler
from params.default import DATASET_PARAMS, DATASET_SPLIT_PARAMS
from utils.utils import (
    dict_to_lower_dict,
    update_dict,
    load_checkpoint,
)
from utils.modes import ExecModes
from utils.evaluate import ModelEvaluator
from utils.template import MeshTemplate, TEMPLATE_SPECS
from utils.model_names import *
from utils.graph_conv import *

log = logger.get_std_logger(__name__)

def _assemble_test_hps(hps, training_hps):
    """ Assemble the test params which are mostly equal to the training params
    but there exist exceptions.
    """

    test_hps = deepcopy(training_hps)

    # Potentially different dataset
    if (hps['DATASET'] == training_hps['DATASET'] and
        (any(hps[k] != training_hps[k] for k in DATASET_SPLIT_PARAMS))):
        raise ValueError(
            "Dataset params seem to have changed since training!"
        )
    test_dataset_params = {
        k: hps[k] for k in (DATASET_PARAMS + DATASET_SPLIT_PARAMS)
    }
    test_hps = update_dict(test_hps, test_dataset_params)

    # Other exceptions
    test_hps['DEVICE'] = hps['DEVICE'][0]
    test_hps['TEST_SPLIT'] = hps['TEST_SPLIT']
    test_hps['MESH_TEMPLATE_ID'] = hps['MESH_TEMPLATE_ID']
    test_hps['MODEL_CONFIG']['N_EULER_STEPS'] = hps['MODEL_CONFIG']['N_EULER_STEPS']
    test_hps['TEST_MODEL_EPOCH'] = hps['TEST_MODEL_EPOCH']
    test_hps['SANITY_CHECK_DATA'] = hps['SANITY_CHECK_DATA']
    test_hps['EVAL_METRICS'] = hps['EVAL_METRICS']
    test_hps['REGISTER_MESHES_TO_VOXELS'] = hps['REGISTER_MESHES_TO_VOXELS']

    # Warnings
    if hps['MESH_TEMPLATE_ID'] != training_hps['MESH_TEMPLATE_ID']:
        log.warning(
            "Using template %s, which is different to training template %s",
            hps['MESH_TEMPLATE_ID'],
            training_hps['MESH_TEMPLATE_ID']
        )

    # str -> object
    test_hps['MODEL_CONFIG']['GC'] = eval(test_hps['MODEL_CONFIG']['GC'])

    return test_hps


def test_routine(hps: dict, resume=False):
    """ A full testing routine for a trained model

    :param dict hps: Hyperparameters to use.
    :param resume: Only for compatibility with training but single test routine
    cannot be resumed.
    """

    experiment_name = hps['EXPERIMENT_NAME']

    if experiment_name is None:
        print("Please specify experiment name for testing with --exp_name.")
        sys.exit(1)
    if resume:
        log.warning(
            "Test routine cannot be resumed, ignoring parameter 'resume'."
        )

    # Assemble test params from current hps and training params
    param_file = logger.get_params_file()
    with open(param_file, 'r') as f:
        training_hps = json.load(f)
    test_hps = _assemble_test_hps(hps, training_hps)
    test_hps_lower = dict_to_lower_dict(test_hps)

    test_split = test_hps.get('TEST_SPLIT', 'test')
    device = test_hps['DEVICE']
    torch.cuda.set_device(device)

    # Directoy where test results are written to
    prefix = "post_processed_" if test_hps['REGISTER_MESHES_TO_VOXELS'] else ""
    logger.set_eval_dir_name(
        os.path.join(
            prefix
            + test_split
            + "_template_"
            + test_hps['MESH_TEMPLATE_ID']
            + f"_{test_hps['DATASET']}"
            + f"_n_{test_hps['MODEL_CONFIG']['N_EULER_STEPS']}"
        )
    )
    test_dir = logger.get_eval_dir()

    log.info("Testing %s...", experiment_name)

    # Load test dataset
    log.info("Loading dataset %s...", test_hps['DATASET'])
    train_set, val_set, test_set = dataset_split_handler[test_hps['DATASET']](
        save_dir=test_dir,
        load_only=test_split,
        check_dir=logger.get_log_dir(),
        **test_hps_lower
    )
    if test_split == 'validation':
        test_set = val_set
    if test_split == 'train':
        test_set = train_set
    log.info("%d test files.", len(test_set))

    # Load template
    # All meshes should have the same transformation matrix
    trans_affine = test_set.get_data_element(0)[
        'trans_affine_label'
    ]
    assert all(
        np.allclose(
            trans_affine,
            test_set.get_data_element(i)[
                'trans_affine_label'
            ]
        )
        for i in range(len(test_set))
    )

    template = MeshTemplate(
        mesh_label_names=list(test_set.mesh_label_names.keys()),
        trans_affine=trans_affine,
        **TEMPLATE_SPECS[test_hps['MESH_TEMPLATE_ID']]
    )

    evaluator = ModelEvaluator(
        eval_dataset=test_set,
        save_dir=test_dir,
        mesh_template=template,
        **test_hps_lower
    )

    # Test models
    model = ModelHandler[test_hps['ARCHITECTURE']].value(
        ndims=test_hps['NDIMS'],
        n_v_classes=test_hps['N_V_CLASSES'],
        n_m_classes=test_hps['N_M_CLASSES'],
        patch_size=test_hps['PATCH_SIZE'],
        **test_hps_lower['model_config']
    ).float()

    # Select best model by default or model of a certain epoch
    if test_hps['TEST_MODEL_EPOCH'] > 0:
        model_names = ["epoch_" + str(test_hps['TEST_MODEL_EPOCH']) + ".pt"]
    else:
        model_names = [
            fn for fn in os.listdir(logger.get_experiment_dir()) if (
                BEST_MODEL_NAME in fn
            )
        ]

    epochs_file = os.path.join(logger.get_experiment_dir(), "models_to_epochs.json")
    try:
        with open(epochs_file, 'r') as f:
            models_to_epochs = json.load(f)
    except FileNotFoundError:
        log.warning(
            "No models-to-epochs file found, don't know epochs of stored"
            " models."
        )
        models_to_epochs = {}
        for mn in model_names:
            models_to_epochs[mn] = -1 # -1 = unknown

    epochs_tested = []

    for mn in model_names:
        model_path = os.path.join(logger.get_experiment_dir(), mn)
        epoch = models_to_epochs.get(mn, int(test_hps['TEST_MODEL_EPOCH']))

        # Test each epoch that has been stored
        if epoch not in epochs_tested or epoch == -1:
            log.info(
                "Test model %s stored in training epoch %d on dataset split '%s'",
                model_path, epoch, test_split
            )

            # Avoid problem of cuda out of memory by first loading to cpu, see
            # https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/3
            model, _, _, _ = load_checkpoint(model, model_path, 'cpu')
            model.to(device)
            model.eval()

            results = evaluator.evaluate(
                model, epoch, device, save_predictions=len(test_set),
                remove_previous_meshes=False,
                register_meshes_to_voxels=test_hps['REGISTER_MESHES_TO_VOXELS']
            )
            results.to_csv(
                os.path.join(test_dir, f"eval_results_epoch_{epoch}.csv"),
                index=False
            )
            results_summary = results.groupby(
                ['Metric', 'Tissue']
            ).mean().reset_index()
            results_summary.to_csv(
                os.path.join(
                    test_dir,
                    f"eval_results_summary_epoch_{epoch}.csv"
                ),
                index=False
            )
            log.info("Summary of evaluation:")
            log.info(results_summary.groupby('Metric').mean())
            log.info("For detailed output see " + test_dir)

            try:
                logger.wandb_test_summary(
                    test_split,
                    results,
                    hps['ENTITY'],
                    hps['PROJ_NAME'],
                    hps['EXPERIMENT_NAME']
                )
            except:
                log.debug("Writing test results to wandb not possible")

            epochs_tested.append(epoch)

    return experiment_name
