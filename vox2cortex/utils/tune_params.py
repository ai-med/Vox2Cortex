
""" Hyperparameter tuning """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import os
import copy
import logging

import json

from utils.utils import string_dict, update_dict
from utils.train import create_exp_directory, Solver
from utils.modes import ExecModes
from utils.logging import init_logging, finish_wandb_run, init_wandb_logging
from utils.evaluate import ModelEvaluator
from models.model_handler import ModelHandler
from data.dataset_split_handler import dataset_split_handler

def tuning_routine(hps, experiment_name=None, loglevel='INFO', **kwargs):
    """
    A hyperparameter sweep.

    :param dict hps: Hyperparameters to use.
    :param str experiment_name (optional): The name of the experiment
    directory. If None, a name is created automatically.
    :param loglevel: The loglevel of the standard logger to use.
    :return: The name of the experiment.
    """
    ###### Prepare tuning experiment ######

    experiment_base_dir = hps['EXPERIMENT_BASE_DIR']

    # Only consider few epochs when tuning parameters
    hps['N_EPOCHS'] = 1500

    # Create directories
    experiment_name, experiment_dir, log_dir =\
            create_exp_directory(experiment_base_dir, experiment_name)
    hps['EXPERIMENT_NAME'] = experiment_name

    # Get a list of possible values for the parameter to tune
    fine_tune = hps['PARAMS_TO_FINE_TUNE'] is not None
    if fine_tune:
        params_to_tune = hps['PARAMS_TO_FINE_TUNE']
    else:
        params_to_tune = hps['PARAMS_TO_TUNE']
    for p in params_to_tune:
        if "." in p: # nested parameter
            plist = p.split(".")
            hps_sub = hps
            for p_ in plist[:-1]:
                hps_sub = hps_sub[p_]
            if not fine_tune:
                hps_sub[plist[-1]] = 'will be tuned'
        else: # not nested
            if not fine_tune:
                hps[p] = 'will be tuned'

    param_possibilities = get_all_possibilities(params_to_tune, hps, fine_tune)

    # Configure logging
    hps_to_write = string_dict(hps)
    param_file = os.path.join(experiment_dir, "params.json")
    with open(param_file, 'w') as f:
        json.dump(hps_to_write, f)
    init_logging(logger_name=ExecModes.TRAIN.name,
                 exp_name=experiment_name,
                 log_dir=log_dir,
                 loglevel=loglevel,
                 mode=ExecModes.TUNE,
                 proj_name=hps['PROJ_NAME'],
                 group_name=hps['GROUP_NAME'],
                 params=hps_to_write,
                 time_logging=hps['TIME_LOGGING'])
    trainLogger = logging.getLogger(ExecModes.TRAIN.name)
    trainLogger.info("Start tuning experiment '%s'...", experiment_name)
    trainLogger.info("Parameter under consideration: %s", params_to_tune)
    trainLogger.info("%d possible choices", len(param_possibilities))

    ###### Load data ######
    hps_lower = dict((k.lower(), v) for k, v in hps.items())
    trainLogger.info("Loading dataset %s...", hps['DATASET'])
    training_set, validation_set, _ = dataset_split_handler[hps['DATASET']](
        save_dir=experiment_dir, **hps_lower
    )
    trainLogger.info("%d training files.", len(training_set))
    trainLogger.info("%d validation files.", len(validation_set))

    # Evaluation during training on validation set
    evaluator = ModelEvaluator(eval_dataset=validation_set,
                               save_dir=experiment_dir, **hps_lower)

    ###### Training with each configuration ######
    results = {
        "Parameter": params_to_tune,
        "All results": {}
    }

    best_score = 0.0
    best_choice = None

    for i, choice in enumerate(param_possibilities):

        run_name = "run_" + str(i)
        run_dir = os.path.join(experiment_base_dir, experiment_name, run_name)
        os.makedirs(run_dir, exist_ok=experiment_name=="debug")

        # Update params with current choice
        hps["ID"] = os.path.join(experiment_name + "." + run_name)
        hps = update_dict(hps, choice)
        hps_to_write = string_dict(hps)
        param_file = os.path.join(run_dir, "params.json")
        with open(param_file, 'w') as f:
            json.dump(hps_to_write, f)

        # Init wandb for every run
        init_wandb_logging(
            exp_name=experiment_name, log_dir=log_dir,
            wandb_proj_name=hps['PROJ_NAME'],
            wandb_group_name=hps['GROUP_NAME'],
            wandb_job_type=ExecModes.TUNE.name.lower(),
            notes=run_name,
            id=hps["ID"],
            params=string_dict(hps)
        )

        trainLogger.info("Choice: %s, %d/%d", str(choice), i+1,
                         len(param_possibilities))

        # Lower case param names as input to constructors/functions
        hps_lower = dict((k.lower(), v) for k, v in hps.items())
        model_config = dict((k.lower(), v) for k, v in hps['MODEL_CONFIG'].items())

        model = ModelHandler[hps['ARCHITECTURE']].value(
            ndims=hps['NDIMS'],
            n_v_classes=hps['N_V_CLASSES'],
            n_m_classes=hps['N_M_CLASSES'],
            patch_shape=hps['PATCH_SIZE'],
            **model_config
        )
        # New training
        start_epoch = 1

        solver = Solver(evaluator=evaluator, save_path=run_dir, **hps_lower)

        # Final validation score is the value of interest
        final_val_score = solver.train(model=model,
                     training_set=training_set,
                     n_epochs=hps['N_EPOCHS'],
                     batch_size=hps['BATCH_SIZE'],
                     early_stop=hps['EARLY_STOP'],
                     eval_every=hps['EVAL_EVERY'],
                     start_epoch=start_epoch,
                     save_models=True)

        finish_wandb_run()

        if final_val_score > best_score or i == 0:
            best_score = final_val_score
            best_choice = choice

        results["All results"][str(choice)] = final_val_score

        # Save intermediate results
        intermediate_file = os.path.join(experiment_dir, 'intermediate_tuning_results.json')
        with open(intermediate_file, 'w') as f:
            json.dump(results, f)

    # Finalize
    results["Best score"] = best_score
    results["Best choice"] = best_choice
    results_file = os.path.join(experiment_dir, 'tuning_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f)
    trainLogger.info("Tuning results written to %s", results_file)

    trainLogger.info("Tuning finished.")

    return experiment_name

def get_all_possibilities(params_to_tune, hps, fine_tune):
    """ Get a dict for each possible configuration containing the parameter
    names as keys and the values of the parameters as values."""
    possibilities_per_param = {}
    for p in params_to_tune:
        if fine_tune:
            if p not in possible_fine_tune_values:
                raise RuntimeError(f"Parameter {p} unknown.")
            param_possibilities = possible_fine_tune_values[p](hps[p])
        else:
            if p not in possible_values:
                raise RuntimeError(f"Parameter {p} unknown.")
            param_possibilities = possible_values[p]()
        param_possibilities.sort()

        possibilities_per_param[p] = param_possibilities

    all_perms = create_permutations_of_param_choices(possibilities_per_param)

    # Convert nested parameters of the form x.y into a dict such that it can
    # be accessed as x[y]
    all_perms_new = []
    for perm in all_perms:
        perm_new = perm.copy()
        for k, v in perm.items():
            if "." in k: # nested parameter
                klist = k.split(".")
                perm_sub = perm_new
                for k_ in klist[:-1]:
                    if k_ not in perm_sub.keys():
                        perm_sub[k_] = {}
                    perm_sub = perm_sub[k_]
                perm_sub[klist[-1]] = v
                del perm_new[k]
        all_perms_new.append(perm_new)

    return all_perms_new

def get_lrs():
    possible_values_for_lr = [1e-3, 1e-4, 5e-5, 1e-5]

    return possible_values_for_lr

def get_mesh_loss_func_weights():
    n_losses = 5 # Should be equal to the number of losses used
    possible_values_per_weight = [1.0, 0.1, 0.01, 0.001]

    weights_list = create_permutations(n_losses, possible_values_per_weight)

    # Reduce possibilities by assuming that Chamfer/Wasserstein as well as edge
    # loss get weight 1. All other losses can only get lower weight.
    weights_list = [w for w in weights_list\
                    if(w[0] == 1.0 and w[-1] == 1.0
                       and all([w[0] > w_i for w_i in w[1:-1]]))]

    return weights_list

def get_mesh_loss_func_weights_fine(anchor_weights):
    """ Generate mesh loss weights for a fine-tuning procedure, i.e., starting
    from given anchor weights.
    """
    n_losses = len(anchor_weights)
    possible_values_per_weight = {}
    for i, w in enumerate(anchor_weights):
        w_ = float(w)
        # Keep chamfer and edge weight, tune all others
        if i == 0 or i == len(anchor_weights) - 1:
            possible_values_per_weight[f"w_{i}"] = [w]
        else:
            possible_values_per_weight[f"w_{i}"] = [
                w_ + 0.5 * w_, w_, w_ - 0.5 * w_
            ]

    weights_dict = create_permutations_of_param_choices(
        possible_values_per_weight
    )
    weights_list = [list(item.values()) for item in weights_dict]

    return weights_list

def get_voxel_loss_func_weights():
    n_losses = 1 # Should be equal to the number of losses used
    possible_values_per_weight = [1.0, 0.5, 0.1]

    weights_list = create_permutations(n_losses, possible_values_per_weight)

    return weights_list

def create_permutations_of_param_choices(params_and_possibilities: dict):
    """ Create a dict for each permutation of parameters such that every
    possibility to combine params_and_possibilities is covered.
    """
    k = list(params_and_possibilities)[-1]
    if len(params_and_possibilities) == 1:
        v = params_and_possibilities[k]
        perm = [{k: v_i} for v_i in v]
    else:
        perm = []
        v = params_and_possibilities.pop(k)
        subperm = create_permutations_of_param_choices(params_and_possibilities)
        for sub_p in subperm:
            sub_p_new = copy.deepcopy(sub_p)
            for v_i in v:
                sub_p_new[k] = v_i
                perm.append(copy.deepcopy(sub_p_new))
    return perm

def create_permutations(n_positions, possibilities_per_position):
    """ Create a list of all permutations from 'n_positions' where each has one
    of 'possibilities_per_position' values.
    """
    if n_positions == 1:
        perm = [[p] for p in possibilities_per_position]
    else:
        perm = []
        subperm = create_permutations(n_positions-1, possibilities_per_position)
        for sub_p in subperm:
            sub_p_new = copy.deepcopy(sub_p)
            for p in possibilities_per_position:
                perm.append(sub_p_new + [p])
    return perm

possible_values = {
    'MESH_LOSS_FUNC_WEIGHTS': get_mesh_loss_func_weights,
    'VOXEL_LOSS_FUNC_WEIGHTS': get_voxel_loss_func_weights,
    'OPTIM_PARAMS.lr': get_lrs,
    'OPTIM_PARAMS.graph_lr': get_lrs
}
possible_fine_tune_values = {
    'MESH_LOSS_FUNC_WEIGHTS': get_mesh_loss_func_weights_fine,
}
