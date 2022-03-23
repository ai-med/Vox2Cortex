#!/usr/bin/env python3

""" Main file """

import os
from argparse import ArgumentParser, RawTextHelpFormatter

import torch

from data.supported_datasets import (
    dataset_paths,
)
from utils.params import (
    hyper_ps_default,
    CHECK_DIR,
    MISC_DIR,
)
from utils.modes import ExecModes
from utils.utils import update_dict
from utils.train import training_routine
from utils.tune_params import tuning_routine
from utils.test import test_routine
from utils.train_test import train_test_routine
from utils.ablation_study import (
    AVAILABLE_ABLATIONS,
    set_ablation_params_
)

# Overwrite default parameters for a common training procedure
hyper_ps = {
    # Overwriting std values from utils.params
    #######################
    'EXPERIMENT_NAME': None,  # Attention: "debug" overwrites previous dir"
                              # should be set with console argument
    #######################

    # Data
    'SANITY_CHECK_DATA': False, # Save some memory

    # Learning
    'N_EPOCHS': 100,
    'BATCH_SIZE': 2,
    'EVAL_EVERY': 2,

    # Evaluation
    'EVAL_METRICS': [
        'SymmetricHausdorff',
        'JaccardVoxel',
        'JaccardMesh',
        'Chamfer',
        'CorticalThicknessError',
        'AverageDistance'
    ],

    # Template
    'N_TEMPLATE_VERTICES': 42016,
}

# Overwrite params for overfitting (often useful for debugging and development)
hyper_ps_overfit = {
    # Learning
    'BATCH_SIZE': 1,
    'SANITY_CHECK_DATA': True
}

mode_handler = {
    ExecModes.TRAIN.value: training_routine,
    ExecModes.TEST.value: test_routine,
    ExecModes.TRAIN_TEST.value: train_test_routine,
    ExecModes.TUNE.value: tuning_routine
}

def main(hps):
    """
    Main function for training, validation, test
    """
    argparser = ArgumentParser(description="Vox2Cortex",
                               formatter_class=RawTextHelpFormatter)
    argparser.add_argument('--architecture',
                           type=str,
                           default="vox2cortex",
                           help="The name of the algorithm. Supported:\n"
                           "- vox2cortex")
    argparser.add_argument('--dataset',
                           type=str,
                           default="Not specified",
                           help="The name of the dataset.")
    argparser.add_argument('--train',
                           action='store_true',
                           help="Train a model.")
    argparser.add_argument('--test',
                           type=int,
                           default=None,
                           nargs='?',
                           const=-1,
                           help="Test a model, optionally specified by epoch."
                           " If no epoch is specified, the best (w.r.t. IoU)"
                           " and the last model are evaluated.")
    argparser.add_argument('--tune',
                           default=None,
                           type=str,
                           dest='params_to_tune',
                           nargs='+',
                           help="Specify the name of a parameter to tune.")
    argparser.add_argument('--fine-tune',
                           default=None,
                           type=str,
                           dest='params_to_fine_tune',
                           nargs='+',
                           help="Specify the name of a parameter to tune.")
    argparser.add_argument('--resume',
                           action='store_true',
                           help="Resume an existing, potentially unfinished"
                           " experiment.")
    argparser.add_argument('--log',
                           type=str,
                           dest='loglevel',
                           default='INFO',
                           help="Specify log level.")
    argparser.add_argument('--proj',
                           type=str,
                           dest='proj_name',
                           default="vox2cortex",
                           help="Specify the name of the wandb project.")
    argparser.add_argument('--group',
                           type=str,
                           dest='group_name',
                           default='uncategorized',
                           help="Specify the name of the wandb group.")
    argparser.add_argument('--device',
                           type=str,
                           dest='device',
                           default='cuda:0',
                           help="Specify the device for execution.")
    argparser.add_argument('--overfit',
                           type=int,
                           nargs='?',
                           const=1,
                           default=False,
                           help="Overfit on a few training samples.")
    argparser.add_argument('--time',
                           action='store_true',
                           help="Measure time of some functions.")
    argparser.add_argument('--n_test_vertices',
                           type=int,
                           default=-1,
                           help="Set the number of template vertices during"
                           " testing.")
    argparser.add_argument('--ablation_study',
                           type=str,
                           nargs=1,
                           help="Perform an ablation study."
                           f"Available options are: {AVAILABLE_ABLATIONS}")
    argparser.add_argument('-n', '--exp_name',
                           dest='exp_name',
                           type=str,
                           default=None,
                           help="Name of experiment:\n"
                           "- 'debug' means that the results are  written "
                           "into a directory \nthat might be overwritten "
                           "later. This may be useful for debugging \n"
                           "where the experiment result does not matter.\n"
                           "- Any other name cannot overwrite an existing"
                           " directory.\n"
                           "- If not specified, experiments are automatically"
                           " enumerated with exp_i and stored in"
                           " ../experiments.")
    args = argparser.parse_args()
    hps['EXPERIMENT_NAME'] = args.exp_name
    hps['ARCHITECTURE'] = args.architecture
    hps['DATASET'] = args.dataset
    hps['LOGLEVEL'] = args.loglevel
    hps['PROJ_NAME'] = args.proj_name
    hps['GROUP_NAME'] = args.group_name
    hps['DEVICE'] = args.device
    hps['OVERFIT'] = args.overfit
    hps['TIME_LOGGING'] = args.time
    hps['PARAMS_TO_TUNE'] = args.params_to_tune
    hps['PARAMS_TO_FINE_TUNE'] = args.params_to_fine_tune
    hps['TEST_MODEL_EPOCH'] = args.test
    hps['N_TEMPLATE_VERTICES_TEST'] = args.n_test_vertices
    if args.ablation_study:
        hps['ABLATION_STUDY'] = args.ablation_study[0]
    else:
        hps['ABLATION_STUDY'] = False

    if args.params_to_tune and args.params_to_fine_tune:
        raise RuntimeError(
            "Cannot tune and fine-tune parameters at the same time."
        )

    torch.cuda.set_device(args.device)

    # Fill hyperparameters with defaults
    hps = update_dict(hyper_ps_default, hps)

    # Automatically choose template
    hps['MODEL_CONFIG']['MESH_TEMPLATE'] = os.path.join(
        hps['TEMPLATE_PATH'],
        hps['TEMPLATE_NAME'](
            hps['N_TEMPLATE_VERTICES'],
            hps['SELECT_PATCH_SIZE'],
            hps['PATCH_SIZE'])
    )

    # Set dataset paths
    update_dict(hps, dataset_paths[args.dataset])

    # Update again for overfitting
    if hps['OVERFIT']:
        hps = update_dict(hps, hyper_ps_overfit)

    # Set the number of test vertices
    if hps['N_TEMPLATE_VERTICES_TEST'] == -1:
        hps['N_TEMPLATE_VERTICES_TEST'] = hps['N_TEMPLATE_VERTICES']

    if args.params_to_tune or args.params_to_fine_tune:
        mode = ExecModes.TUNE
    else:
        if args.train and not args.test:
            mode = ExecModes.TRAIN.value
        if args.test and not args.train:
            mode = ExecModes.TEST.value
        if args.train and args.test:
            mode = ExecModes.TRAIN_TEST.value
        if not args.test and not args.train:
            print("Please use either --train or --test or both.")
            return

    # Set params for ablation study
    if args.ablation_study:
        set_ablation_params_(hps, args.ablation_study[0])

    # No voxel decoder --> set voxel loss weights to 0
    if not hps['MODEL_CONFIG']['VOXEL_DECODER']:
        hps['VOXEL_LOSS_FUNC_WEIGHTS'] = []
        hps['VOXEL_LOSS_FUNC'] = []
        if 'JaccardVoxel' in hps['EVAL_METRICS']:
            hps['EVAL_METRICS'].remove('JaccardVoxel')

    # Add patch size to model config
    hps['MODEL_CONFIG']['PATCH_SIZE'] = hps['PATCH_SIZE']

    # Create output dirs
    if not os.path.isdir(MISC_DIR):
        os.mkdir(MISC_DIR)
    if not os.path.isdir(CHECK_DIR):
        os.mkdir(CHECK_DIR)

    # Run
    routine = mode_handler[mode]
    routine(hps, experiment_name=hps['EXPERIMENT_NAME'],
            loglevel=hps['LOGLEVEL'], resume=args.resume)


if __name__ == '__main__':
    main(hyper_ps)
