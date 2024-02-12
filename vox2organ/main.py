#!/usr/bin/env python3

""" Main file """

import os
import sys
import json
import logging
import random
from argparse import ArgumentParser, RawTextHelpFormatter

# Temporarily filter warnings
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

import logger
from data.supported_datasets import (
    dataset_paths,
)
from params.default import hyper_ps_default
from params.groups import assemble_group_params, hyper_ps_groups
from utils.modes import ExecModes
from utils.utils import update_dict
from utils.train import training_routine
from utils.test import test_routine
from utils.train_test import train_test_routine
from utils.eval_metrics import *

log = logger.get_std_logger(__name__)

# Overwrite params for overfitting (often useful for debugging and development)
hyper_ps_overfit = {
    # Learning
    'SANITY_CHECK_DATA': True,
    'N_EPOCHS': 5,
    'BATCH_SIZE': 1,
}

# Parameters that are overwritten globally. For groups of parameters that are
# fixed together, see params.groups
hyper_ps_master = {
    # Overwrite single params here
}

def get_version(version_str):
    return float(".".join(sys.version.split(" ")[0].split(".")[:2]))

# Overwrite master params if the value is different from the
# default value
def ovwr(hyper_ps, key, value):
    if value != hyper_ps_default[key]:
        hyper_ps[key] = value

mode_handler = {
    ExecModes.TRAIN.value: training_routine,
    ExecModes.TEST.value: test_routine,
    ExecModes.TRAIN_TEST.value: train_test_routine,
}


def single_experiment(hyper_ps, mode, resume):
    """ Run a single experiment.
    """
    # Assemble params from default and group-specific params
    hps = assemble_group_params(hyper_ps['GROUP_NAME'])

    # Set dataset paths
    hps = update_dict(
        hps,
        dataset_paths[hyper_ps.get('DATASET', hps['DATASET'])]
    )

    # Overwrite with master params
    hps = update_dict(hps, hyper_ps)

    # Main device
    main_device = hps['DEVICE'] if (
        isinstance(hps['DEVICE'], str)
    ) else hps['DEVICE'][0]
    torch.cuda.set_device(main_device)

    # Potentially overfit
    if hps['OVERFIT']:
        hps = update_dict(hps, hyper_ps_overfit)

    # Set global loglevel
    logger.set_loglevel(getattr(logging, hps['LOGLEVEL']))

    # Set up experiment dir and create experiment name if not specified. If a
    # training is resumed, a model should only be tested, or the name is
    # 'debug', the experiment dir may already exist.
    exist_ok = (
        mode == ExecModes.TEST.value or
        resume or
        hps['EXPERIMENT_NAME'] == "debug"
    )
    hps['EXPERIMENT_NAME'],  hps['LOGLEVEL'] = logger.init_experiment(
        experiment_base_dir=hps['EXPERIMENT_BASE_DIR'],
        experiment_name=hps['EXPERIMENT_NAME'],
        prefix=hps['EXP_PREFIX'],
        exist_ok=exist_ok,
        create=True,
        log_time=hps['TIME_LOGGING'],
        log_level=hps['LOGLEVEL'],
    )

    # No wandb in debug runs
    if hps['EXPERIMENT_NAME'] == "debug":
        hps['USE_WANDB'] = False

    # Run
    routine = mode_handler[mode](hps, resume=resume)

    return hps['EXPERIMENT_NAME']

def main(hyper_ps):
    """
    Main function for training, validation, test
    """
    argparser = ArgumentParser(description="Vox2Cortex",
                               formatter_class=RawTextHelpFormatter)
    argparser.add_argument('--dataset',
                           type=str,
                           default=hyper_ps_default['DATASET'],
                           help="The name of the dataset.")
    argparser.add_argument('--master_port',
                           type=int,
                           default=hyper_ps_default['MASTER_PORT'],
                           help="The master port for GPU communication in DDP.")
    argparser.add_argument('--master_seed',
                           type=int,
                           default=hyper_ps_default['MASTER_SEED'],
                           help="The seed for torch, numpy etc. If None, no"
                           " seed is set.")
    argparser.add_argument('--train',
                           action='store_true',
                           help="Train a model.")
    argparser.add_argument('--test',
                           type=int,
                           default=hyper_ps_default['TEST_MODEL_EPOCH'],
                           nargs='?',
                           const=-1,
                           help="Test a model, optionally specified by epoch."
                           " If no epoch is specified, the best (w.r.t. IoU)"
                           " and the last model are evaluated.")
    argparser.add_argument('--resume',
                           action='store_true',
                           help="Resume an existing, potentially unfinished")
    argparser.add_argument('--n_epochs',
                           type=int,
                           default=hyper_ps_default['N_EPOCHS'],
                           help="The number of training epochs.")
    argparser.add_argument('--pretrained_model',
                           type=str,
                           default=hyper_ps_default['PRE_TRAINED_MODEL_PATH'],
                           nargs='?',
                           help="Specify the path to load pre-trained model weights.")
    argparser.add_argument('--log',
                           type=str,
                           dest='loglevel',
                           default=hyper_ps_default['LOGLEVEL'],
                           help="Specify log level.")
    argparser.add_argument('--no-wandb',
                           dest='use_wandb',
                           action='store_false',
                           help="Don't use wandb logging.")
    argparser.add_argument('--proj',
                           type=str,
                           dest='proj_name',
                           default=hyper_ps_default['PROJ_NAME'],
                           help="Specify the name of the wandb project.")
    argparser.add_argument('--group',
                           type=str,
                           dest='group_name',
                           nargs='+',
                           help="Specify the name(s) of the experiment"
                           " group(s)."
                           " Corresponding parameters are chosen from"
                           " params/groups.py")
    argparser.add_argument('--device',
                           type=str,
                           dest='device',
                           nargs='+',
                           default=hyper_ps_default['DEVICE'],
                           help="Specify the device(s) for execution.")
    argparser.add_argument('--overfit',
                           type=int,
                           nargs='?',
                           const=1, # Assume 1 sample without further spec.
                           default=hyper_ps_default['OVERFIT'],
                           help="Overfit on a few training samples.")
    argparser.add_argument('--time',
                           action='store_true',
                           help="Measure time of some functions.")
    argparser.add_argument('--exp_prefix',
                           type=str,
                           default=hyper_ps_default['EXP_PREFIX'],
                           help="A folder prefix for automatically enumerated"
                           " experiments.")
    argparser.add_argument('--experiment_base_dir',
                           type=str,
                           default=hyper_ps_default['EXPERIMENT_BASE_DIR'],
                           help="The base dir for experiments.")
    argparser.add_argument('-n', '--exp_name',
                           dest='exp_name',
                           type=str,
                           default=hyper_ps_default['EXPERIMENT_NAME'],
                           help="Name of experiment:\n"
                           "- 'debug' means that the results are  written "
                           "into a directory \nthat might be overwritten "
                           "later. This may be useful for debugging \n"
                           "where the experiment result does not matter.\n"
                           "- Any other name cannot overwrite an existing"
                           " directory.\n"
                           "- If not specified, experiments are automatically"
                           " enumerated with exp_i and stored in"
                           " the experiment_base_dir.")
    args = argparser.parse_args()

    ovwr(hyper_ps, 'EXPERIMENT_NAME', args.exp_name)
    ovwr(hyper_ps, 'EXPERIMENT_BASE_DIR', args.experiment_base_dir)
    ovwr(hyper_ps, 'DATASET', args.dataset)
    ovwr(hyper_ps, 'LOGLEVEL', args.loglevel)
    ovwr(hyper_ps, 'PROJ_NAME', args.proj_name)
    ovwr(hyper_ps, 'GROUP_NAME', args.group_name)
    ovwr(hyper_ps, 'DEVICE', args.device)
    ovwr(hyper_ps, 'OVERFIT', args.overfit)
    ovwr(hyper_ps, 'TIME_LOGGING', args.time)
    ovwr(hyper_ps, 'TEST_MODEL_EPOCH', args.test)
    ovwr(hyper_ps, 'EXP_PREFIX', args.exp_prefix)
    ovwr(hyper_ps, 'USE_WANDB', args.use_wandb)
    ovwr(hyper_ps, 'PRE_TRAINED_MODEL_PATH', args.pretrained_model)
    ovwr(hyper_ps, 'MASTER_PORT', args.master_port)
    ovwr(hyper_ps, 'MASTER_SEED', args.master_seed)
    ovwr(hyper_ps, 'N_EPOCHS', args.n_epochs)

    # Potentially set seed
    if args.master_seed is not None:
        log.info("Experiment with seed %d.", args.master_seed)
        torch.manual_seed(args.master_seed)
        np.random.seed(args.master_seed)
        random.seed(args.master_seed)

    if args.train and not args.test:
        mode = ExecModes.TRAIN.value
    if args.test and not args.train:
        mode = ExecModes.TEST.value
    if args.train and args.test:
        mode = ExecModes.TRAIN_TEST.value
    if not args.test and not args.train:
        log.error("Please use either --train or --test or both.")
        sys.exit(1)

    # Parameter group name(s)
    if mode == ExecModes.TEST.value:
        param_group_names = json.load(
            open(os.path.join(
                 args.experiment_base_dir,
                 args.exp_name,
                 logger.PARAM_FILE_NAME), 'r')
        )['GROUP_NAME']
    else:
        param_group_names = args.group_name if (
            args.group_name != hyper_ps_default['GROUP_NAME']
        ) else hyper_ps.get('GROUP_NAME', args.group_name)

    if isinstance(param_group_names, str):
        param_group_names = [param_group_names]

    if not all(n in hyper_ps_groups for n in param_group_names):
        log.error("Not all parameter groups exist.")
        sys.exit(1)

    previous_exp_name = None

    # Iterate over parameter groups
    for i, param_group_name in enumerate(param_group_names):
        # Potentially reference previous experiment. If the experiment is the
        # first in a row of experiments and 'PREVIOUS_EXPERIMENT_NAME' is set
        # in the master params, take this value.
        if 'PREVIOUS_EXPERIMENT_NAME' not in hyper_ps or i > 0:
            hyper_ps['PREVIOUS_EXPERIMENT_NAME'] = previous_exp_name
        hyper_ps['GROUP_NAME'] = param_group_name

        resume = args.resume if i == 0 else False

        previous_exp_name = single_experiment(hyper_ps, mode, resume)


if __name__ == '__main__':
    # Check version
    if get_version(sys.version) < 3.9:
        raise RuntimeError("Vox2Organ requires at least python version 3.9")
    # Run application
    main(hyper_ps_master)
