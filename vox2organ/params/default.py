""" Documentation of project-wide parameters and default values

Ideally, all occurring parameters should be documented here.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import Enum

import torch

import logger
from utils.losses import *
from utils.graph_conv import (
    GraphConvNorm,
    SparseGraphConv,
)
from utils.eval_metrics import *


def hps_to_wandb(hps, job_type='train', notes=""):
    """ Convert local project params to wandb init arguments. """
    return {
        'name': hps['EXPERIMENT_NAME'],
        'dir': logger.get_log_dir(),
        'config': logger.string_dict(hps),
        'project': hps['PROJ_NAME'],
        'entity': hps['ENTITY'],
        'group': hps['BASE_GROUP'] if (
            hps['BASE_GROUP'] is not None
        ) else hps['GROUP_NAME'],
        'job_type': job_type,
        'notes': notes,
        'id': hps['EXPERIMENT_NAME'],
        'resume': "allow",
        'reinit': True,
    }


DATASET_SPLIT_PARAMS = (
    'DATASET_SEED',
    'DATASET_SPLIT_PROPORTIONS',
    'ALL_IDS_FILE',
    'FIXED_SPLIT',
    'OVERFIT'
)


DATASET_PARAMS = (
    'DATASET',
    'RAW_DATA_DIR',
)


hyper_ps_default = {

    # >>> Note: Using tuples (...) instead of lists [...] may lead to problems
    # when resuming broken trainings (json converts tuples to lists when dumping).
    # Therefore, it is recommended to use lists for parameters here.


    ### Paths ###

    # The directory where experiments are stored
    'EXPERIMENT_BASE_DIR': "../experiments/",


    ### Experiment description ###

    # The name of an experiment (=base folder for all data stored throughout
    # training and testing);
    # Attention: 'debug' overwrites previous debug experiment
    'EXPERIMENT_NAME': None,

    # Name of a previous experiment reference, e.g. to load a pre-trained model
    'PREVIOUS_EXPERIMENT_NAME': None,

    # A prefix for automatically enumerated experiments
    'EXP_PREFIX': 'exp_',

    # Username for wandb
    'WANDB_USER': 'fabibo3',

    # Project name used for wandb
    'PROJ_NAME': 'vox2organ',

    # Entity of wandb, e.g. group name
    'ENTITY': 'team-segmentation',

    # The loglevel for output logs
    'LOGLEVEL': 'INFO',

    # Whether to use wandb for logging
    'USE_WANDB': True,

    # Wandb logging group and/or parameter group
    'GROUP_NAME': 'uncategorized',

    # A base group for rarely-to-change parameters
    'BASE_GROUP': None,

    # The device(s) to train on; needs to be a list
    'DEVICE': ['cuda:0'],

    # Whether to do overfitting; this is helpful for debugging
    'OVERFIT': False,

    # If execution times should be measured for some functions
    'TIME_LOGGING': False,

    # Master port XXXXX for distributed training.
    # If training on a slurm node, make sure to use a different master port for
    # each run as problems might occur otherwise
    'MASTER_PORT': 29500,

    # Whether to register output meshes to voxel output
    'REGISTER_MESHES_TO_VOXELS': False,

    ### Model ###

    # The architecture to use
    'ARCHITECTURE': 'vox2cortex',

    # Template identifier, see utils.template
    'MESH_TEMPLATE_ID': "fsaverage-smooth-no-parc",

    # A path to a pre-trained (sub-) model
    'PRE_TRAINED_MODEL_PATH': None,

    # Model params
    'MODEL_CONFIG': {

        # UNet channels
        'ENCODER_CHANNELS': [8, 16, 32, 64, 128],
        'DECODER_CHANNELS': [64, 32, 16, 8], # Voxel decoder

        # Graph network channels
        'GRAPH_CHANNELS': [256, 64, 64, 64, 64],

        # Image channels
        'NUM_INPUT_CHANNELS': 1,

        # UNet deep supervision
        'DEEP_SUPERVISION': True,

        # The normalization (e.g. 'batch' for batch norm) to use in CNN and
        # GNN. Supported: 'batch', 'instance', 'layer'
        'NORM': 'batch',

        # Number of hidden layers in the graph conv blocks
        'N_F2F_HIDDEN_LAYER': 2,

        # Number of residual blocks in a GNN deformation step
        'N_RESIDUAL_BLOCKS': 3,

        # Whether to use a voxel decoder
        'VOXEL_DECODER': True,

        # The graph conv implementation to use
        'GC': SparseGraphConv,

        # Dropout probability of UNet blocks
        'P_DROPOUT_UNET': None,

        # Dropout probability of graph conv blocks
        'P_DROPOUT_GRAPH': None,

        # The mechanism for voxel feature aggregations, can be 'trilinear',
        # 'bilinear', or 'lns'
        'AGGREGATE': 'trilinear',

        # Where to take the features from the UNet
        'AGGREGATE_INDICES': [[4,5,6,7],[3,4,7,8],[2,3,8,9],[1,2,8,9]],

        # The number of vertex classes (i.e., the length of the vertex feature
        # vector in the template)
        'N_VERTEX_CLASSES': 2,

        # The number of steps in the forward Euler integration (1/h with h the
        # step size).
        'N_EULER_STEPS': 1,

        # The black box ODE solver to apply in the graph network. Supported are
        # "Euler" or "Midpoint"
        'ODE_SOLVER': "Euler",
    },


    ### Data ###

    # Directory of raw data, usually preprocessed from 'FS_DIR'
    'RAW_DATA_DIR': "/raw/data/dir", # <<<< Needs to set (e.g. in main.py)

    # The dataset to use, see data.supported_datasets
    'DATASET': 'ADNI_CSR_large',

    # Whether to resample training targets (points etc.) every epoch
    'RESAMPLE_TARGETS': True,

    # Whether to use subsampled cortex meshes during training. Note that this
    # only affects training shapes in the cortex.
    'REDUCED_GT': True,

    # Whether to use registered/resampled meshes as ground truth (V2CC)
    'REGISTERED_GT_MESHES': False,

    # The number of points sampled per mesh
    'N_REF_POINTS_PER_STRUCTURE': 100000,

    # The number of image dimensions. This parameter is deprecated since
    # dimensionality is now inferred from the patch size.
    'NDIMS': 3,

    # The number of voxel classes to distinguish (including background)
    'N_V_CLASSES': 3,

    # The number of mesh classes. This is usually the number of connected
    # components/structures/organs
    'N_M_CLASSES': 4,

    # The structure type. This is a generic name that identifies the labels and
    # meshes to use, see function _get_seg_and_mesh_label_names in the datasets
    'STRUCTURE_TYPE': "cortex-all",

    # Check if data has been transformed correctly. This leads potentially to a
    # larger memory consumption since meshes are voxelized and voxel labels are
    # loaded (even though only one of them is typically used)
    'SANITY_CHECK_DATA': False,

    # Choose either 'voxelized_meshes' or 'voxel_seg' segmentation ground truth
    # labels
    'SEG_GROUND_TRUTH': 'voxel_seg',

    # Data augmentation
    'AUGMENT_TRAIN': False,

    # Final image size. From the raw input images, a patch of size
    # 'SELECT_PATCH_SIZE' is cut out and subsequently zoomed to 'PATCH_SIZE'
    'PATCH_SIZE': [192, 208, 192],

    # Size of cut out image patch
    'SELECT_PATCH_SIZE': [192, 208, 192],

    # Seed for torch, numpy etc. If None, no seed is used.
    'MASTER_SEED': None,

    # Seed for dataset splitting, only relevant if 'FIXED_SPLIT' is None
    'DATASET_SEED': 1234,

    # File containing all image ids
    'ALL_IDS_FILE': "all_ids.txt",

    # Proportions of dataset splits, only relevant if 'FIXED_SPLIT' is None
    'DATASET_SPLIT_PROPORTIONS': [80, 10, 10],

    # Dict or bool value that allows for specifying fixed ids for dataset
    # splitting.
    # If specified, 'DATASET_SEED' and 'DATASET_SPLIT_PROPORTIONS' will be
    # ignored. The dict should contain values for keys 'train', 'validation',
    # and 'test'. Alternatively, a list of files can be specified containing
    # IDs for 'train', 'validation', and 'test'
    'FIXED_SPLIT': None,


    ### Evaluation ###

    # The metrics used for evaluation, see utils.evaluate.EvalMetrics for
    # options
    'EVAL_METRICS': [
        SurfaceDistance(),
        SelfIntersections(),
        VoxelDice(),
        # MeshDice(),
    ],

    # Main validation metric according to which the best model is determined.
    'MAIN_EVAL_METRIC': 'ASSD',

    # For testing the model from a certain training epoch; if None, the best
    # model is used
    'TEST_MODEL_EPOCH': None,

    # Either 'test' or 'validation' (or 'train' if desired)
    'TEST_SPLIT': 'test',


    ### Learning ###

    # Lr scheduler
    'SCHEDULER_CLASS': torch.optim.lr_scheduler.CyclicLR,

    # Freeze pre-trained model parameters
    'FREEZE_PRE_TRAINED': False,

    # The batch size used during training
    'BATCH_SIZE': 1,

    # Optionally provide a norm for gradient clipping
    'CLIP_GRADIENT': 200000,

    # Accumulate n gradients before doing a backward pass
    'ACCUMULATE_N_GRADIENTS': 1,

    # The number of training epochs
    'N_EPOCHS': 500,

    # The optimizer used for training
    'OPTIMIZER_CLASS': torch.optim.AdamW,

    # Parameters for the optimizer. A separate learning rate for the graph
    # network can be specified
    'OPTIM_PARAMS': {
        'lr': 1e-4,  # voxel lr
        'graph_lr': 5e-5,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'weight_decay': 0.0001
    },

    # Parameters for lr scheduler
    'LR_SCHEDULER_PARAMS': {
        'cycle_momentum': False,
        'gamma_plateau': 0.5,
        'patience': 20,
    },

    # Whether or not to use Pytorch's automatic mixed precision
    'MIXED_PRECISION': True,

    # The used loss functions for the voxel segmentation
    'VOXEL_LOSS_FUNC': [torch.nn.CrossEntropyLoss()],

    # The weights for the voxel loss functions
    'VOXEL_LOSS_FUNC_WEIGHTS': [1.0],

    # The used loss functions for the mesh
    'MESH_LOSS_FUNC': [
        ChamferAndNormalsLoss(curv_weight_max=5.0),
        LaplacianDeformationFieldLoss(),
        NormalConsistencyLoss(),
        EdgeLoss(0.0)
    ],

    # The weights for the mesh loss functions
    # Order of structures: lh_white, rh_white, lh_pial, rh_pial; mesh loss
    # weights should respect this order!
    'MESH_LOSS_FUNC_WEIGHTS': [
        [4.0] * 4,  # Chamfer
        [0.01] * 2 + [0.0125] * 2,  # Cosine,
        [0.1] * 2 + [0.25] * 2,  # Laplace,
        [0.001] * 2 + [0.00225] * 2,  # NormalConsistency
        [5.0] * 4  # Edge
    ],

    # Log losses etc. every n iterations or 'epoch'
    'LOG_EVERY': 'epoch',

    # Evaluate model every n epochs
    'EVAL_EVERY': 10,
}
