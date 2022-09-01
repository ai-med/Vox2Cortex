""" Documentation of project-wide parameters and default values

Ideally, all occurring parameters should be documented here.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from enum import Enum

import torch

from utils.losses import (
    ChamferAndNormalsLoss,
    LaplacianLoss,
    NormalConsistencyLoss,
    EdgeLoss
)
from utils.utils_vox2cortex.graph_conv import (
    GraphConvNorm
)

DATASET_SPLIT_PARAMS = (
    'DATASET_SEED',
    'DATASET_SPLIT_PROPORTIONS',
    'FIXED_SPLIT',
    'OVERFIT'
)

DATASET_PARAMS = (
    'DATASET',
    'RAW_DATA_DIR',
    'PREPROCESSED_DATA_DIR'
)

# Directory for output to check
CHECK_DIR = "../to_check"

# Miscellanous output
MISC_DIR = "../misc"

hyper_ps_default={

    # >>> Note: Using tuples (...) instead of lists [...] may lead to problems
    # when resuming broken trainings (json converts tuples to lists when dumping).
    # Therefore, it is recommended to use lists for parameters here.

    # The path where templates are stored
    'TEMPLATE_PATH': "../supplementary_material/templates/",

    # The template name in dependence of the number of vertices N,
    # 'SELECT_PATH_SIZE' (sps) and 'PATCH_SIZE' (ps)
    'TEMPLATE_NAME': (
        lambda N, sps, ps: f"cortex_4_smoothed_{N}_sps{sps}_ps{ps}.obj"
    ),

    # The number of vertex classes to distinguish (including background)
    'N_V_CLASSES': 2,

    # The wandb project name
    'PROJ_NAME': "cortex",

    # The number of mesh classes. This is usually the number of non-connected
    # components/structures
    'N_M_CLASSES': 4,

    # The number of vertices in a single template structure
    'N_TEMPLATE_VERTICES': 162,

    # The number of vertices in a single template structure used during testing
    # (may be different than 'N_TEMPLATE_VERTICES'; -1 means that
    # 'N_TEMPLATE_VERTICES' is used)
    'N_TEMPLATE_VERTICES_TEST': -1,

    # The number of reference points in a cortex structure
    'N_REF_POINTS_PER_STRUCTURE': 40962,

    # Either use a mesh or a pointcloud as ground truth. Basically, if one
    # wants to compute only point losses like the Chamfer loss, a pointcloud is
    # sufficient while other losses like cosine distance between vertex normals
    # require a mesh (pointcloud + faces)
    'MESH_TARGET_TYPE': "mesh",

    # The type of meshes used, either 'freesurfer' or 'marching cubes'
    'MESH_TYPE': 'freesurfer',

    # The mode for reduction of mesh regularization losses, either 'linear' or
    # 'none'
    'REDUCE_REG_LOSS_MODE': 'none',

    # The structure type for cortex data, either 'cerebral_cortex' or
    # 'white_matter'
    'STRUCTURE_TYPE': ['white_matter', 'cerebral_cortex'],

    # Check if data has been transformed correctly. This leads potentially to a
    # larger memory consumption since meshes are voxelized and voxel labels are
    # loaded (even though only one of them is typically used)
    'SANITY_CHECK_DATA': True,

    # The batch size used during training
    'BATCH_SIZE': 1,

    # Optionally provide a norm for gradient clipping
    'CLIP_GRADIENT': 200000,

    # Activate/deactivate patch mode for the cortex dataset. Possible values
    # are "no", "single-patch", "multi-patch"
    'PATCH_MODE': "no",

    # Accumulate n gradients before doing a backward pass
    'ACCUMULATE_N_GRADIENTS': 1,

    # The number of training epochs
    'N_EPOCHS': 5,

    # Freesurfer ground truth meshes with reduced resolution. 1.0 = original
    # resolution (in terms of number of vertices)
    'REDUCED_FREESURFER': 0.3,

    # Choose either 'voxelized_meshes' or 'aseg' segmentation ground truth
    # labels
    'SEG_GROUND_TRUTH': 'voxelized_meshes',

    # Whether to use curvatures of the meshes. If set to True, the ground truth
    # points are vertices and not sampled surface points
    'PROVIDE_CURVATURES': True,

    # The optimizer used for training
    'OPTIMIZER_CLASS': torch.optim.Adam,

    # Parameters for the optimizer. A separate learning rate for the graph
    # network can be specified
    'OPTIM_PARAMS': {
        'lr': 1e-4, # voxel lr
        'graph_lr': 5e-5,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'weight_decay': 0.0
    },

    # Data augmentation
    'AUGMENT_TRAIN': False,

    # Whether or not to use Pytorch's automatic mixed precision
    'MIXED_PRECISION': True,

    # The used loss functions for the voxel segmentation
    'VOXEL_LOSS_FUNC': [torch.nn.CrossEntropyLoss()],

    # The weights for the voxel loss functions
    'VOXEL_LOSS_FUNC_WEIGHTS': [1.], # BCE

    # The used loss functions for the mesh
    'MESH_LOSS_FUNC': [
       ChamferAndNormalsLoss(curv_weight_max=5.0),
       LaplacianLoss(),
       NormalConsistencyLoss(),
       EdgeLoss(0.0)
    ],

    # The weights for the mesh loss functions.
    # Order of structures: lh_white, rh_white, lh_pial, rh_pial; mesh loss
    # weights should respect this order!
    'MESH_LOSS_FUNC_WEIGHTS': [
        [1.0] * 4, # Chamfer
        [0.01] * 2 + [0.0125] * 2, # Cosine,
        [0.1] * 2 + [0.25] * 2, # Laplace,
        [0.001] * 2 + [0.00225] * 2, # NormalConsistency
        [5.0] * 4 # Edge
    ],

    # Penalize large vertex displacements, can be seen as a regularization loss
    # function weight
    'PENALIZE_DISPLACEMENT': 0.0,

    # Has no effect if vertices are used as ground truth, see N_REF_POINTS_PER_STRUCTURE
    'N_SAMPLE_POINTS': None,

    # The way the weighted average of the losses is computed,
    # e.g. 'linear' weighted average, 'geometric' mean
    'LOSS_AVERAGING': 'linear',

    # Log losses etc. every n iterations or 'epoch'
    'LOG_EVERY': 'epoch',

    # Evaluate model every n epochs
    'EVAL_EVERY': 1,

    # Use early stopping
    'EARLY_STOP': False,

    # The metrics used for evaluation, see utils.evaluate.EvalMetrics for
    # options
    'EVAL_METRICS': [
        'Wasserstein',
        'SymmetricHausdorff',
        'JaccardVoxel',
        'JaccardMesh',
        'Chamfer'
    ],

    # Main validation metric according to which the best model is determined.
    # Note: This one must also be part of 'EVAL_METRICS'!
    'MAIN_EVAL_METRIC': 'JaccardMesh',

    # The number of image dimensions. This parameter is deprecated since
    # dimensionality is now inferred from the patch size.
    'NDIMS': 3,

    # Set of model parameters
    'MODEL_CONFIG': {
        'FIRST_LAYER_CHANNELS': 16,
        'ENCODER_CHANNELS': [16, 32, 64, 128, 256],
        'DECODER_CHANNELS': [64, 32, 16, 8],
        'GRAPH_CHANNELS': [256, 64, 64, 64, 64],
        'NUM_INPUT_CHANNELS': 1,
        'STEPS': 4,
        'DEEP_SUPERVISION': True,
        # Only for graph convs, in UNet batch norm is not affected by this parameter
        'NORM': 'batch',
        # Number of hidden layers in the graph conv blocks
        'GRAPH_CONV_LAYER_COUNT': 4,
        'MESH_TEMPLATE': '../supplementary_material/spheres/icosahedron_162.obj',
        'UNPOOL_INDICES': [0,0,0,0],
        'USE_ADOPTIVE_UNPOOL': False,
        # Weighted feature aggregation in graph convs (only possible with
        # pytorch-geometric graph convs)
        'WEIGHTED_EDGES': False,
        # Whether to use a voxel decoder
        'VOXEL_DECODER': True,
        # The graph conv implementation to use
        'GC': GraphConvNorm,
        # Whether to propagate coordinates in the graph decoder in addition to
        # voxel features
        'PROPAGATE_COORDS': True,
        # Dropout probability of UNet blocks
        'P_DROPOUT': None,
        # The used patch size, should be equal to global patch size
        'PATCH_SIZE': [128, 144, 128],
        # The ids of structures that should be grouped in the graph net.
        # Example: if lh_white and rh_white have ids 0 and 1 and lh_pial and
        # rh_pial have ids 2 and 3, then the groups should be specified as
        # ((0,1),(2,3))
        'GROUP_STRUCTS': [[0, 1], [2, 3]], # False for single-surface reconstruction
        # Whether to exchange coordinates between groups
        'EXCHANGE_COORDS': True,
        # The number of neighbors considered for feature aggregation from
        # vertices of different structures in the graph net
        'K_STRUCT_NEIGHBORS': 5,
        # The mechanism for voxel feature aggregations, can be 'trilinear',
        # 'bilinear', or 'lns'
        'AGGREGATE': 'trilinear',
        # Where to take the features from the UNet
        'AGGREGATE_INDICES': [
            [3,4,5,6],
            [2,3,6,7],
            [1,2,7,8],
            [0,1,7,8] # 8 = last decoder skip
        ],
    },

    # Decay the learning rate by multiplication with 'LR_DECAY_RATE' if no
    # improvement for 'LR_DECAY_AFTER' epochs. Even though it is set, we
    # usually stop training before this has an effect.
    'LR_DECAY_RATE': 0.5,
    'LR_DECAY_AFTER': 30, # has usually no impact

    # Patch size of the images
    'PATCH_SIZE': [128, 144, 128],

    # This is the size of selected patches before potential downsampling
    'SELECT_PATCH_SIZE': [192, 208, 192],

    # Seed for dataset splitting
    'DATASET_SEED': 1234,

    # Proportions of dataset splits
    'DATASET_SPLIT_PROPORTIONS': [80, 10, 10],

    # Dict or bool value that allows for specifying fixed ids for dataset
    # splitting.
    # If specified, 'DATASET_SEED' and 'DATASET_SPLIT_PROPORTIONS' will be
    # ignored. The dict should contain values for keys 'train', 'validation',
    # and 'test'. Alternatively, a list of files can be specified containing
    # IDs for 'train', 'validation', and 'test'
    'FIXED_SPLIT': False,

    # The directory where experiments are stored
    'EXPERIMENT_BASE_DIR': "../experiments/",

    # Directory of raw data
    'RAW_DATA_DIR': "/raw/data/dir", # <<<< Needs to set (e.g. in main.py)

    # Directory of preprocessed data, e.g., containing thickness values from
    # FreeSurfer
    'PREPROCESSED_DATA_DIR': "/preprocessed/data/dir", # <<<< Needs to set (e.g. in main.py)
}
