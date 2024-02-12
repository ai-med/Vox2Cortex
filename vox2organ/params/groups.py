""" Experiment-specific parameters. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from params.default import hyper_ps_default
from utils.utils import update_dict
from utils.losses import *
from utils.graph_conv import (
    GraphConvNorm,
    LinearLayer,
)
from utils.eval_metrics import *

# This dict contains groups of parameters that kind of belong together in order
# to conduct certain experiments
hyper_ps_groups = {


    ###### PARAMETER TUNING ######

    'TUNE_0': {
        'BASE_GROUP': 'UNetFlow',
        'VOXEL_LOSS_FUNC_WEIGHTS': [1.0],
    },


    # Vox2Cortex
    'Vox2Cortex': {
        'BASE_GROUP': None,
        'BATCH_SIZE': 1,
    },

    # V2C-Flow
    'V2C-Flow-S': {
        'BASE_GROUP': 'Vox2Cortex',
        'BATCH_SIZE': 1,
        'MESH_LOSS_FUNC': [
            ChamferLoss(curv_weight_max=5.0),
            EdgeLoss(0),
            NormalConsistencyLoss(),
        ],
        'MESH_LOSS_FUNC_WEIGHTS': [
            [1.0] * 4,  # Chamfer
            [1.0] * 4, # Edge
            [0.0001] * 4  # NC
        ],
        'MODEL_CONFIG': {
            # S2 I5
            'GRAPH_CHANNELS': [8, 64, 64],
            'N_EULER_STEPS': 5,
            # Selected aggregation to save memory
            'AGGREGATE_INDICES': [
                [3,4,5,6,7],
                [0,1,2,8,9,10],
            ],
        },
    },

    'V2C-Flow-S-small': {
        'BASE_GROUP': 'Vox2Cortex',
        'BATCH_SIZE': 1,
        'MESH_LOSS_FUNC': [
            ChamferLoss(curv_weight_max=5.0),
            EdgeLoss(0),
            NormalConsistencyLoss(),
        ],
        'MESH_LOSS_FUNC_WEIGHTS': [
            [1.0] * 4,  # Chamfer
            [1.0] * 4, # Edge
            [0.0001] * 4  # NC
        ],
        'MODEL_CONFIG': {
            # S2 I5
            'GRAPH_CHANNELS': [8, 64, 64],
            'N_EULER_STEPS': 5,
            # Selected aggregation to save memory
            'AGGREGATE_INDICES': [
                [3,4,5,6,7],
                [0,1,2,8,9,10],
            ],
        },
        'MESH_TEMPLATE_ID': 'fsaverage6-smooth-no-parc',
    },

    'V2C-Flow-F': {
        'BASE_GROUP': 'V2C-Flow-S',
        'MESH_TEMPLATE_ID': 'fsaverage-no-parc',
    },


    ### UNetFlow ###
    'UNetFlow': {
        'BASE_GROUP': None,
        'ARCHITECTURE': 'unetflow',
        'BATCH_SIZE': 2,
        'OPTIM_PARAMS': {
            'graph_lr': None,
        },
        'MODEL_CONFIG': {
            'NORM': 'instance',
            'N_VERTEX_CLASSES': 5,
            'N_EULER_STEPS': 5,
            'ENCODER_CHANNELS': [16, 32, 64, 128, 256],
            'DECODER_CHANNELS': [128, 64, 32, 16],
        },
        'MESH_LOSS_FUNC': [
            ChamferLoss(),  # Not curv. weighted
            EdgeLoss(0.0)
        ],
        'MESH_LOSS_FUNC_WEIGHTS': [
            [1.0] * 5,  # Chamfer
            [10.0] * 5  # Edge
        ],
        'MESH_TEMPLATE_ID': "abdomen-ct-1k",
        # For abdomen
        'PATCH_SIZE': [224, 224, 96],
        'SELECT_PATCH_SIZE': [224, 224, 96],
        'STRUCTURE_TYPE': "abdomen-all",
        'N_M_CLASSES': 5,
        'N_V_CLASSES': 5,  # 4 organs + background
    },
}


def assemble_group_params(group_name: str):
    """ Combine group params for a certain group name and potential base
    groups.
    """
    group_params = hyper_ps_groups[group_name]
    if group_params['BASE_GROUP'] is not None:
        base_params = assemble_group_params(group_params['BASE_GROUP'])
    else:
        base_params = hyper_ps_default

    return update_dict(base_params, group_params)
