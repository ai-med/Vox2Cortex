""" Vox2Cortex """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from itertools import chain
from typing import Union, Tuple, Sequence
from deprecated import deprecated

import torch
import torch.nn as nn
from pytorch3d.structures import MeshesXD

import logger
from utils.mesh import vff_to_Meshes, verts_faces_to_Meshes
from models.u_net import ResidualUNet
from models.base_model import V2MModel
from models.graph_net import GraphDecoder

log = logger.get_std_logger(__name__)

class Vox2Cortex(V2MModel):
    """ Implementation of the Vox2Cortex model.

    :param n_v_classes: Number of voxel classes to distinguish
    :param n_m_classes: Number of mesh classes to distinguish
    :param num_input_channels: The number of channels of the input image.
    :param encoder_channels: The number of channels of the encoder
    :param decoder_channels: The number of channels of the decoder
    :param graph_channels: The number of graph features per graph layer
    :param norm: The normalization to apply. Supported: 'batch', 'instance',
    'layer'
    :param voxel_decoder: Whether or not to use a voxel decoder
    :param GC: The graph conv implementation to use
    :param patch_size: The used patch size of input images.
    :param aggregate_indices: Where to take the features from the UNet
    :param aggregate: 'trilinear', 'bilinear', or 'lns'
    :param p_dropout_unet: Dropout probability for UNet blocks
    :param p_dropout_graph: Dropout probability for  graph conv blocks
    :param ndims: Dimensionality of images
    group left and right white matter hemisphere into group "white matter".
    During a graph net forward pass, features are exchanged between distinct
    groups but not within a group. For example, white surface vertex positions
    can be provided to the pial vertices and vice versa.
    :param n_vertex_classes: The number of vertex classes.
    :param n_euler_steps: The number of integration steps (1/h)
    :param n_f2f_hidden_layer: The number of GNN hidden layers in a residual
    block.
    :param n_residual_blocks: The number of residual GNN blocks in a
    deformation step.
    """

    def __init__(self,
                 n_v_classes: int,
                 n_m_classes: int,
                 num_input_channels: int,
                 encoder_channels: Union[list, tuple],
                 decoder_channels: Union[list, tuple],
                 graph_channels: Union[list, tuple],
                 norm: str,
                 deep_supervision: bool,
                 voxel_decoder: bool,
                 gc,
                 patch_size: Tuple[int, int, int],
                 aggregate: str,
                 aggregate_indices: Tuple[Tuple[int]],
                 p_dropout_unet: float,
                 p_dropout_graph: float,
                 ndims: int,
                 n_vertex_classes: int,
                 n_euler_steps: float,
                 n_f2f_hidden_layer: int,
                 n_residual_blocks: int,
                 ode_solver: str,
                 **kwargs
                 ):
        super().__init__()

        log.debug('n voxel classes {}'.format(n_v_classes))
        log.debug('n vertex classes {}'.format(n_vertex_classes))

        # The channel sizes of the UNet
        skip_channels = (
            [num_input_channels] + # Input image
            encoder_channels + # Enoder channels
            decoder_channels + # Decoder channels
            [n_v_classes] # Segmentation output
        )

        # The number of mesh deformation stages
        self.deform_stages = len(graph_channels) - 1

        # Voxel network
        self.voxel_net = ResidualUNet(
            num_classes=n_v_classes,
            num_input_channels=num_input_channels,
            patch_shape=patch_size,
            down_channels=encoder_channels,
            up_channels=decoder_channels,
            deep_supervision=deep_supervision,
            voxel_decoder=voxel_decoder,
            p_dropout=p_dropout_unet,
            ndims=ndims,
            norm=norm,
        )

        # Graph network
        self.graph_net = GraphDecoder(
            norm=norm,
            graph_channels=graph_channels,
            skip_channels=skip_channels,
            patch_size=patch_size,
            aggregate_indices=aggregate_indices,
            aggregate=aggregate,
            GC=gc,
            p_dropout=p_dropout_graph,
            n_vertex_classes=n_vertex_classes,
            ndims=ndims,
            n_euler_steps=n_euler_steps,
            n_f2f_hidden_layer=n_f2f_hidden_layer,
            n_residual_blocks=n_residual_blocks,
            ode_solver=ode_solver,
        )

        # Vox2Cortex has a fixed batch size associated to the model as the
        # edges in the sparse graph convs are fixed. This is set during the
        # first ever forward pass or when loading the state dict
        self._batch_size = nn.Parameter(
            torch.zeros(1).long(),
            requires_grad=False
        )

    @property
    def batch_size(self):
        if self._batch_size == 0:
            raise ValueError(
                "Batch size has not yet been set. This means usually that"
                " the model has not completed any forward pass."
            )
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if self._batch_size > 0:
            raise RuntimeError("Batch size can only be set once.")
        if value <= 0:
            raise ValueError("Batch size needs to be > 0.")
        self._batch_size = nn.Parameter(
            torch.tensor([value]).long(),
            requires_grad=False
        )

    @logger.measure_time
    def forward(self, x: torch.Tensor, input_meshes: MeshesXD) -> tuple:
        """ Forward pass. """
        encoder_skips, decoder_skips, seg_out = self.voxel_net(x)
        pred_meshes = self.graph_net(
            encoder_skips + decoder_skips, input_meshes
        )

        # Batch size is fixed after first ever forward pass
        try:
            self.batch_size = x.shape[0]
            log.info("Batch size set to %d", self.batch_size)
        except RuntimeError:
            pass

        # pred has the form
        #   - batch of predicted meshes
        #   - batch of voxel predictions,
        pred = (pred_meshes, seg_out)

        return pred

    def save(self, path):
        """ Save model with its parameters to the given path.
        Conventionally the path should end with "*.model".

        :param str path: The path where the model should be saved.
        """

        torch.save(self.state_dict(), path)

    @staticmethod
    def pred_to_voxel_pred(pred) -> torch.Tensor:
        """ Get the final voxel prediction with argmax over classes applied """
        if pred[1]:
            return pred[1][-1].argmax(dim=1)
        return None

    @staticmethod
    def pred_to_raw_voxel_pred(pred) -> Union[torch.Tensor, Sequence]:
        """ Get the voxel prediction per class. May be a list if deep
        supervision is used. """
        return pred[1]

    @staticmethod
    def pred_to_pred_meshes(pred) -> MeshesXD:
        """ Get the predicted meshes. """
        return pred[0]

    @staticmethod
    def pred_to_final_mesh_pred(pred) -> MeshesXD:
        """ Get the final mesh prediction. In contrast to 'pred_to_pred_meshes'
        this does not involve intermediate deformation stages. """
        mesh_pred = pred[0]
        S, B, C = mesh_pred.X_dims()
        final_meshpred = MeshesXD(
            mesh_pred.verts_list()[-B*C:],
            mesh_pred.faces_list()[-B*C:],
            X_dims=(B, C),
            verts_features=mesh_pred.verts_features_list()[-B*C:]
        )
        return final_meshpred
