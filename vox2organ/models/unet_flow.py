""" Implementation of a new UNet-based deformation network.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from typing import Sequence

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from pytorch3d.structures import MeshesXD

import logger
from models.base_model import V2MModel
from models.u_net import ResidualUNet
from utils.feature_aggregation import (
    aggregate_trilinear,
)


# No autocast for grid sample
def apply_flow(discrete_flow, V0, X_dims, n_integration):
    """ Apply a discrete flow field to V0 points by interpolation. """
    # Flow has shape BxCxHxWxD
    ndims = V0[0].shape[-1]

    # Stepsize
    h = 1 / float(n_integration)

    # Number of different organs/meshes
    B, C = X_dims

    VT = [None] * B * C

    # Iterate over surfaces
    for c in range(C):
        # Verts of this mesh in every sample of the batch
        ids = [x * C + c for x in range(B)]
        surf_verts = torch.stack([V0[i] for i in ids])
        # Either one flow field for all surfaces or a separate flow field per
        # surface
        if discrete_flow.shape[1] == C * ndims:
            surf_flow_field = discrete_flow[:, c * ndims:(c + 1) * ndims, ...]
        else:
            surf_flow_field = discrete_flow
        # Integrate numerically
        for _ in range(n_integration):
            # Get flow at vertex positions
            flow = aggregate_trilinear(
                surf_flow_field,
                surf_verts,
                mode="bilinear",
            )

            # V <- V + h * Flow
            surf_verts = surf_verts + h * flow

        for j, i in enumerate(ids):
            VT[i] = surf_verts[j]

    return VT


def apply_translation(translation, V0, X_dims):
    """Forward pass"""

    ndims = V0[0].shape[-1]

    # Number of different organs/meshes
    B, C = X_dims

    VT = [None] * B * C

    # Iterate over surfaces
    for c in range(C):
        # Verts of this mesh in every sample of the batch
        ids = [x * C + c for x in range(B)]
        surf_verts = torch.stack([V0[i] for i in ids])
        flow = translation[:, c * ndims:(c + 1) * ndims]
        # V <- V + translation
        surf_verts = surf_verts + flow.unsqueeze(1)

        for j, i in enumerate(ids):
            VT[i] = surf_verts[j]

    return VT


class UNetFlow(V2MModel):
    """ A novel UNet-based mesh-deformation model

    :param n_m_classes: Number of mesh classes to distinguish
    :param num_input_channels: The number of image channels
    :param encoder_channels: Encoder channels for the UNet
    :param decoder_channels: Decoder channels for the UNet
    :param p_dropout_unet: UNet dropout probability
    :param ndims: Dimension of the space (usually 3)
    :param n_euler_steps: The number of numercial integration steps.
    :param patch_size: The patch size of the input images
    """

    def __init__(
        self,
        n_m_classes: int,
        n_v_classes: int,
        num_input_channels: int,
        encoder_channels: Sequence[Sequence],
        decoder_channels: Sequence[Sequence],
        deep_supervision: bool,
        p_dropout_unet: float,
        ndims: int,
        n_euler_steps: int,
        patch_size: Sequence[int],
        norm: str,
        **kwargs
    ):

        super().__init__()

        self.n_integration = n_euler_steps
        n_out_dims = ndims * n_m_classes

        # UNet
        self.unet = ResidualUNet(
            num_classes=n_v_classes,
            num_input_channels=num_input_channels,
            down_channels=encoder_channels,
            up_channels=decoder_channels,
            deep_supervision=deep_supervision,
            patch_shape=patch_size,
            voxel_decoder=True,
            p_dropout=p_dropout_unet,
            ndims=ndims,
            init_last_zero=True,
            norm=norm,
        )

        # Output layers
        n_unet_steps = len(decoder_channels)
        self.pool = nn.AvgPool3d(
            (torch.tensor(patch_size) * (0.5) ** n_unet_steps).long().tolist()
        )
        self.linear = nn.Linear(encoder_channels[-1], n_out_dims, bias=False)
        nn.init.constant_(self.linear.weight, 0.0)

        out_layers = []
        for i in range(n_unet_steps):
            n_out = ndims if i == n_unet_steps - 1 else n_out_dims
            out_layer = nn.Conv3d(
                decoder_channels[i], n_out, 1, bias=False
            )
            nn.init.constant_(out_layer.weight, 0.0)
            out_layers.append(out_layer)
        self.out_layers = nn.ModuleList(out_layers)

    def load_part(self, path:str):
        super().load_part(path)

        params = self.state_dict()

        # Init flow output layers with 0
        for k, v in params.items():
            if k == 'linear.weight' or 'out_layers' in k:
                nn.init.constant_(v, 0.0)

    @logger.measure_time
    def forward(self, x: torch.Tensor, input_meshes: MeshesXD):
        """Forward pass"""
        B, C = input_meshes.X_dims()

        pred_verts_list = input_meshes.verts_list()
        pred_faces_list = input_meshes.faces_list()
        pred_features_list = input_meshes.verts_features_list()

        # UNet features
        encoder_skips, decoder_skips, seg_out = self.unet(x)

        # Translation
        bottleneck_features = self.pool(encoder_skips[-1]).view(B, -1)
        translation_flow = self.linear(bottleneck_features)
        VT_1 = apply_translation(translation_flow, pred_verts_list, (B, C))
        pred_verts_list += VT_1
        pred_faces_list += input_meshes.faces_list()
        pred_features_list += input_meshes.verts_features_list()

        # Integrate intermediate flows
        VT = VT_1
        for s, out_layer in enumerate(self.out_layers, start=1):
            # Mesh deformation
            discrete_flow = out_layer(decoder_skips[s - 1]).float()
            if discrete_flow.isnan().any():
                breakpoint()
            VT = apply_flow(
                discrete_flow,
                VT,
                (B, C),
                self.n_integration
            )
            pred_verts_list += VT
            pred_faces_list += input_meshes.faces_list()
            pred_features_list += input_meshes.verts_features_list()

        pred_meshes = MeshesXD(
            pred_verts_list,
            pred_faces_list,
            X_dims=(len(self.out_layers) + 2, B, C),
            verts_features=pred_features_list,
        )

        if logger.debug():
            import trimesh
            import os
            for i, (v, f) in enumerate(
                zip(pred_meshes.verts_list(), pred_meshes.faces_list())
            ):
                trimesh.Trimesh(
                    v.detach().cpu().numpy(),
                    f.detach().cpu().numpy(),
                    process=False
                ).export(
                    os.path.join(logger.get_log_dir(), f"pred_mesh_{i}.ply")
                )

        return pred_meshes, seg_out

    def save(self, path):
        """Save model with its parameters to the given path.
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
    def pred_to_raw_voxel_pred(pred) -> Sequence[torch.Tensor]:
        """ Get the voxel prediction per class. May be a list if deep
        supervision is used. """
        return pred[1]

    def pred_to_pred_meshes(pred) -> MeshesXD:
        """Get the predicted meshes."""
        return pred[0]

    @staticmethod
    def pred_to_final_mesh_pred(pred) -> MeshesXD:
        """Get the final mesh prediction. In contrast to 'pred_to_pred_meshes'
        this does not involve intermediate deformation stages."""
        mesh_pred = pred[0]
        S, B, C = mesh_pred.X_dims()
        final_meshpred = MeshesXD(
            mesh_pred.verts_list()[-B * C:],
            mesh_pred.faces_list()[-B * C:],
            X_dims=(B, C),
            verts_features=mesh_pred.verts_features_list()[-B * C:],
        )
        return final_meshpred
