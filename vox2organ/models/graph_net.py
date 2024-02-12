
""" Graph (sub-)networks """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from typing import Union, Tuple, Sequence

import torch
from torch import nn
from torch.cuda.amp import autocast
from pytorch3d.structures import MeshesXD

import logger
from utils.graph_conv import Features2FeaturesResidual
from utils.feature_aggregation import (
    aggregate_from_indices
)

def _solve_euler(n_steps, V0, f):
    h = 1./n_steps
    Vt = V0
    for i in range(n_steps):
        dV = f(i * h, Vt)
        Vt = Vt + h * dV
    return Vt

def _solve_midpoint(n_steps, V0, f):
    h = 1./n_steps
    Vt = V0
    for i in range(n_steps):
        dV1 = f(i * h, Vt)
        dV2 = f(i * h + h/2, Vt + h * dV1/2)
        Vt = Vt + h * dV2
    return Vt

def solve_ode(n_steps, V0, f, solver):
    if solver == "Euler":
        return _solve_euler(n_steps, V0, f)
    if solver == "Midpoint":
        return _solve_midpoint(n_steps, V0, f)
    raise ValueError(f"Unknown solver {solver}")


# Autocast should be off for pytorch3d convs
autocast_on = True

class SurfaceDeform(nn.Module):
    """ Module implementing the differential equation dV/dt = f(t, V0).
    Our equation is autonomous
    (no explicit dependence on t) but has this parameter as an input for
    the sake of generality.

    Vt should be given in padded representation.
    """

    def __init__(
        self,
        norm: str,
        graph_channels_in: Union[list, tuple],
        graph_channels_out: Union[list, tuple],
        GC,
        patch_size: Tuple[int, int, int],
        aggregate_indices: Tuple[int],
        skip_channels: Union[list, tuple],
        p_dropout: float=None,
        ndims: int=3,
        aggregate: str='trilinear',
        n_residual_blocks: int=3,
        n_f2f_hidden_layer: int=2,
    ):

        super().__init__()

        self.GC = GC
        self.p_dropout = p_dropout
        self.patch_size = patch_size
        self.ndims = ndims
        self.aggregate = aggregate
        self.agg_indices = aggregate_indices

        # Set at each forward pass
        self.skips = None
        self.in_graph_features = None
        self.out_graph_features = None
        self.template_edges_packed = None

        # Number of image features from CNN
        skip_features_count = torch.sum(
            torch.tensor(skip_channels)[aggregate_indices, None]
        ).item()

        # Input channels consist of coordinates, vertex features,
        # neighboring features and surface IDs, and image features
        n_in_channels = (
            ndims + graph_channels_in + skip_features_count
        )

        res_blocks = []
        # First block gets concatenated features as input
        res_blocks.append(
            Features2FeaturesResidual(
                n_in_channels,
                graph_channels_out,
                hidden_layer_count=n_f2f_hidden_layer,
                norm=norm,
                GC=GC,
                p_dropout=p_dropout,
            )
        )
        # Later blocks get features of previous block as input
        for _ in range(n_residual_blocks - 1):
            res_blocks.append(
                Features2FeaturesResidual(
                    graph_channels_out,
                    graph_channels_out,
                    hidden_layer_count=n_f2f_hidden_layer,
                    norm=norm,
                    GC=GC,
                    p_dropout=p_dropout,
            )
        )

        # Cannot be nn.Sequential because graph convs take two inputs but
        # provide only one output. Maybe try torch_geometric.nn.Sequential
        self.f2f_res = nn.ModuleList(res_blocks)

        # Feature to vertex layer
        self.f2v = GC(graph_channels_out, ndims, init='zero')

        # Optionally create lns layers
        if self.aggregate == 'lns':
            raise NotImplementedError(
                "LNS not implemented, see" " Voxel2Mesh repo."
            )

    def update_image_features(self, new_feature_maps):
        """ Set the image feature maps on which the deformation is conditioned.
        """
        self.skips = [(f if autocast_on else f.float()) for f in new_feature_maps]


    def update_latent_graph_features(self, new_features):
        """ Provide per-vertex features from previous deformation block.
        """
        if new_features.dim() != 3:
            raise ValueError("Wrong dimensionality of features.")
        self.in_graph_features = new_features


    def get_out_features(self):
        """ Provide the output vertex features from the GNN.
        """
        return self.out_graph_features


    def set_template_edges_packed(self, edges):
        self.template_edges_packed = edges


    # No autocast for pytorch3d convs possible
    @autocast(enabled=autocast_on)
    @logger.measure_time
    def forward(self, t, Vt) -> torch.Tensor:
        """ Computing the derivative dV(t) from a surface given by vertices
        V(t). Vt should be in shape (batch size, V template, dim.) which is
        also the shape of the output.
        """

        batch_size, V, D = Vt.shape # No scene dim.

        verts_padded = Vt
        edges_packed = self.template_edges_packed

        # Latent features of vertices from voxels
        skipped_features = aggregate_from_indices(
            self.skips,
            verts_padded,
            self.agg_indices,
            mode=self.aggregate
        )

        # Concatenate along feature dimension
        latent_features_padded = torch.cat(
            (verts_padded, self.in_graph_features, skipped_features), dim=2
        )

        # Pass latent feature through GNN blocks
        latent_features_packed = latent_features_padded.view(
            batch_size * V, -1
        )
        for f2f in self.f2f_res:
            latent_features_packed = f2f(
                latent_features_packed, edges_packed
            )

        # Store output features
        self.out_graph_features = latent_features_packed.view(
            batch_size, V, -1
        )

        # dV
        dV = self.f2v(latent_features_packed, edges_packed)

        return dV.view(batch_size, V, D)


class GraphDecoder(nn.Module):
    """ A graph decoder that takes a template mesh and voxel features as input.
    """
    def __init__(
        self,
        graph_channels: Union[list, tuple],
        aggregate_indices: Tuple[Tuple[int]],
        ndims: int,
        n_vertex_classes: int,
        n_euler_steps: int,
        ode_solver: str,
        **kwargs
    ):
        super().__init__()

        assert (len(graph_channels) - 1 == len(aggregate_indices)),\
                "Graph channels and aggregation indices must"\
                " match the number of mesh decoder steps."

        self.num_steps = len(graph_channels) - 1
        self.n_euler_steps = n_euler_steps
        self.ndims = ndims
        self.ode_solver = ode_solver
        self.h = nn.Parameter(
            torch.tensor(1./n_euler_steps), requires_grad=False
        )


        ### Graph decoder

        # Initial GNN block that processes input coordinates and potentially
        # associated vertex features
        n_in_features = ndims + n_vertex_classes
        self.gnn_block_0 = Features2FeaturesResidual(
            n_in_features,
            graph_channels[0],
            kwargs.get('n_f2f_hidden_layer'),
            norm=kwargs.get('norm'),
            GC=kwargs.get('GC')
        )

        self.deform_layers = []
        for i in range(self.num_steps):
            self.deform_layers.append(
                SurfaceDeform(
                    graph_channels_in=graph_channels[i],
                    graph_channels_out=graph_channels[i+1],
                    aggregate_indices=aggregate_indices[i],
                    **kwargs
                )
            )

        self.deform_layers = nn.ModuleList(self.deform_layers)

    @logger.measure_time
    def forward(self, skips, input_meshes: MeshesXD):
        """ Forward pass.
        Parameters:
            - skips: Image feature maps on which the deformation should be
            condidtioned.
            - input_meshes: MeshesOfMeshes of the correct batch size which are
            deformed.
        """
        batch_size, C, V_max, D = input_meshes.verts_padded_XD().shape
        V_scene = torch.sum(input_meshes.num_verts_per_mesh()[:C]).cpu().item()

        # Template/input meshes as initial "prediction" in step 0
        pred_verts = [input_meshes.verts_packed()]
        pred_features = [torch.cat(
            [input_meshes.verts_features_packed(),
            torch.zeros_like(input_meshes.verts_packed())], # Zero displacement
            dim=-1
        )]

        # Initial creation of latent features (packed)
        with autocast(enabled=autocast_on):
            latent_features = self.gnn_block_0(
                torch.cat(
                    [input_meshes.verts_packed(),
                     input_meshes.verts_features_packed()],
                    dim=-1
                ),
                input_meshes.edges_packed()
            )

        # Iterate over decoder steps
        for step, deform in enumerate(self.deform_layers):

            # Get vertex features from previous step
            deform.update_latent_graph_features(
                latent_features.view(batch_size, V_scene, -1)
            )
            # Get image features
            deform.update_image_features(skips)

            # Set connections only once
            if deform.template_edges_packed is None:
                deform.set_template_edges_packed(
                    input_meshes.edges_packed()
                )

            # Load initial value as output from previous step; reshaping
            # assumes that all subjects have the same number of vertices which
            # is true for template-based deformation
            V0 = pred_verts[step]

            # Forward Euler integration
            VT = solve_ode(
                self.n_euler_steps,
                V0.view(batch_size, -1, D),
                deform,
                self.ode_solver
            ).view(-1, D)

            # Latent feature from deform block
            latent_features = deform.out_graph_features.view(
                batch_size * V_scene, -1
            )

            # Predicted vertices
            pred_verts.append(VT)
            # Predicted features = input features and displacement vectors
            pred_features.append(
                torch.cat(
                    [input_meshes.verts_features_packed(), VT - V0],
                    dim=-1
                )
            )

        # Create output meshes with dimensions
        # (S, B, C, V, D)
        verts_list = []
        features_list = []
        split_size = input_meshes.num_verts_per_mesh().tolist()
        for v, vf in zip(pred_verts, pred_features):
            verts_list += list(v.split(split_size, dim=0))
            features_list += list(vf.split(split_size, dim=0))

        pred_meshes = MeshesXD(
            verts=verts_list,
            faces=input_meshes.faces_list() * (self.num_steps + 1), # Const. connectivity
            X_dims=(self.num_steps + 1, batch_size, C),
            verts_features=features_list
        )

        if logger.debug():
            import trimesh
            import os
            for i, (v, f) in enumerate(
                zip(pred_meshes.verts_list(), pred_meshes.faces_list())
            ):
                trimesh.Trimesh(
                    v.detach().cpu().numpy(), f.detach().cpu().numpy(), process=False
                ).export(
                    os.path.join(logger.get_log_dir(), f"pred_mesh_{i}.ply")
                )

        return pred_meshes
