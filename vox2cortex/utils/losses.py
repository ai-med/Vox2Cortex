
""" Loss function collection for convenient calculation over multiple classes
and/or instances of meshes.

    Notation:
        - B: batch size
        - S: number of instances, e.g. steps/positions where the loss is computed in
        the model
        - C: number of channels (= number of classes usually)
        - V: number of vertices
        - F: number of faces
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from abc import ABC, abstractmethod
from typing import Union
from collections.abc import Sequence

import torch
import pytorch3d.structures
from pytorch3d.structures import Meshes
from pytorch3d.loss import (chamfer_distance,
                            mesh_edge_loss,
                            mesh_laplacian_smoothing,
                            mesh_normal_consistency)
from pytorch3d.ops import sample_points_from_meshes, laplacian
from torch.cuda.amp import autocast

from utils.utils import choose_n_random_points
from utils.mesh import curv_from_cotcurv_laplacian

def point_weigths_from_curvature(curvatures: torch.Tensor,
                                 points: torch.Tensor,
                                 max_weight: Union[float, int, torch.Tensor],
                                 padded_coordinates=(-1.0, -1.0, -1.0)):
    """ Calculate Chamfer weights from curvatures such that they are in
    [1, max_weight]. In addition, the weight of padded points is set to zero."""

    if not isinstance(max_weight, torch.Tensor):
        max_weight = torch.tensor(max_weight).float()

    # Weights in [1, max_weight]
    weights = torch.minimum(1 + curvatures, max_weight.cuda())

    # Set weights of padded vertices to 0
    padded_coordinates = torch.Tensor(padded_coordinates).to(points.device)
    weights[torch.isclose(points, padded_coordinates).all(dim=2)] = 0.0

    return weights

def meshes_to_edge_normals_2D_packed(meshes: Meshes):
    """ Helper function to get normals of 2D meshes (contours) for every edge.
    The normals have the same length as the respective edge they belong
    to."""
    verts_packed = meshes.verts_packed()
    edges_packed = meshes.faces_packed() # edges=faces

    v0_idx, v1_idx = edges_packed[:, 0], edges_packed[:, 1]

    # Normal of edge [x,y] is [-y,x]
    normals = (verts_packed[v1_idx] - verts_packed[v0_idx]).flip(
        dims=[1]) * torch.tensor([-1.0, 1.0]).to(meshes.device)

    # Length of normals is automatically equal to the edge length as the normals
    # are just rotated edges.

    return normals, v0_idx, v1_idx

def meshes_to_vertex_normals_2D_packed(meshes: Meshes):
    """ Helper function to get normals of 2D meshes (contours) for every vertex.
    The normals are defined as the sum of the normals of the two adjacent edges
    of the vertex. """
    # Normals of edges
    edge_normals, v0_idx, v1_idx = meshes_to_edge_normals_2D_packed(
        meshes
    )
    # (Normals at vertices) = (sum of normals at adjacent edges of the
    # respective edge length)
    normals_next = edge_normals[torch.argsort(v0_idx)]
    normals_prev = edge_normals[torch.argsort(v1_idx)]

    return normals_next + normals_prev

class MeshLoss(ABC):
    """ Abstract base class for all mesh losses. """

    def __init__(self, ignore_coordinates=(-1.0, -1.0, -1.0)):
        self.ignore_coordinates = torch.tensor(ignore_coordinates)

    def __str__(self):
        return self.__class__.__name__ + "()"

    def __call__(self, pred_meshes, target, weights=None):
        """ Mesh loss calculation

        :param pred_meshes: A multidimensional array of predicted meshes of shape
        (S, C), each of type pytorch3d.structures.Meshes
        :param target: A multidimensional array of target points of shape (C)
        i.e. one tensor per class
        :param weights: Losses are weighed per class.
        :return: The calculated loss.
        """
        if isinstance(self, ChamferAndNormalsLoss):
            mesh_loss = torch.tensor([0,0]).float().cuda()
        else:
            mesh_loss = torch.tensor(0).float().cuda()

        S = len(pred_meshes)
        C = len(pred_meshes[0])

        if weights is not None:
            if len(weights) != C:
                raise ValueError("Weights should be specified per class.")
        else: # no per-class-weights provided
            weights = torch.tensor([1.0] * C).float().cuda()

        for s in range(S):
            for c, w in zip(range(C), weights):
                mesh_loss += self.get_loss(pred_meshes[s][c], target[c]) * w

        return mesh_loss

    @abstractmethod
    def get_loss(self,
                 pred_meshes: pytorch3d.structures.Meshes,
                 target: Union[pytorch3d.structures.Pointclouds,
                               pytorch3d.structures.Meshes,
                               tuple,
                               list]):
        raise NotImplementedError()


class ChamferLoss(MeshLoss):
    """ Chamfer distance between the predicted mesh and randomly sampled
    surface points or a reference mesh. """

    def __init__(self, curv_weight_max=None):
        super().__init__()
        self.curv_weight_max = curv_weight_max

    def __str__(self):
        return f"ChamferLoss(curv_weight_max={self.curv_weight_max})"

    def get_loss(self, pred_meshes, target):
        if isinstance(target, pytorch3d.structures.Pointclouds):
            n_points = torch.min(target.num_points_per_cloud())
            target_ = target
            if self.curv_weight_max is not None:
                raise RuntimeError("Can only apply curvature weights if they"
                                   " are provided in the target.")

        if isinstance(target, pytorch3d.structures.Meshes):
            n_points = torch.min(target.num_verts_per_mesh())
            if target.verts_padded().shape[1] == 3:
                target_ = sample_points_from_meshes(target, n_points)
            else: # 2D --> choose vertex points
                target_ = choose_n_random_points(
                    target.verts_padded(), n_points
                )
            if self.curv_weight_max is not None:
                raise RuntimeError("Cannot apply curvature weights for"
                                   " targets of type 'Meshes'.")

        if isinstance(target, Sequence):
            # target = (verts, normals, curvatures)
            target_ = target[0] # Only vertices relevant
            assert target_.ndim == 3 # padded
            n_points = target_.shape[1]
            target_curvs = target[2]
            point_weights = point_weigths_from_curvature(
                target_curvs, target_points, self.curv_weight_max
            ) if self.curv_weight_max else None

        if pred_meshes.verts_packed().shape[-1] == 3:
            pred_points = sample_points_from_meshes(pred_meshes, n_points)
        else: # 2D
            pred_points = choose_n_random_points(
                pred_meshes.verts_padded(), n_points
            )

        pred_lengths = (~torch.isclose(
            pred_points, self.ignore_coordinates.to(pred_points.device)
        ).all(dim=2)).sum(dim=1)
        target_lengths = (~torch.isclose(
            target_, self.ignore_coordinates.to(pred_points.device)
        ).all(dim=2)).sum(dim=1)
        return chamfer_distance(
            pred_points,
            target_,
            point_weights=point_weights,
            x_lengths=pred_lengths,
            y_lengths=target_lengths
        )[0]


class ChamferAndNormalsLoss(MeshLoss):
    """ Chamfer distance & cosine distance between the vertices and normals of
    the predicted mesh and a reference mesh.
    """

    def __init__(self, curv_weight_max=None):
        super().__init__()
        self.curv_weight_max = curv_weight_max

    def __str__(self):
        return f"ChamferAndNormalsLoss(curv_weight_max={self.curv_weight_max})"

    def get_loss(self, pred_meshes, target):
        if len(target) < 2:
            raise TypeError("ChamferAndNormalsLoss requires vertices and"\
                            " normals.")
        target_points, target_normals = target[0], target[1]
        assert target_points.ndim == 3 and target_normals.ndim == 3
        n_points = target_points.shape[1]
        if target_points.shape[-1] == 3: # 3D
            pred_points, pred_normals = sample_points_from_meshes(
                pred_meshes, n_points, return_normals=True
            )
            target_curvs = target[2]
            point_weights = point_weigths_from_curvature(
                target_curvs, target_points, self.curv_weight_max
            ) if self.curv_weight_max else None
        else: # 2D
            pred_points, idx = choose_n_random_points(
                pred_meshes.verts_padded(), n_points, return_idx=True
            )
            pred_normals = meshes_to_vertex_normals_2D_packed(pred_meshes)
            # Select the normals of the corresponding points
            pt_shape = pred_points.shape
            pred_normals = pred_normals.view(
                pt_shape)[idx.unbind(1)].view(pt_shape)
            # Normals are required to be 3 dim. for chamfer function
            N, V, _ = target_normals.shape
            target_normals = torch.cat(
                [target_normals,
                 torch.zeros((N,V,1)).to(target_normals.device)], dim=2
            )
            pred_normals = torch.cat(
                [pred_normals,
                 torch.zeros((N,V,1)).to(pred_normals.device)], dim=2
            )
            if self.curv_weight_max is not None:
                raise RuntimeError("Cannot apply curvature weights in 2D.")

        pred_lengths = (~torch.isclose(
            pred_points, self.ignore_coordinates.to(pred_points.device)
        ).all(dim=2)).sum(dim=1)
        target_lengths = (~torch.isclose(
            target_points, self.ignore_coordinates.to(pred_points.device)
        ).all(dim=2)).sum(dim=1)
        losses = chamfer_distance(
            pred_points,
            target_points,
            x_normals=pred_normals,
            x_lengths=pred_lengths,
            y_normals=target_normals,
            y_lengths=target_lengths,
            point_weights=point_weights,
            oriented_cosine_similarity=True
        )
        d_chamfer = losses[0]
        d_cosine = losses[1]

        return torch.stack([d_chamfer, d_cosine])


class LaplacianLoss(MeshLoss):
    def __init(self):
        super().__init__()
    # Method does not support autocast
    @autocast(enabled=False)
    def get_loss(self, pred_meshes, target=None):
        # pytorch3d loss for 3D
        if pred_meshes.verts_padded().shape[-1] == 3:
            loss = mesh_laplacian_smoothing(
                Meshes(pred_meshes.verts_padded().float(),
                       pred_meshes.faces_padded().float()),
                method='uniform'
            )
        # 2D
        else:
            verts_packed = pred_meshes.verts_packed()
            edges_packed = pred_meshes.faces_packed() # faces = edges
            V = len(verts_packed)
            # Uniform Laplacian
            with torch.no_grad():
                L = laplacian(verts_packed, edges_packed)
            loss = L.mm(verts_packed).norm(dim=1).sum() / V

        return loss


class NormalConsistencyLoss(MeshLoss):
    def __init(self):
        super().__init__()
    def get_loss(self, pred_meshes, target=None):
        # 2D: assumes clock-wise ordering of vertex indices in each edge
        if pred_meshes.verts_padded().shape[-1] == 2:
            normals, v0_idx, v1_idx = meshes_to_edge_normals_2D_packed(pred_meshes)
            loss = 1 - torch.cosine_similarity(normals[v0_idx],
                                               normals[v1_idx])
            return loss.sum() / len(v0_idx)

        # 3D: pytorch3d
        return mesh_normal_consistency(pred_meshes)


class EdgeLoss(MeshLoss):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def __str__(self):
        return f"EdgeLoss({self.target_length})"

    def get_loss(self, pred_meshes, target=None):
        # 2D
        if pred_meshes.verts_padded().shape[2] == 2:
            verts_packed = pred_meshes.verts_packed()
            edges_packed = pred_meshes.faces_packed() # edges=faces
            verts_edges = verts_packed[edges_packed]
            v0, v1 = verts_edges.unbind(1)
            loss = ((v0 - v1).norm(dim=1, p=2) - self.target_length) ** 2.0
            return loss.sum() / len(edges_packed)
        # 3D
        return mesh_edge_loss(pred_meshes, target_length=self.target_length)


def linear_loss_combine(losses, weights):
    """ Compute the losses in a linear manner, e.g.
    a1 * loss1 + a2 * loss2 + ...

    :param losses: The individual losses.
    :param weights: The weights for the losses.
    :returns: The overall (weighted) loss.
    """
    loss_total = 0
    for loss, weight in zip(losses, weights):
        loss_total += weight * loss

    return loss_total


def _add_MultiLoss_to_dict(loss_dict, loss_func, mesh_pred,
                            mesh_target, weights, names):
    """ Add a multi-loss (i.e. a loss that returns multiple values like
    ChamferAndNormalsLoss) to the loss_dict.
    """
    # All weights a Sequence or none of them
    assert (all(map(lambda x: isinstance(x, Sequence), weights))
            or
            not any(map(lambda x: isinstance(x, Sequence), weights)))
    # Same length of weights and names
    assert len(weights) == len(names)
    # Either per-class weight or single weight for all classes
    if isinstance(weights[0], Sequence):
        # Reorder weights by exchanging class and loss function dimension
        weights = torch.tensor(weights).cuda().T
        # Weights processed by loss function
        ml = loss_func(mesh_pred, mesh_target, weights)
        for i, n in enumerate(names):
            loss_dict[n] = ml[i]
    else:
        ml = loss_func(mesh_pred, mesh_target)
        # Weights multiplied here
        for i, (n, w) in enumerate(zip(names, weights)):
            loss_dict[n] = ml[i] * w

def all_linear_loss_combine(voxel_loss_func, voxel_loss_func_weights,
                            voxel_pred, voxel_target,
                            mesh_loss_func, mesh_loss_func_weights,
                            mesh_pred, deltaV_mesh_pred, mesh_target):
    """ Linear combination of all losses. """
    losses = {}
    # Voxel losses
    if voxel_pred is not None:
        if not isinstance(voxel_pred, Sequence):
            # If deep supervision is used, voxel prediction is a list. Therefore,
            # non-list predictions are made compatible
            voxel_pred = [voxel_pred]
        for lf, w in zip(voxel_loss_func, voxel_loss_func_weights):
            losses[str(lf)] = 0.0
            for vp in voxel_pred:
                losses[str(lf)] += lf(vp, voxel_target) * w

    # Mesh losses
    mesh_loss_weights_iter = iter(mesh_loss_func_weights)
    for lf in mesh_loss_func:
        weight = next(mesh_loss_weights_iter)
        if isinstance(lf, ChamferAndNormalsLoss):
            w1 = weight
            w2 = next(mesh_loss_weights_iter)
            _add_MultiLoss_to_dict(
                losses, lf, mesh_pred, mesh_target, (w1, w2),
                ("ChamferLoss()", "CosineLoss()")
            )
        else: # add single loss to dict
            if isinstance(weight, Sequence):
                if isinstance(lf, LaplacianLoss):
                    # Use relative coordinates
                    ml = lf(deltaV_mesh_pred, mesh_target, weight)
                else:
                    ml = lf(mesh_pred, mesh_target, weight)
                losses[str(lf)] = ml
            else:
                if isinstance(lf, LaplacianLoss):
                    # Use relative coordinates
                    ml = lf(deltaV_mesh_pred, mesh_target)
                else:
                    ml = lf(mesh_pred, mesh_target)
                losses[str(lf)] = ml * weight

    loss_total = sum(losses.values())

    return losses, loss_total
