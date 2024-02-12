
""" Evaluation metrics. Those metrics are typically computed directly from the
model prediction, i.e., in normalized coordinate space unless specified
otherwise."""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from abc import ABC, abstractmethod

import numpy as np
import torch
import pandas as pd
import pymeshlab as pyml
from scipy.spatial.distance import dice
from pytorch3d.ops import (
    sample_points_from_meshes
)
from pytorch3d.structures import Meshes, Pointclouds, MeshesXD

import logger
from utils.utils import (
    voxelize_mesh,
)
from utils.cortical_thickness import _point_mesh_face_distance_unidirectional


class EvalMetric(ABC):
    """ Abstract base class handling evaluation of meshes/voxels. All
    evaluation methods should overwrite its __call__ method. """

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(
        self,
        mesh_pred: MeshesXD,
        mesh_gt: MeshesXD,
        n_m_classes,
        mesh_label_names,
        voxel_pred,
        voxel_gt,
        n_v_classes,
        voxel_label_names,
    ):
        pass

    def result_blueprint(self):
        return {"Tissue": [], "Metric": [], "Value": []}


class SurfaceDistance(EvalMetric):
    """ Compute mesh-to-mesh distances by sampling random points on either
    surfaces. """

    def __init__(self):
        self.n_points = 100000

    def __call__(
        self,
        mesh_pred,
        mesh_gt,
        n_m_classes,
        mesh_label_names,
        voxel_pred,
        voxel_gt,
        n_v_classes,
        voxel_label_names,
    ):

        gt_vertices = mesh_gt.verts_list()
        gt_faces = mesh_gt.faces_list()

        pred_vertices = mesh_pred.verts_list()
        pred_faces = mesh_pred.faces_list()

        # Iterate over structures
        results = self.result_blueprint()
        for name, pred_v, pred_f, gt_v, gt_f in zip(
            mesh_label_names,
            pred_vertices,
            pred_faces,
            gt_vertices,
            gt_faces
        ):

            pred_mesh = Meshes([pred_v], [pred_f])
            pred_pcl = sample_points_from_meshes(pred_mesh, self.n_points)
            pred_pcl = Pointclouds(pred_pcl)

            gt_mesh = Meshes([gt_v], [gt_f])
            gt_pcl = sample_points_from_meshes(gt_mesh, self.n_points)
            gt_pcl = Pointclouds(gt_pcl)

            # Compute distance
            P2G_dist = _point_mesh_face_distance_unidirectional(
                gt_pcl, pred_mesh
            ).cpu().numpy()
            G2P_dist = _point_mesh_face_distance_unidirectional(
                pred_pcl, gt_mesh
            ).cpu().numpy()

            assd = ((P2G_dist.sum() + G2P_dist.sum()) /
                    float(P2G_dist.shape[0] + G2P_dist.shape[0]))
            results["Tissue"].append(name)
            results["Metric"].append("ASSD")
            results["Value"].append(assd)
            hd90 = max(np.percentile(P2G_dist, 90),
                       np.percentile(G2P_dist, 90))
            results["Tissue"].append(name)
            results["Metric"].append("HD90")
            results["Value"].append(hd90)
            hd95 = max(np.percentile(P2G_dist, 95),
                       np.percentile(G2P_dist, 95))
            results["Tissue"].append(name)
            results["Metric"].append("HD95")
            results["Value"].append(hd95)
            hd99 = max(np.percentile(P2G_dist, 99),
                       np.percentile(G2P_dist, 99))
            results["Tissue"].append(name)
            results["Metric"].append("HD99")
            results["Value"].append(hd99)

        return results


class SelfIntersections(EvalMetric):
    """ Comute relative number of self-intersections """

    def __init__(self):
        pass

    def __call__(
        self,
        mesh_pred,
        mesh_gt,
        n_m_classes,
        mesh_label_names,
        voxel_pred,
        voxel_gt,
        n_v_classes,
        voxel_label_names,
    ):
        """ Based on
        https://bitbucket.csiro.au/projects/CRCPMAX/repos/corticalflow/browse/src/metrics.py
        """

        pred_vertices = mesh_pred.verts_list()
        pred_faces = mesh_pred.faces_list()

        results = self.result_blueprint()

        for name, pred_v, pred_f in zip(
            mesh_label_names,
            pred_vertices,
            pred_faces,
        ):

            ms = pyml.MeshSet()
            ms.add_mesh(pyml.Mesh(pred_v.cpu().numpy(), pred_f.cpu().numpy()))
            faces = ms.compute_topological_measures()['faces_number']
            ms.select_self_intersecting_faces()
            ms.delete_selected_faces()
            nnSI_faces = ms.compute_topological_measures()['faces_number']
            SI_faces = faces-nnSI_faces
            fracSI = (SI_faces/faces)*100

            results["Tissue"].append(name)
            results["Metric"].append("Intersections")
            results["Value"].append(fracSI)

        return results


class VoxelDice(EvalMetric):
    """ Compute the Dice score between voxel prediction and voxel ground truth
    """

    def __init__(self):
        pass

    def __call__(
        self,
        mesh_pred,
        mesh_gt,
        n_m_classes,
        mesh_label_names,
        voxel_pred,
        voxel_gt,
        n_v_classes,
        voxel_label_names,
    ):
        label_ids = range(1, n_v_classes)

        results = self.result_blueprint()

        for i, name in zip(label_ids, voxel_label_names):
            sim = 1 - dice(
                (voxel_pred == i).cpu().numpy().flatten(),
                (voxel_gt == i).cpu().numpy().flatten()
            )
            results["Tissue"].append(name)
            results["Metric"].append("VoxelDice")
            results["Value"].append(sim)

        return results


class MeshDice(EvalMetric):
    """ Compute Dice score between mesh prediction and voxel ground truth """

    def __init__(self):
        pass

    def __call__(
        self,
        mesh_pred,
        mesh_gt,
        n_m_classes,
        mesh_label_names,
        voxel_pred,
        voxel_gt,
        n_v_classes,
        voxel_label_names,
    ):

        pred_vertices = mesh_pred.verts_list()
        pred_faces = mesh_pred.faces_list()

        label_ids = range(1, n_v_classes)

        results = self.result_blueprint()

        for i, name in zip(label_ids, voxel_label_names):

            # Find corresponding mesh(es)
            mesh_names = [j for j in mesh_label_names if name in j]
            mesh_ids = [mesh_label_names.index(j) for j in mesh_names]

            pred_mesh = Meshes(
                [pred_vertices[j] for j in mesh_ids],
                [pred_faces[j] for j in mesh_ids]
            )
            pred_voxelized = voxelize_mesh(
                pred_mesh.verts_packed().cpu().numpy(),
                pred_mesh.faces_packed().cpu().numpy(),
                voxel_gt.shape
            )

            sim = 1 - dice(
                (pred_voxelized == 1).cpu().numpy().flatten(),
                (voxel_gt == i).cpu().numpy().flatten()
            )

            results["Tissue"].append(name)
            results["Metric"].append("MeshDice")
            results["Value"].append(sim)

        return results


def Jaccard(pred, target, n_classes):
    """ Jaccard/Intersection over Union """
    ious = []
    # Ignoring background class 0
    for c in range(1, n_classes):
        pred_idxs = pred == c
        target_idxs = target == c
        intersection = pred_idxs[target_idxs].long().sum().data.cpu()
        union = (
            pred_idxs.long().sum().data.cpu() +
            target_idxs.long().sum().data.cpu() -
            intersection
        )
        # +1 for smoothing (no division by 0)
        ious.append(float(intersection + 1) / float(union + 1))

    # Return average iou over classes ignoring background
    return np.sum(ious)/(n_classes - 1)
