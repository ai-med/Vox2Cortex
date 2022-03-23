
""" Evaluation metrics. Those metrics are typically computed directly from the
model prediction, i.e., in normalized coordinate space unless specified
otherwise."""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from enum import IntEnum

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import (
    knn_points,
    knn_gather,
    sample_points_from_meshes
)
from pytorch3d.structures import Meshes, Pointclouds
from scipy.spatial.distance import directed_hausdorff

from utils.utils import (
    voxelize_mesh,
    voxelize_contour,
)
from utils.logging import (
    write_img_if_debug,
    measure_time)
from utils.cortical_thickness import cortical_thickness
from utils.coordinate_transform import transform_mesh_affine
from utils.mesh import Mesh
from utils.cortical_thickness import _point_mesh_face_distance_unidirectional

class EvalMetrics(IntEnum):
    """ Supported evaluation metrics """
    # Jaccard score/ Intersection over Union from voxel prediction
    JaccardVoxel = 1

    # Chamfer distance between ground truth mesh and predicted mesh
    Chamfer = 2

    # Jaccard score/ Intersection over Union from mesh prediction
    JaccardMesh = 3

    # Symmetric Hausdorff distance between two meshes
    SymmetricHausdorff = 4

    # Wasserstein distance between point clouds
    # Wasserstein = 5

    # Difference in cortical thickness compared to ground truth
    CorticalThicknessError = 6

    # Average distance of predicted and groun truth mesh in terms of
    # point-to-mesh distance
    AverageDistance = 7

def AverageDistanceScore(pred, data, n_v_classes, n_m_classes, model_class,
                         padded_coordinates=(-1.0, -1.0, -1.0)):
    """ Compute point-to-mesh distance between prediction and ground truth. """

    padded_coordinates = torch.Tensor(padded_coordinates).cuda()

    # Ground truth
    gt_mesh = data[2]
    # Back to original coordinate space
    gt_vertices, gt_faces= gt_mesh.vertices, gt_mesh.faces
    ndims = gt_vertices.shape[-1]

    # Prediction: Only consider mesh of last step
    pred_vertices, pred_faces = model_class.pred_to_verts_and_faces(pred)
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims)
    pred_faces = pred_faces[-1].view(n_m_classes, -1, ndims)

    # Iterate over structures
    assd_all = []
    for pred_v, pred_f, gt_v, gt_f in zip(
        pred_vertices,
        pred_faces,
        gt_vertices.cuda(),
        gt_faces.cuda()
    ):

        # Prediction
        pred_mesh = Meshes([pred_v], [pred_f])
        pred_pcl = sample_points_from_meshes(pred_mesh, 100000)
        pred_pcl = Pointclouds(pred_pcl)

        # Remove padded vertices from gt
        gt_v = gt_v[~torch.isclose(gt_v, padded_coordinates).all(dim=1)]
        gt_mesh = Meshes([gt_v], [gt_f])
        gt_pcl = sample_points_from_meshes(gt_mesh, 100000)
        gt_pcl = Pointclouds(gt_pcl)

        # Compute distance
        P2G_dist = _point_mesh_face_distance_unidirectional(
            gt_pcl, pred_mesh
        ).cpu().numpy()
        G2P_dist = _point_mesh_face_distance_unidirectional(
            pred_pcl, gt_mesh
        ).cpu().numpy()

        assd2 = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])

        assd_all.append(assd2)

    return assd_all

def CorticalThicknessScore(pred, data, n_v_classes, n_m_classes, model_class):
    """ Compare cortical thickness to ground truth in terms of average absolute
    difference per vertex. In order for this measure to be meaningful, predited
    and ground truth meshes are transformed into the original coordinate space."""

    if n_m_classes not in (2, 4):
        raise ValueError("Cortical thickness score requires 2 or 4 surface meshes.")

    gt_mesh = data[2]
    trans_affine = data[3]
    # Back to original coordinate space
    new_vertices, new_faces = transform_mesh_affine(
        gt_mesh.vertices, gt_mesh.faces, np.linalg.inv(trans_affine)
    )
    gt_mesh_transformed = Mesh(new_vertices, new_faces, features=gt_mesh.features)
    gt_thickness = gt_mesh_transformed.features.view(n_m_classes, -1).cuda()
    ndims = gt_mesh_transformed.ndims
    gt_vertices = gt_mesh_transformed.vertices.view(n_m_classes, -1, ndims).cuda()

    # Prediction: Only consider mesh of last step
    pred_vertices, pred_faces = model_class.pred_to_verts_and_faces(pred)
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims)
    pred_faces = pred_faces[-1].view(n_m_classes, -1, ndims)
    pred_vertices, pred_faces = transform_mesh_affine(
        pred_vertices, pred_faces, np.linalg.inv(trans_affine)
    )

    # Thickness prediction
    pred_meshes = cortical_thickness(pred_vertices, pred_faces)

    # Compare white surface thickness prediction to thickness of nearest
    # gt point
    th_all = []
    for p_mesh, gt_v, gt_th in zip(pred_meshes, gt_vertices, gt_thickness):
        pred_v = p_mesh.vertices.view(1, -1, ndims)
        pred_th = p_mesh.features.view(-1)
        _, knn_idx, _ = knn_points(pred_v, gt_v.view(1, -1, ndims))
        nearest_thickness = knn_gather(gt_th.view(1, -1, 1), knn_idx).squeeze()

        thickness_score = torch.abs(pred_th - nearest_thickness).mean()

        th_all.append(thickness_score.cpu().item())

    return th_all

@measure_time
def SymmetricHausdorffScore(pred, data, n_v_classes, n_m_classes, model_class,
                           padded_coordinates=(-1.0, -1.0, -1.0)):
    """ Symmetric Hausdorff distance between predicted point clouds.
    """
    # Ground truth
    mesh_gt = data[2]
    ndims = mesh_gt.ndims
    gt_vertices = mesh_gt.vertices.view(n_m_classes, -1, ndims)

    # Prediction: Only consider mesh of last step
    pred_vertices, _ = model_class.pred_to_verts_and_faces(pred)
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims).cpu().numpy()

    hds = []
    for p, gt_ in zip(pred_vertices, gt_vertices):
        # Remove padded vertices from gt
        gt = gt_[~np.isclose(gt_, padded_coordinates).all(axis=1)]
        d = max(directed_hausdorff(p, gt)[0],
                directed_hausdorff(gt, p)[0])
        hds.append(d)

    return hds

@measure_time
def JaccardMeshScore(pred, data, n_v_classes, n_m_classes, model_class,
                     strip=True, compare_with='mesh_gt'):
    """ Jaccard averaged over classes ignoring background. The mesh prediction
    is compared against the voxel ground truth or against the mesh ground truth.
    """
    assert compare_with in ("voxel_gt", "mesh_gt")
    input_img = data[0].cuda()
    voxel_gt = data[1].cuda()
    mesh_gt = data[2]
    ndims = mesh_gt.ndims
    shape = voxel_gt.shape
    if compare_with == 'mesh_gt':
        vertices, faces = mesh_gt.vertices, mesh_gt.faces
        if ndims == 3:
            voxel_target = voxelize_mesh(
                vertices, faces, shape, n_m_classes
            ).cuda()
        else: # 2D
            voxel_target = voxelize_contour(
                vertices, shape
            ).cuda()
    else: # voxel gt
        voxel_target = voxel_gt
    vertices, faces = model_class.pred_to_verts_and_faces(pred)
    # Only mesh of last step considered and batch dimension squeezed out
    vertices = vertices[-1].view(n_m_classes, -1, ndims)
    faces = faces[-1].view(n_m_classes, -1, ndims)
    if ndims == 3:
        voxel_pred = voxelize_mesh(
            vertices, faces, shape, n_m_classes
        ).cuda()
    else: # 2D
        voxel_pred = voxelize_contour(
            vertices, shape
        ).cuda()

    if voxel_target.ndim == 3:
        voxel_target = voxel_target.unsqueeze(0)
        # Combine all structures into one voxelization
        voxel_pred = voxel_pred.sum(0).bool().long().unsqueeze(0)

    # Debug
    write_img_if_debug(input_img.squeeze().cpu().numpy(),
                       "../misc/voxel_input_img_eval.nii.gz")
    for i, (vp, vt) in enumerate(zip(voxel_pred, voxel_target)):
        write_img_if_debug(vp.squeeze().cpu().numpy(),
                           f"../misc/mesh_pred_img_eval_{i}.nii.gz")
        write_img_if_debug(vt.squeeze().cpu().numpy(),
                           f"../misc/voxel_target_img_eval_{i}.nii.gz")

    # Jaccard per structure
    j_vox_all = []
    for vp, vt in zip(voxel_pred.cuda(), voxel_target.cuda()):
        j_vox_all.append(
            Jaccard(vp.cuda(), vt.cuda(), 2)
        )

    return j_vox_all

@measure_time
def JaccardVoxelScore(pred, data, n_v_classes, n_m_classes, model_class, *args):
    """ Jaccard averaged over classes ignoring background """
    voxel_pred = model_class.pred_to_voxel_pred(pred)
    voxel_label = data[1].cuda()

    return Jaccard(voxel_pred, voxel_label, n_v_classes)

@measure_time
def Jaccard_from_Coords(pred, target, n_v_classes):
    """ Jaccard/ Intersection over Union from lists of occupied voxels. This
    necessarily implies that all occupied voxels belong to one class.

    Attention: This function is usally a lot slower than 'Jaccard' (probably
    because it does not exploit cuda).

    :param pred: Shape (C, V, 3)
    :param target: Shape (C, V, 3)
    :param n_v_classes: C
    """
    ious = []
    # Ignoring background class 0
    for c in range(1, n_v_classes):
        if isinstance(pred[c], torch.Tensor):
            pred[c] = pred[c].cpu().numpy()
        if isinstance(target[c], torch.Tensor):
            target[c] = target[c].cpu().numpy()
        intersection = 0
        for co in pred[c]:
            if any(np.equal(target[c], co).all(1)):
                intersection += 1

        union = pred[c].shape[0] + target[c].shape[0] - intersection

        # +1 for smoothing (no division by 0)
        ious.append(float(intersection + 1) / float(union + 1))

    return np.sum(ious)/(n_v_classes - 1)

@measure_time
def Jaccard(pred, target, n_classes):
    """ Jaccard/Intersection over Union """
    ious = []
    # Ignoring background class 0
    for c in range(1, n_classes):
        pred_idxs = pred == c
        target_idxs = target == c
        intersection = pred_idxs[target_idxs].long().sum().data.cpu()
        union = pred_idxs.long().sum().data.cpu() + \
                    target_idxs.long().sum().data.cpu() -\
                    intersection
        # +1 for smoothing (no division by 0)
        ious.append(float(intersection + 1) / float(union + 1))

    # Return average iou over classes ignoring background
    return np.sum(ious)/(n_classes - 1)

def ChamferScore(pred, data, n_v_classes, n_m_classes, model_class,
                 padded_coordinates=(-1.0, -1.0, -1.0), **kwargs):
    """ Chamfer distance averaged over classes

    Note: In contrast to the ChamferLoss, where the Chamfer distance may be computed
    between the predicted loss and randomly sampled surface points, here the
    Chamfer distance is computed between the predicted mesh and the ground
    truth mesh. """
    pred_vertices, _ = model_class.pred_to_verts_and_faces(pred)
    gt_vertices = data[2].vertices.cuda()
    padded_coordinates = torch.Tensor(padded_coordinates).cuda()
    if gt_vertices.ndim == 2:
        gt_vertices = gt_vertices.unsqueeze(0)
    chamfer_scores = []
    for c in range(n_m_classes):
        pv = pred_vertices[-1][c] # only consider last mesh step
        gt = gt_vertices[c][
            ~torch.isclose(gt_vertices[c], padded_coordinates).all(dim=1)
        ]
        chamfer_scores.append(
            chamfer_distance(pv, gt[None])[0].cpu().item()
        )

    # Average over classes
    return np.sum(chamfer_scores) / float(n_m_classes)

EvalMetricHandler = {
    EvalMetrics.JaccardVoxel.name: JaccardVoxelScore,
    EvalMetrics.JaccardMesh.name: JaccardMeshScore,
    EvalMetrics.Chamfer.name: ChamferScore,
    EvalMetrics.SymmetricHausdorff.name: SymmetricHausdorffScore,
    EvalMetrics.CorticalThicknessError.name: CorticalThicknessScore,
    EvalMetrics.AverageDistance.name: AverageDistanceScore
}
