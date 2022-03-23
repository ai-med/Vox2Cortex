#!/usr/bin/env python3

""" Evaluation script that can be applied directly to predicted meshes (no need
to load model etc.) """

import os
from argparse import ArgumentParser

import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from trimesh.proximity import longest_ray
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import (
    sample_points_from_meshes,
    knn_points
)

from utils.file_handle import read_dataset_ids
from utils.cortical_thickness import _point_mesh_face_distance_unidirectional
from utils.mesh import Mesh
from utils.utils import choose_n_random_points
from data.supported_datasets import valid_ids

RAW_DATA_DIR = "/mnt/nas/Data_Neuro/"
EXPERIMENT_DIR = "../experiments/"
SURF_NAMES = ("lh_white", "rh_white", "lh_pial", "rh_pial")
PARTNER = {"rh_white": "rh_pial",
           "rh_pial": "rh_white",
           "lh_white": "lh_pial",
           "lh_pial": "lh_white"}

MODES = ('ad_hd', 'thickness', 'trt')

def eval_trt(mri_id, surf_name, eval_params, epoch, device="cuda:1",
             subfolder="meshes"):
    pred_folder = os.path.join(eval_params['log_path'])
    if "TRT" not in pred_folder:
        raise ValueError("Test-Retest evaluation is meant for TRT dataset.")

    # Skip every second scan (was already compared to its predecessor)
    subject_id, scan_id = mri_id.split("/")
    scan_id_int = int(scan_id.split("_")[1])
    if scan_id_int % 2 == 0:
        return None
    scan_id_next = f"T1_{str(scan_id_int + 1).zfill(2)}"
    mri_id_next = "/".join([subject_id, scan_id_next])

    # Load predicted meshes
    s_index = SURF_NAMES.index(surf_name)
    pred_mesh_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh = trimesh.load(pred_mesh_path)
    pred_mesh.remove_duplicate_faces(); pred_mesh.remove_unreferenced_vertices();
    #
    pred_mesh_partner_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id_next}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh_partner = trimesh.load(pred_mesh_partner_path)
    pred_mesh_partner.remove_duplicate_faces(); pred_mesh_partner.remove_unreferenced_vertices();

    # Register to each other with ICP as done by DeepCSR
    trans_affine, cost = pred_mesh.register(pred_mesh_partner)
    print("Average distance before registration for mesh"
          f" {mri_id}, {surf_name}: {cost}")
    v_pred_mesh_new = trimesh.transform_points(pred_mesh.vertices,
                                               trans_affine)
    pred_mesh.vertices = v_pred_mesh_new
    # _, cost = pred_mesh.register(pred_mesh_partner)
    # print("Average distance after registration for mesh"
          # f" {mri_id}, {surf_name}: {cost}")

    # Compute ad, hd, percentage > 1mm, percentage > 2mm with pytorch3d
    pred_mesh = Meshes(
        [torch.from_numpy(pred_mesh.vertices).float().to(device)],
        [torch.from_numpy(pred_mesh.faces).long().to(device)]
    )
    pred_mesh_partner = Meshes(
        [torch.from_numpy(pred_mesh_partner.vertices).float().to(device)],
        [torch.from_numpy(pred_mesh_partner.faces).long().to(device)]
    )

    # compute with pytorch3d:
    pred_pcl = sample_points_from_meshes(pred_mesh, 100000)
    pred_pcl_partner = sample_points_from_meshes(pred_mesh_partner, 100000)

    # compute point to mesh distances and metrics; not exactly the same as
    # trimesh, it's always a bit larger than the trimesh distances, but a
    # lot faster.
    print(f"Computing point to mesh distances for files {pred_mesh_path} and"
          f" {pred_mesh_partner_path}...")
    P2G_dist = _point_mesh_face_distance_unidirectional(
        Pointclouds(pred_pcl_partner), pred_mesh
    ).cpu().numpy()
    G2P_dist = _point_mesh_face_distance_unidirectional(
        Pointclouds(pred_pcl), pred_mesh_partner
    ).cpu().numpy()

    assd2 = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])
    hd2 = max(np.percentile(P2G_dist, 90),
              np.percentile(G2P_dist, 90))
    greater_1 = ((P2G_dist > 1).sum() + (G2P_dist > 1).sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])
    greater_2 = ((P2G_dist > 2).sum() + (G2P_dist > 2).sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])

    print("\t > Average symmetric surface distance {:.4f}".format(assd2))
    print("\t > Hausdorff surface distance {:.4f}".format(hd2))
    print("\t > Greater 1 {:.4f}%".format(greater_1 * 100))
    print("\t > Greater 2 {:.4f}%".format(greater_2 * 100))

    return assd2, hd2, greater_1, greater_2

def trt_output(results, summary_file):
    ad = results[:, 0]
    hd = results[:, 1]
    greater_1 = results[:, 2]
    greater_2 = results[:, 3]

    cols_str = ';'.join(
        ['MEAN_OF_AD', 'STD_OF_AD', 'MEAN_OF_HD', 'STD_OF_HD',
         'MEAN_OF_>1', 'STD_OF_>1', 'MEAN_OF_>2', 'STD_OF_>2']
    )
    mets_str = ';'.join(
        [str(np.mean(ad)), str(np.std(ad)),
         str(np.mean(hd)), str(np.std(hd)),
         str(np.mean(greater_1)), str(np.std(greater_1)),
         str(np.mean(greater_2)), str(np.std(greater_2))]
    )

    with open(summary_file, 'w') as output_csv_file:
        output_csv_file.write(cols_str+'\n')
        output_csv_file.write(mets_str+'\n')

def eval_thickness(mri_id, surf_name, eval_params, epoch, device="cuda:1",
                       method="nearest", subfolder="meshes"):
    """ Cortical thickness biomarker.
    :param method: 'nearest' or 'ray'.
    """
    print("Evaluate thickness using " + method + " correspondences.")

    pred_folder = os.path.join(eval_params['log_path'])
    thickness_folder = os.path.join(
        eval_params['log_path'], 'thickness'
    )
    if not os.path.isdir(thickness_folder):
        os.mkdir(thickness_folder)

    # load ground-truth meshes
    try:
        gt_mesh_path = os.path.join(
            eval_params['gt_mesh_path'], mri_id, '{}.stl'.format(surf_name)
        )
        gt_mesh = trimesh.load(gt_mesh_path)
    except ValueError:
        gt_mesh_path = os.path.join(
            eval_params['gt_mesh_path'], mri_id, '{}.ply'.format(surf_name)
        )
        gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.remove_duplicate_faces()
    gt_mesh.remove_unreferenced_vertices()
    gt_pntcloud = gt_mesh.vertices
    gt_normals = gt_mesh.vertex_normals
    if "pial" in surf_name: # point to inside
        gt_normals = - gt_normals
    try:
        gt_mesh_path_partner = os.path.join(
            eval_params['gt_mesh_path'],
            mri_id,
            '{}.stl'.format(PARTNER[surf_name])
        )
        gt_mesh_partner = trimesh.load(gt_mesh_path_partner)
    except ValueError:
        gt_mesh_path_partner = os.path.join(
            eval_params['gt_mesh_path'],
            mri_id,
            '{}.ply'.format(PARTNER[surf_name])
        )
        gt_mesh_partner = trimesh.load(gt_mesh_path_partner)
    gt_mesh_partner.remove_duplicate_faces()
    gt_mesh_partner.remove_unreferenced_vertices()

    # Load predicted meshes
    s_index = SURF_NAMES.index(surf_name)
    pred_mesh_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh = trimesh.load(pred_mesh_path)
    pred_mesh.remove_duplicate_faces()
    pred_mesh.remove_unreferenced_vertices()
    pred_pntcloud = pred_mesh.vertices
    pred_normals = pred_mesh.vertex_normals
    if "pial" in surf_name: # point to inside
        pred_normals = - pred_normals
    s_index_partner = SURF_NAMES.index(PARTNER[surf_name])
    pred_mesh_path_partner = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index_partner}_meshpred.ply"
    )
    pred_mesh_partner = trimesh.load(pred_mesh_path_partner)
    pred_mesh_partner.remove_duplicate_faces()
    pred_mesh_partner.remove_unreferenced_vertices()

    if method == "ray":
        # Choose subset of predicted vertices and closest vertices from gt
        pred_pntcloud_sub, pred_idx = choose_n_random_points(
            pred_pntcloud, 10000, return_idx=True
        )
        pred_normals_sub = pred_normals[pred_idx]
        _, gt_idx, gt_pntcloud_sub = knn_points(
            torch.from_numpy(pred_pntcloud_sub)[None].float().to(device),
            torch.from_numpy(gt_pntcloud)[None].float().to(device),
            K=1,
            return_nn=True
        )
        gt_idx = gt_idx.squeeze().cpu().numpy()
        gt_pntcloud_sub = gt_pntcloud_sub.squeeze().cpu().numpy()
        gt_normals_sub = gt_normals[gt_idx]

        # Compute thickness measure using
        # trimesh.proximity.longest_ray
        print(f"Computing ray distances for file {pred_mesh_path}...")
        gt_thickness = longest_ray(gt_mesh_partner, gt_pntcloud_sub, gt_normals_sub)
        pred_thickness = longest_ray(pred_mesh_partner, pred_pntcloud_sub, pred_normals_sub)

        # Set inf values to nan
        gt_thickness[~np.isfinite(gt_thickness)] = np.nan
        pred_thickness[~np.isfinite(pred_thickness)] = np.nan

        error = np.abs(pred_thickness - gt_thickness)

    elif method == "nearest":
        # Use all vertices
        pred_idx = np.array(range(pred_pntcloud.shape[0]))
        gt_idx = np.array(range(gt_pntcloud.shape[0]))

        # Move to gpu
        gt_pntcloud = Pointclouds(
            [torch.from_numpy(gt_pntcloud).float().to(device)]
        )
        gt_mesh_partner = Meshes(
            [torch.from_numpy(gt_mesh_partner.vertices).float().to(device)],
            [torch.from_numpy(gt_mesh_partner.faces).long().to(device)],
        )
        pred_pntcloud = Pointclouds(
            [torch.from_numpy(pred_pntcloud).float().to(device)]
        )
        pred_mesh_partner = Meshes(
            [torch.from_numpy(pred_mesh_partner.vertices).float().to(device)],
            [torch.from_numpy(pred_mesh_partner.faces).long().to(device)],
        )

        # Compute thickness measure using nearest face distance
        print(f"Computing nearest distances for file {pred_mesh_path}...")
        gt_thickness = _point_mesh_face_distance_unidirectional(
            gt_pntcloud, gt_mesh_partner
        ).squeeze().cpu().numpy()
        pred_thickness = _point_mesh_face_distance_unidirectional(
            pred_pntcloud, pred_mesh_partner
        ).squeeze().cpu().numpy()

        # Compute error w.r.t. to nearest gt vertex
        _, nearest_idx, _ = knn_points(
            pred_pntcloud.points_padded(),
            gt_pntcloud.points_padded(),
            K=1,
            return_nn=True
        )
        nearest_idx = nearest_idx.squeeze().cpu().numpy()
        error = np.abs(pred_thickness - gt_thickness[nearest_idx])

        pred_pntcloud = pred_pntcloud.points_packed().cpu().numpy()
        gt_pntcloud = gt_pntcloud.points_packed().cpu().numpy()

    else:
        raise ValueError("Unknown method {}.".format(method))

    error_mean = np.nanmean(error)
    error_median = np.nanmedian(error)

    print("\t > Thickness error mean {:.4f}".format(error_mean))
    print("\t > Thickness error median {:.4f}".format(error_median))

    # Store
    th_gt_file = os.path.join(
        thickness_folder, f"{mri_id}_struc{s_index}_gt.thickness"
    )
    np.save(th_gt_file, gt_thickness)
    th_pred_file = os.path.join(
        thickness_folder, f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.thickness"
    )
    np.save(th_pred_file, pred_thickness)
    err_file = os.path.join(
        thickness_folder, f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.thicknesserror"
    )
    error_features = np.zeros(pred_pntcloud.shape[0])
    error_features[pred_idx] = error
    np.nan_to_num(error_features, copy=False, nan=0.0)
    np.save(err_file, error_features)

    return error_mean, error_median

def thickness_output(results, summary_file):
    means_all = results[:, 0]
    medians_all = results[:, 1]
    mean_of_means = np.mean(means_all)
    std_of_means = np.std(means_all)
    mean_of_medians = np.mean(medians_all)
    std_of_medians = np.std(medians_all)

    cols_str = ';'.join(
        ['MEAN_OF_MEANS', 'STD_OF_MEANS', 'MEAN_OF_MEDIANS', 'STD_OF_MEDIANS']
    )
    mets_str = ';'.join(
        [str(mean_of_means), str(std_of_means),
         str(mean_of_medians), str(std_of_medians)]
    )

    with open(summary_file, 'w') as output_csv_file:
        output_csv_file.write(cols_str+'\n')
        output_csv_file.write(mets_str+'\n')

def eval_ad_hd_pytorch3d(mri_id, surf_name, eval_params, epoch,
                         device="cuda:1", subfolder="meshes"):
    """ AD and HD computed with pytorch3d. """
    pred_folder = os.path.join(eval_params['log_path'])
    ad_hd_folder = os.path.join(
        eval_params['log_path'], 'ad_hd'
    )
    if not os.path.isdir(ad_hd_folder):
        os.mkdir(ad_hd_folder)

    # load ground-truth mesh
    try:
        gt_mesh_path = os.path.join(eval_params['gt_mesh_path'], mri_id, '{}.stl'.format(surf_name))
        gt_mesh = trimesh.load(gt_mesh_path)
    except ValueError:
        gt_mesh_path = os.path.join(eval_params['gt_mesh_path'], mri_id, '{}.ply'.format(surf_name))
        gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.remove_duplicate_faces(); gt_mesh.remove_unreferenced_vertices();
    gt_mesh = Meshes(
        [torch.from_numpy(gt_mesh.vertices).float().to(device)],
        [torch.from_numpy(gt_mesh.faces).long().to(device)]
    )

    # load predicted mesh
    # file endings depending on the post-processing:
    # orig, pp, top_fix
    s_index = SURF_NAMES.index(surf_name)
    pred_mesh_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh = trimesh.load(pred_mesh_path)
    pred_mesh.remove_duplicate_faces(); pred_mesh.remove_unreferenced_vertices();
    pred_mesh = Meshes(
        [torch.from_numpy(pred_mesh.vertices).float().to(device)],
        [torch.from_numpy(pred_mesh.faces).long().to(device)]
    )

    # compute with pytorch3d:
    gt_pcl = sample_points_from_meshes(gt_mesh, 100000)
    pred_pcl = sample_points_from_meshes(pred_mesh, 100000)

    # compute point to mesh distances and metrics; not exactly the same as
    # trimesh, it's always a bit larger than the trimesh distances, but a
    # lot faster.
    print(f"Computing point to mesh distances for file {pred_mesh_path}...")
    P2G_dist = _point_mesh_face_distance_unidirectional(
        Pointclouds(gt_pcl), pred_mesh
    ).cpu().numpy()
    G2P_dist = _point_mesh_face_distance_unidirectional(
        Pointclouds(pred_pcl), gt_mesh
    ).cpu().numpy()

    assd2 = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])
    hd2 = max(np.percentile(P2G_dist, 90),
              np.percentile(G2P_dist, 90))

    print("\t > Average symmetric surface distance {:.4f}".format(assd2))
    print("\t > Hausdorff surface distance {:.4f}".format(hd2))

    ad_pred_file = os.path.join(
        ad_hd_folder, f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ad.npy"
    )
    np.save(ad_pred_file, assd2)
    hd_pred_file = os.path.join(
        ad_hd_folder, f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.hd.npy"
    )
    np.save(hd_pred_file, hd2)

    return assd2, hd2

def eval_ad_hd_trimesh(mri_id, surf_name, eval_params, epoch, subfolder="meshes"):

    print('>>' * 5 + " Evaluating mri {} and surface {}".format(mri_id, surf_name))
    pred_folder = os.path.join(eval_params['log_path'])

    # load ground truth
    gt_pcl, gt_pcl_path, gt_mesh_path = None, None, None

    # load ground-truth mesh
    try:
        gt_mesh_path = os.path.join(eval_params['gt_mesh_path'], mri_id, '{}.stl'.format(surf_name))
        gt_mesh = trimesh.load(gt_mesh_path)
    except ValueError:
        gt_mesh_path = os.path.join(eval_params['gt_mesh_path'], mri_id, '{}.ply'.format(surf_name))
        gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.remove_duplicate_faces(); gt_mesh.remove_unreferenced_vertices();
    print("GT mesh loaded from {} with {} vertices and {} faces".format(
        gt_mesh_path, gt_mesh.vertices.shape, gt_mesh.faces.shape))
    # sample point cloud for ground-truth mesh
    gt_pcl = gt_mesh.sample(100000)
    print("Point cloud with {} dimensions sampled from ground-truth mesh".format(gt_pcl.shape))

    # load predicted mesh
    # file endings depending on the post-processing:
    # orig, pp, top_fix
    s_index = SURF_NAMES.index(surf_name)
    pred_mesh_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh = trimesh.load(pred_mesh_path)
    pred_mesh.remove_duplicate_faces(); pred_mesh.remove_unreferenced_vertices();
    print("Predicted mesh loaded from {} with {} vertices and {} faces".format(
        pred_mesh_path, pred_mesh.vertices.shape, pred_mesh.faces.shape))
    # sampling point cloud in predicted mesh
    pred_pcl = pred_mesh.sample(100000)
    print("Point cloud with {} dimensions sampled from predicted mesh".format(pred_pcl.shape))

    # compute point to mesh distances and metrics
    print(f"Computing point to mesh distances for file {pred_mesh_path}...")
    _, P2G_dist, _ = trimesh.proximity.closest_point(pred_mesh, gt_pcl)
    _, G2P_dist, _ = trimesh.proximity.closest_point(gt_mesh, pred_pcl)
    print("point to mesh distances computed")
    #Average symmetric surface distance
    print("computing metrics...")
    assd = ((P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.size + G2P_dist.size))
    print("\t > Average symmetric surface distance {:.4f}".format(assd))
    # Hausdorff distance
    hd = max(np.percentile(P2G_dist, 90), np.percentile(G2P_dist, 90))
    print("\t > Hausdorff surface distance {:.4f}".format(hd))

    # log and metrics write csv
    cols_str = ';'.join(['MRI_ID', 'SURF_NAME', 'ASSD', 'HD'])
    mets_str = ';'.join([mri_id, surf_name, str(assd), str(hd)])
    print('REPORT_COLS;{}'.format(cols_str))
    print('REPORT_VALS;{}'.format(mets_str))
    met_csv_file_path = os.path.join(eval_params['log_path'], "{}_{}_{}.csv".format(eval_params['metrics_csv_prefix'], mri_id, surf_name))
    with open(met_csv_file_path, 'w') as output_csv_file:
        output_csv_file.write(mets_str+'\n')
    print('>>' * 5 + " Evaluation for {} and {}".format(mri_id, surf_name))

    return assd, hd

def ad_hd_output(results, summary_file):
    assd_all = results[:, 0]
    hd_all = results[:, 1]
    assd_mean = np.mean(assd_all)
    assd_std = np.std(assd_all)
    hd_mean = np.mean(hd_all)
    hd_std = np.std(hd_all)

    cols_str = ';'.join(['AD_MEAN', 'AD_STD', 'HD_MEAN', 'HD_STD'])
    mets_str = ';'.join([str(assd_mean), str(assd_std), str(hd_mean), str(hd_std)])

    with open(summary_file, 'w') as output_csv_file:
        output_csv_file.write(cols_str+'\n')
        output_csv_file.write(mets_str+'\n')

mode_to_function = {"ad_hd": eval_ad_hd_pytorch3d,
                    "thickness": eval_thickness,
                    "trt": eval_trt}
mode_to_output_file = {"ad_hd": ad_hd_output,
                       "thickness": thickness_output,
                       "trt": trt_output}


if __name__ == '__main__':
    argparser = ArgumentParser(description="Mesh evaluation procedure")
    argparser.add_argument('exp_name',
                           type=str,
                           help="Name of experiment under evaluation.")
    argparser.add_argument('epoch',
                           type=int,
                           help="The epoch to evaluate.")
    argparser.add_argument('n_test_vertices',
                           type=int,
                           help="The number of template vertices for each"
                           " structure that was used during testing.")
    argparser.add_argument('dataset',
                           type=str,
                           help="The dataset.")
    argparser.add_argument('mode',
                           type=str,
                           help="The evaluation to perform, possible values"
                           " are " + str(MODES))
    argparser.add_argument('--meshfixed',
                           action='store_true',
                           help="Use MeshFix'ed meshes for evaluation.")

    args = argparser.parse_args()
    exp_name = args.exp_name
    epoch = args.epoch
    mode = args.mode
    dataset = args.dataset
    meshfixed = args.meshfixed

    # Provide params
    eval_params = {}
    if "OASIS" in dataset:
        subdir = "CSR_data"
    else:
        subdir = ""
    eval_params['gt_mesh_path'] = os.path.join(
        RAW_DATA_DIR,
        dataset.replace("_small", "").replace("_large", "").replace("_orig", ""),
        subdir
    )
    eval_params['exp_path'] = os.path.join(EXPERIMENT_DIR, exp_name)
    eval_params['log_path'] = os.path.join(
        EXPERIMENT_DIR, exp_name,
        "test_template_"
        + str(args.n_test_vertices)
        + f"_{dataset}"
    )
    eval_params['metrics_csv_prefix'] = "eval_" + mode

    if meshfixed:
        eval_params['metrics_csv_prefix'] += "_meshfixed"
        subfolder = "meshfix"
    else:
        subfolder = "meshes"

    # Read dataset split
    dataset_file = os.path.join(eval_params['exp_path'], 'dataset_ids.txt')

    # Use all valid ids in the case of test-retest (everything is 'test') and
    # test split otherwise
    if mode == 'trt':
        ids = valid_ids(eval_params['gt_mesh_path'])
    else:
        ids = read_dataset_ids(dataset_file)

    res_all = []
    res_surfaces = {s: [] for s in SURF_NAMES}
    for mri_id in ids:
        for surf_name in SURF_NAMES:
            result = mode_to_function[mode](
                mri_id, surf_name, eval_params, epoch, subfolder=subfolder
            )
            if result is not None:
                res_all.append(result)
                res_surfaces[surf_name].append(result)

    # Averaged over surfaces
    summary_file = os.path.join(
        eval_params['log_path'],
        f"{eval_params['metrics_csv_prefix']}_summary.csv"
    )
    mode_to_output_file[mode](np.array(res_all), summary_file)

    # Per-surface results
    for surf_name in SURF_NAMES:
        summary_file = os.path.join(
            eval_params['log_path'],
            f"{surf_name}_{eval_params['metrics_csv_prefix']}_summary.csv"
        )
        mode_to_output_file[mode](np.array(res_surfaces[surf_name]), summary_file)

    print("Done.")
