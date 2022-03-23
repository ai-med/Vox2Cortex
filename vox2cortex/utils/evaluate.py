""" Evaluation of a model """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import os
import logging
import glob
from collections.abc import Sequence

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

from utils.modes import ExecModes
from utils.eval_metrics import EvalMetricHandler
from utils.utils import (
    create_mesh_from_voxels,
)
from utils.coordinate_transform import (
    transform_mesh_affine,
    unnormalize_vertices_per_max_dim,
)
from utils.mesh import Mesh
from utils.logging import (
    write_img_if_debug
)
from utils.cortical_thickness import cortical_thickness

def add_to_results_(result_dict, metric_name, result):
    """ Helper function to add evaluation results to the result dict."""
    # Extract atomic results
    if isinstance(result, Sequence):
        if len(result) == 1:
            result = result[0]

    # Add to dict
    if metric_name not in result_dict:
        result_dict[metric_name] = []
    if not isinstance(result, Sequence): # Atomic result
        result_dict[metric_name].append(result)
    else: # Per-structure result
        result_dict[metric_name].append(np.mean(result))
        for i, res in enumerate(result):
            name = metric_name + f"_Struc{i}"
            if name not in result_dict:
                result_dict[name] = []
            result_dict[name].append(res)


class ModelEvaluator():
    """ Class for evaluation of models.

    :param eval_dataset: The dataset split that should be used for evaluation.
    :param save_dir: The experiment directory where data can be saved.
    :param n_v_classes: Number of vertex classes.
    :param n_m_classes: Number of mesh classes.
    :param eval_metrics: A list of metrics to use for evaluation.
    :param mc_step_size: Marching cubes step size.
    """
    def __init__(self, eval_dataset, save_dir, n_v_classes, n_m_classes, eval_metrics,
                 mc_step_size=1, **kwargs):
        self._dataset = eval_dataset
        self._save_dir = save_dir
        self._n_v_classes = n_v_classes
        self._n_m_classes = n_m_classes
        self._eval_metrics = eval_metrics
        self._mc_step_size = mc_step_size

        self._mesh_dir = os.path.join(self._save_dir, "meshes")
        if not os.path.isdir(self._mesh_dir):
            os.mkdir(self._mesh_dir)

    def evaluate(self, model, epoch, save_meshes=5,
                 remove_previous_meshes=True, store_in_orig_coords=False):
        """ Evaluate a given model and optionally store predicted meshes. """

        results_all = {}
        model_class = model.__class__

        # Iterate over data split
        for i in tqdm(range(len(self._dataset)), desc="Evaluate..."):
            data = self._dataset.get_item_and_mesh_from_index(i)
            write_img_if_debug(data[1].squeeze().cpu().numpy(),
                               "../misc/raw_voxel_target_img_eval.nii.gz")
            write_img_if_debug(data[0].squeeze().cpu().numpy(),
                               "../misc/raw_voxel_input_img_eval.nii.gz")
            with torch.no_grad():
                pred = model(data[0][None].cuda())

            for metric in self._eval_metrics:
                res = EvalMetricHandler[metric](pred, data,
                                                self._n_v_classes,
                                                self._n_m_classes,
                                                model_class)
                add_to_results_(results_all, metric, res)

            if i < save_meshes: # Store meshes for visual inspection
                filename =\
                        self._dataset.get_file_name_from_index(i).split(".")[0]
                self.store_meshes(
                    pred, data, filename, epoch, model_class,
                    remove_previous=remove_previous_meshes,
                    convert_to_orig_coords=store_in_orig_coords
                )

        # Just consider means over evaluation set
        results = {k: np.mean(v) for k, v in results_all.items()}

        return results

    def store_meshes(self, pred, data, filename, epoch, model_class,
                     show_all_steps=False, remove_previous=True,
                     convert_to_orig_coords=False):
        """ Save predicted meshes and ground truth
        """
        if "/" in filename:
            subdir = os.path.join(self._mesh_dir, filename.split("/")[0])
            if not os.path.isdir(subdir):
                os.mkdir(subdir)

        # Remove previously stored files to avoid dumping storage
        if remove_previous:
            for suffix in ("*_meshpred.ply", "*_voxelpred.ply",
                           "*_meshpred.png", "*_voxelpred.png"):
                files_to_delete = glob.glob(os.path.join(
                    self._mesh_dir, filename + suffix
                ))
                for f in files_to_delete:
                    try:
                        os.remove(f)
                    except:
                        print("Error while deleting file ", f)
        # Data
        img = data[0].squeeze()
        if img.ndim == 3:
            img_filename = filename + "_mri.nii.gz"
            img_filename = os.path.join(self._mesh_dir, img_filename)
            if not os.path.isfile(img_filename):
                nib_img = nib.Nifti1Image(img.cpu().numpy(), np.eye(4))
                nib.save(nib_img, img_filename)

        # Label
        gt_mesh = data[2]
        ndims = gt_mesh.ndims
        logging.getLogger(ExecModes.TEST.name).debug(
            "%d vertices in ground truth mesh",
            len(gt_mesh.vertices.view(-1, ndims))
        )
        # Store ground truth if it does not exist yet
        if ndims == 3:
            trans_affine = data[3]
            # Back to original coordinate space
            if convert_to_orig_coords:
                new_vertices, new_faces = transform_mesh_affine(
                    gt_mesh.vertices, gt_mesh.faces, np.linalg.inv(trans_affine)
                )
            else:
                new_vertices, new_faces = gt_mesh.vertices, gt_mesh.faces
            gt_mesh_transformed = Mesh(new_vertices, new_faces, features=gt_mesh.features)
            gt_filename = filename + "_gt.ply"
            gt_filename = os.path.join(self._mesh_dir, gt_filename)
            if not os.path.isfile(gt_filename):
                if self._n_m_classes in (2, 4):
                    gt_mesh_transformed.store_with_features(gt_filename)
                else:
                    gt_mesh_transformed.store(gt_filename)
        else:
            raise ValueError("Wrong dimensionality.")

        # Mesh prediction
        vertices, faces = model_class.pred_to_verts_and_faces(pred)
        if show_all_steps:
            # Visualize meshes of all steps
            for s, (v_, f_) in enumerate(zip(vertices, faces)):
                v, f = v_.squeeze(), f_.squeeze()
                # Optionally convert to original coordinate space
                if convert_to_orig_coords:
                    assert ndims == 3, "Only for 3 dim meshes."
                    v, f = transform_mesh_affine(
                        v, f, np.linalg.inv(trans_affine)
                    )
                if self._n_m_classes in (2, 4) and convert_to_orig_coords:
                    # Meshes with thickness
                    pred_meshes = cortical_thickness(v, f)
                    # Store the mesh of each structure separately
                    for i, m in enumerate(pred_meshes):
                        pred_mesh_filename =\
                                filename + "_epoch" + str(epoch) + "_step" +\
                                str(s) + "_struc" + str(i) +\
                                "_meshpred.ply"
                        pred_mesh_filename = os.path.join(self._mesh_dir,
                                                          pred_mesh_filename)
                        m.store_with_features(pred_mesh_filename)
                else: # Do not consider structures separately
                    pred_mesh = Mesh(v.cpu(), f.cpu())
                    logging.getLogger(ExecModes.TEST.name).debug(
                        "%d vertices in predicted mesh", len(v.view(-1, ndims))
                    )
                    if ndims == 3:
                        pred_mesh_filename = filename + "_epoch" + str(epoch) +\
                            "_step" + str(s) + "_meshpred.ply"
                        pred_mesh_filename = os.path.join(self._mesh_dir,
                                                          pred_mesh_filename)
                        pred_mesh.store(pred_mesh_filename)
                    else: # 2D
                        pred_mesh_filename = filename + "_epoch" + str(epoch) +\
                            "_step" + str(s) + "_meshpred.png"
                        pred_mesh_filename = os.path.join(self._mesh_dir,
                                                          pred_mesh_filename)
                        pred_mesh = pred_mesh.to_pytorch3d_Meshes()
                        show_img_with_contour(
                            img,
                            unnormalize_vertices_per_max_dim(
                                pred_mesh.verts_packed(), img.shape
                            ),
                            pred_mesh.faces_packed(),
                            pred_mesh_filename
                        )
        else:
            # Only visualize last step
            v, f = vertices[-1].squeeze(), faces[-1].squeeze()
            # Optionally transform back to original coordinates
            if convert_to_orig_coords:
                assert ndims == 3, "Only for 3 dim meshes."
                v, f = transform_mesh_affine(
                    v, f, np.linalg.inv(trans_affine)
                )
            if self._n_m_classes in (2, 4) and convert_to_orig_coords:
                # Meshes with thickness
                pred_meshes = cortical_thickness(v, f)
                # Store the mesh of each structure separately
                for i, m in enumerate(pred_meshes):
                    pred_mesh_filename = filename + "_epoch" + str(epoch) +\
                        "_struc" + str(i) + "_meshpred.ply"
                    pred_mesh_filename = os.path.join(self._mesh_dir,
                                                      pred_mesh_filename)
                    m.store_with_features(pred_mesh_filename)
            else: # Do not consider structures separately
                pred_mesh = Mesh(v.cpu(), f.cpu())
                logging.getLogger(ExecModes.TEST.name).debug(
                    "%d vertices in predicted mesh", len(v.view(-1, ndims))
                )
                if ndims == 3:
                    pred_mesh_filename = filename + "_epoch" + str(epoch) +\
                        "_meshpred.ply"
                    pred_mesh_filename = os.path.join(self._mesh_dir,
                                                      pred_mesh_filename)
                    pred_mesh.store(pred_mesh_filename)
                else: # 2D
                    pred_mesh_filename = filename + "_epoch" + str(epoch) +\
                        "_meshpred.png"
                    pred_mesh_filename = os.path.join(self._mesh_dir,
                                                      pred_mesh_filename)
                    pred_mesh = pred_mesh.to_pytorch3d_Meshes()
                    show_img_with_contour(
                        img,
                        unnormalize_vertices_per_max_dim(
                            pred_mesh.verts_packed(),
                            img.shape
                        ),
                        pred_mesh.faces_packed(),
                        pred_mesh_filename
                    )

        # Voxel prediction
        voxel_pred = model_class.pred_to_voxel_pred(pred)
        if voxel_pred is not None: # voxel_pred can be empty
            for c in range(1, self._n_v_classes):
                voxel_pred_class = voxel_pred.squeeze()
                voxel_pred_class[voxel_pred_class != c] = 0
                if ndims == 3:
                    pred_voxel_filename = filename + "_epoch" + str(epoch) +\
                        "_class" + str(c) + "_voxelpred.ply"
                    pred_voxel_filename = os.path.join(self._mesh_dir,
                                                       pred_voxel_filename)
                    try:
                        mc_pred_mesh = create_mesh_from_voxels(
                            voxel_pred_class, self._mc_step_size
                        ).to_trimesh(process=True)
                        if convert_to_orig_coords:
                            v, f = transform_mesh_affine(
                                mc_pred_mesh.vertices,
                                mc_pred_mesh.faces,
                                np.linalg.inv(trans_affine)
                            )
                            mc_pred_mesh = Mesh(v, f).to_trimesh()
                        mc_pred_mesh.export(pred_voxel_filename)
                    except ValueError as e:
                        logging.getLogger(ExecModes.TEST.name).warning(
                               "In voxel prediction for file: %s: %s."
                               " This means usually that the prediction"
                               " is all 1.", filename, e)
                    except RuntimeError as e:
                        logging.getLogger(ExecModes.TEST.name).warning(
                               "In voxel prediction for file: %s: %s ",
                               filename, e)
                    except AttributeError:
                        # No voxel prediction exists
                        pass
                else:
                    raise ValueError("Wrong dimensionality.")
