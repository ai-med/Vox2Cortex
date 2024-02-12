""" Evaluation of a model """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import re
import os
import glob
from collections.abc import Sequence

import numpy as np
import torch
import trimesh
import pandas as pd
import nibabel as nib
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from pytorch3d.structures import MeshesXD

import logger

from models.vox2cortex import Vox2Cortex
from utils.template import MeshTemplate
from utils.eval_metrics import MeshDice
from utils.coordinate_transform import normalize_vertices
from utils.utils import update_dict
from utils.mesh import register_mesh_to_voxel_seg


log = logger.get_std_logger(__name__)


class ModelEvaluator:
    """Class for evaluation of models.

    :param eval_dataset: The dataset split that should be used for evaluation.
    :param mesh_template: The template for the deformations.
    :param save_dir: The experiment directory where data can be saved.
    :param n_v_classes: Number of vertex classes.
    :param n_m_classes: Number of mesh classes.
    :param eval_metrics: A list of metrics to use for evaluation.
    """

    def __init__(
        self,
        eval_dataset,
        mesh_template: MeshTemplate,
        save_dir,
        n_v_classes,
        n_m_classes,
        eval_metrics,
        **kwargs,
    ):
        self._dataset = eval_dataset
        self._save_dir = save_dir
        self._n_v_classes = n_v_classes
        self._n_m_classes = n_m_classes
        self._eval_metrics = eval_metrics
        self._mesh_template = mesh_template

    def evaluate(
        self,
        model,
        epoch,
        device,
        register_meshes_to_voxels=False,
        save_predictions=5,
        remove_previous_meshes=True
    ):
        """Evaluate a model at a certain epoch."""

        results_all = pd.DataFrame(columns=("ID", "Tissue", "Metric", "Value"))

        # Decide whether to run the model or just load the meshes
        mesh_0_name = list(self._dataset.mesh_label_names.keys())[0]
        file_0 = os.path.join(
            self._save_dir,
            self._dataset.get_ID_from_index(0),
            f"{mesh_0_name}_epoch_{epoch}.ply"
        )
        if os.path.exists(file_0):
            log.info(f"Found {file_0}, loading data for evaluation")
            run_model = False
        else:
            log.info("Running model for evaluation")
            run_model = True
            if isinstance(model, DDP):
                model_class = model.module.__class__
            else:
                model_class = model.__class__

            # Batch size fixed for Vox2Cortex
            if model_class == Vox2Cortex:
                if isinstance(model, DDP):
                    batch_size = model.module.batch_size
                else:
                    batch_size = model.batch_size
            else:
                batch_size = 1

        # Iterate over data split
        for i in tqdm(
            range(len(self._dataset)), desc=f"Evaluate on {device}..."
        ):
            filename = self._dataset.get_ID_from_index(i)
            data = self._dataset.get_data_element(i)

            # Affine from normalized image space to scanner RAS
            vox2ras = torch.inverse(
                torch.tensor(
                    data["trans_affine_label"], device=device
                ).float()
            )

            if run_model:
                # Generate prediction and store
                input_img = nib.Nifti1Image(
                    data["img"].cpu().numpy(),
                    np.eye(4)
                )
                if logger.debug():
                    nib.save(
                        input_img,
                        os.path.join(
                            logger.get_log_dir(),
                            f"eval_input_img_{filename}.nii.gz"
                        )
                    )
                voxel_label = nib.Nifti1Image(
                    data["voxel_label"].cpu().numpy(),
                    np.eye(4)
                )
                if logger.debug():
                    nib.save(
                        voxel_label,
                        os.path.join(
                            logger.get_log_dir(), f"eval_label_{filename}.nii.gz"
                        )
                    )

                with torch.no_grad():
                    # Even though if batch_size is > 1, only one subject is
                    # processed at a time
                    img = data["img"].float().to(device)
                    input_img = img.expand(batch_size, *img.shape)
                    input_meshes = self._mesh_template.create_template_batch_size(
                        batch_size, device=device
                    )

                    # Model prediction; we need to use model.module in DDP
                    # models to avoid problems in distributed training,
                    # see also
                    # https://discuss.pytorch.org/t/distributeddataparallel-barrier-doesnt-work-as-expected-during-evaluation/99867
                    model_fwd = model.module if (
                        isinstance(model, DDP)
                    ) else model
                    pred = model_fwd(input_img, input_meshes)

                # First mesh in the batch from the final deformation step
                mesh_pred = model_class.pred_to_final_mesh_pred(pred)
                mesh_pred = MeshesXD(
                    mesh_pred.verts_list()[: self._n_m_classes],
                    mesh_pred.faces_list()[: self._n_m_classes],
                    X_dims=(self._n_m_classes,),
                )

                mesh_pred = mesh_pred.transform(vox2ras)
                # Undo padding etc.
                try:
                    voxel_pred = model_class.pred_to_voxel_pred(pred)[0]
                    voxel_pred = self._dataset.label_to_original_size(voxel_pred)
                except TypeError:
                    # Not all models produce a voxel pred
                    voxel_pred = None

                # Optimize mesh prediction based on voxel prediction
                if register_meshes_to_voxels:
                    mesh_pred = register_mesh_to_voxel_seg(
                        mesh_pred,
                        list(self._dataset.mesh_label_names.keys()),
                        voxel_pred,
                        list(self._dataset.voxel_label_names.keys()),
                        self._dataset.image_affine(i)
                    )

                # Save
                if i < save_predictions:
                    self.save_pred(
                        i,
                        mesh_pred,
                        voxel_pred,
                        epoch,
                        remove_previous_meshes
                    )

            else:
                # Load prediction
                verts, faces = [], []
                for mn in list(self._dataset.mesh_label_names.keys()):
                    mesh = trimesh.load(
                        os.path.join(self._save_dir, filename, mn + f"_epoch_{epoch}.ply"),
                        process=False
                    )
                    verts.append(torch.tensor(mesh.vertices).float())
                    faces.append(torch.tensor(mesh.faces).long())
                mesh_pred = MeshesXD(
                    verts, faces, X_dims=(self._n_m_classes)
                ).to(device)
                try:
                    voxel_pred = nib.load(
                        os.path.join(
                            self._save_dir,
                            filename,
                            f"pred_epoch_{epoch}.nii.gz"
                        )
                    ).get_fdata()
                    voxel_pred = torch.tensor(voxel_pred).to(device)
                except FileNotFoundError:
                    voxel_pred = None

            # Ground truth
            mesh_norm_space = data["mesh_label"].to(device)
            mesh_gt = (
                MeshesXD(
                    mesh_norm_space.verts_list(),
                    mesh_norm_space.faces_list(),
                    X_dims=[mesh_norm_space.verts_padded().shape[0]],
                ).to(device).transform(vox2ras)  # --> scanner RAS
            )
            voxel_gt = self._dataset.label_to_original_size(
                data["voxel_label"]
            ).to(device)

            # Pred mesh in normalized image coordinates
            dtype = mesh_pred.verts_packed().dtype
            mesh_img_coo = mesh_pred.clone().transform(
                torch.tensor(
                    np.linalg.inv(self._dataset.image_affine(i)),
                    dtype=dtype,
                    device=device
                )
            )
            _, _, affine = normalize_vertices(
                mesh_img_coo.verts_list()[0].cpu().numpy(),  # dummy
                voxel_gt.shape,
                mesh_img_coo.faces_list()[0].cpu().numpy(),  # dummy
                return_affine=True
            )
            mesh_pred_img_norm_coo = mesh_img_coo.transform(
                torch.tensor(affine, dtype=dtype, device=device)
            )

            for metric in self._eval_metrics:
                res = metric(
                    mesh_pred_img_norm_coo if (
                        isinstance(metric, MeshDice)
                    ) else mesh_pred,
                    mesh_gt,
                    self._n_m_classes,
                    list(self._dataset.mesh_label_names.keys()),
                    voxel_pred,
                    voxel_gt,
                    self._n_v_classes,
                    list(self._dataset.voxel_label_names.keys()),
                )
                res = pd.DataFrame(res)
                res["ID"] = filename
                results_all = pd.concat([results_all, res], ignore_index=True)

            # Free memory
            if run_model:
                del pred
                torch.cuda.empty_cache()

        return results_all

    def save_pred(
        self,
        index,
        mesh_pred,
        voxel_pred,
        epoch,
        remove_previous_meshes
    ):
        """ Save mesh and voxel prediction """
        filename = self._dataset.get_ID_from_index(index)
        subdir = os.path.join(self._save_dir, filename)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
        if remove_previous_meshes:
            previous_meshes = glob.glob(os.path.join(subdir, "*_epoch_*.ply"))
            previous_segs = glob.glob(os.path.join(subdir, "*_epoch_*.nii.gz"))
            for mn in previous_meshes:
                os.remove(mn)
            for seg in previous_segs:
                os.remove(seg)
        if voxel_pred is not None:
            pred_nifti = nib.Nifti1Image(
                voxel_pred.cpu().numpy(),
                self._dataset.image_affine(index),
            )
            nii_fn = os.path.join(subdir, f"pred_epoch_{epoch}.nii.gz")
            nib.save(pred_nifti, nii_fn)
        for name, v, f in zip(
            list(self._dataset.mesh_label_names.keys()),
            mesh_pred.verts_list(),
            mesh_pred.faces_list(),
        ):
            trimesh.Trimesh(
                v.cpu().numpy(), f.cpu().numpy(), process=False
            ).export(os.path.join(subdir, name + f"_epoch_{epoch}.ply"))
