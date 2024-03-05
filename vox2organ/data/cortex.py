
""" Cortex dataset handler """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import re
import os
from typing import Union, Sequence
from enum import IntEnum

import trimesh
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

import logger
from data.image_and_mesh_dataset import ImageAndMeshDataset
from utils.modes import DataModes
from utils.mesh import Mesh

log = logger.get_std_logger(__name__)

class CortexLabels(IntEnum):
    """ Mapping IDs in segmentation masks to names.
    """
    right_white_matter = 41
    left_white_matter = 2
    left_cerebral_cortex = 3
    right_cerebral_cortex = 42


class CortexDataset(ImageAndMeshDataset):
    """ Cortex dataset

    This dataset contains images and meshes and has additional functionality
    specifically for cortex data.

    :param structure_type: Either 'cerebral_cortex' (outer cortex surfaces)
    or 'white_matter' (inner cortex surfaces) or both
    :param reduced_freesurfer: The factor of reduced freesurfer meshes, e.g.,
    0.3
    :param morph_data_dir: A directory containing morphological brain data,
    e.g. thickness values.
    :param kwargs: Parameters for ImageAndMeshDataset
    """

    image_file_name = "mri.nii.gz"
    seg_file_name = "aseg.nii.gz" # For FS segmentations

    LabelMap = CortexLabels

    def _get_seg_and_mesh_label_names(self, structure_type):
        if structure_type == "rh-pial-only":
            seg_label_names = {
                "gray_matter": ("right_cerebral_cortex",),
            }
            mesh_label_names = {
                "rh_pial": "rh_pial",
            }
        elif structure_type == "rh-only":
            seg_label_names = {
                "gray_matter": ("right_cerebral_cortex",),
                "white_matter": ("right_white_matter",),
            }
            mesh_label_names = {
                "rh_white": "rh_white",
                "rh_pial": "rh_pial"
            }
        elif structure_type == "pial-only":
            seg_label_names = {
                "gray_matter": ("left_cerebral_cortex", "right_cerebral_cortex"),
            }
            mesh_label_names = {
                "lh_pial": "lh_pial",
                "rh_pial": "rh_pial"
            }
        elif structure_type == "rh-wm-only":
            seg_label_names = {
                "white_matter": ("right_white_matter",),
            }
            mesh_label_names = {
                "rh_white": "rh_white",
            }
        elif structure_type == "wm-only":
            seg_label_names = {
                "white_matter": ("left_white_matter", "right_white_matter"),
            }
            mesh_label_names = {
                "lh_white": "lh_white",
                "rh_white": "rh_white"
            }
        elif structure_type == "cortex-all":
            seg_label_names = {
                "white_matter": ("left_white_matter", "right_white_matter"),
                "gray_matter": ("left_cerebral_cortex", "right_cerebral_cortex"),
            }
            mesh_label_names = {
                "lh_white": "lh_white",
                "rh_white": "rh_white",
                "lh_pial": "lh_pial",
                "rh_pial": "rh_pial"
            }
        else:
            raise ValueError("Unknown structure type.")

        if self.reduced_gt:
            log.info('Using reduced FS labels!')
            for k, v in mesh_label_names.items():
                mesh_label_names[k] = v + "_reduced_0.3"

        return seg_label_names, mesh_label_names

    def __init__(
        self,
        structure_type: str,
        **kwargs
    ):

        self.reduced_gt = kwargs.get('reduced_gt', False)

        # Map structure type to (file-)names
        (self.voxel_label_names,
         self.mesh_label_names) = self._get_seg_and_mesh_label_names(
             structure_type
         )

        super().__init__(
            image_file_name=self.image_file_name,
            mesh_file_names=list(self.mesh_label_names.values()),
            seg_file_name=self.seg_file_name,
            **kwargs
        )


class MindboggleDataset(CortexDataset):
    """ Mindboggle data """

    seg_file_name = "aparc+aseg_manual.nii.gz"  # Manual segmentations

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
