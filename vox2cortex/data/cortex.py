
""" Cortex dataset handler """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import os
import random
import logging
from copy import deepcopy
from collections.abc import Sequence
from typing import Union

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import trimesh
from trimesh import Trimesh
from trimesh.scene.scene import Scene
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

from utils.params import CHECK_DIR
from utils.visualization import show_difference
from utils.eval_metrics import Jaccard
from utils.modes import DataModes, ExecModes
from utils.logging import measure_time
from utils.mesh import Mesh, curv_from_cotcurv_laplacian
from utils.ico_template import (
    generate_sphere_template,
    generate_ellipsoid_template
)
from utils.utils import (
    choose_n_random_points,
    voxelize_mesh,
    voxelize_contour,
    create_mesh_from_voxels,
    create_mesh_from_pixels,
    mirror_mesh_at_plane,
    normalize_min_max,
)
from utils.coordinate_transform import (
    transform_mesh_affine,
    unnormalize_vertices_per_max_dim,
    normalize_vertices_per_max_dim,
)
from data.dataset import (
    DatasetHandler,
    flip_img,
    img_with_patch_size,
)
from data.supported_datasets import (
    valid_ids,
)
from data.cortex_labels import (
    combine_labels,
)

def _get_seg_and_mesh_label_names(structure_type, patch_mode, ndims):
    """ Helper function to map the structure type and the patch mode to the
    correct segmentation and mesh label names.

    For seg_label_names and voxelized_mesh_label_names entries can/should be
    grouped s.t. they represent one "voxel class" in the segmentation maps.
    """
    voxelized_mesh_label_names = None # Does not always exist
    if structure_type == "cerebral_cortex":
        if patch_mode=="single-patch":
            seg_label_names = (("right_cerebral_cortex",),)
            voxelized_mesh_label_names = (("rh_pial",),)
            mesh_label_names = ("rh_pial",)
        else: # not patch mode
            if ndims == 3: # 3D
                seg_label_names = (("left_cerebral_cortex",
                                   "right_cerebral_cortex"),)
                mesh_label_names = ("lh_pial", "rh_pial")
                voxelized_mesh_label_names = (("lh_pial", "rh_pial"),)
            else:
                raise NotImplementedError()

    elif structure_type == "white_matter":
        if patch_mode=="single-patch":
            seg_label_names = (("right_white_matter",),)
            mesh_label_names = ("rh_white",)
            voxelized_mesh_label_names = (("rh_white",),)
        else: # not patch mode
            if ndims == 3: # 3D
                seg_label_names = (("left_white_matter",
                                    "right_white_matter"),)
                mesh_label_names = ("lh_white", "rh_white")
                voxelized_mesh_label_names = (("lh_white", "rh_white"),)
            else:
                raise ValueError("Wrong dimensionality.")

    elif ("cerebral_cortex" in structure_type
          and "white_matter" in structure_type):
        if patch_mode == "single-patch":
            seg_label_names = (("right_white_matter",
                                "right_cerebral_cortex"),)
            mesh_label_names = ("rh_white", "rh_pial")
        else:
            # Not patch mode
            seg_label_names = (("left_white_matter",
                               "right_white_matter",
                               "left_cerebral_cortex",
                               "right_cerebral_cortex"),)
            mesh_label_names = ("lh_white",
                                "rh_white",
                                "lh_pial",
                                "rh_pial")
            voxelized_mesh_label_names = (("lh_white",
                                           "rh_white",
                                           "lh_pial",
                                           "rh_pial"),)
    else:
        raise ValueError("Unknown structure type.")

    return seg_label_names, mesh_label_names, voxelized_mesh_label_names

class Cortex(DatasetHandler):
    """ Cortex dataset

    It loads all data specified by 'ids' directly into memory.

    :param list ids: The ids of the files the dataset split should contain, example:
        ['1000_3', '1001_3',...]
    :param DataModes datamode: TRAIN, VALIDATION, or TEST
    :param str raw_data_dir: The raw base folder, contains folders
    corresponding to sample ids
    :param augment: Use image augmentation during training if 'True'
    :param patch_size: The patch size of the images, e.g. (256, 256, 256)
    :param mesh_target_type: 'mesh' or 'pointcloud'
    :param n_ref_points_per_structure: The number of ground truth points
    per 3D structure.
    :param structure_type: Either 'white_matter' or 'cerebral_cortex'
    :param patch_mode: "single-patch" or "no"
    :param patch_origin: The anker of an extracted patch, only has an effect if
    patch_mode is True.
    :param select_patch_size: The size of the cut out patches. Can be different
    to patch_size, e.g., if extracted patches should be resized after
    extraction.
    :param mc_step_size: The marching cubes step size.
    :param reduced_freesurfer: Use a freesurfer mesh with reduced number of
    vertices as mesh ground truth, e.g., 0.3. This has only an effect if patch_mode='no'.
    :param mesh_type: 'freesurfer' or 'marching cubes'
    :param provide_curvatures: Whether curvatures should be part of the items.
    :param preprocessed_data_dir: A directory that contains additional
    preprocessed data, e.g., thickness values.
    As a consequence, the ground truth can only consist of vertices and not of
    sampled surface points.
    :param seg_ground_truth: Either 'voxelized_meshes' or 'aseg'
    """

    img_filename = "mri.nii.gz"
    label_filename = "aseg.nii.gz" # For FS segmentations

    def __init__(self,
                 ids: Sequence,
                 mode: DataModes,
                 raw_data_dir: str,
                 augment: bool,
                 patch_size,
                 mesh_target_type: str,
                 n_ref_points_per_structure: int,
                 structure_type: Union[str, Sequence],
                 patch_mode: str="no",
                 patch_origin=(0,0,0),
                 select_patch_size=None,
                 reduced_freesurfer: int=None,
                 mesh_type='marching cubes',
                 provide_curvatures=False,
                 preprocessed_data_dir=None,
                 seg_ground_truth='voxelized_meshes',
                 **kwargs):
        super().__init__(ids, mode)

        assert patch_mode in ("single-patch", "no"),\
                "Unknown patch mode."
        assert mesh_type in ("marching cubes", "freesurfer"),\
                "Unknown mesh type"

        (self.seg_label_names,
         self.mesh_label_names,
         self.voxelized_mesh_label_names) = _get_seg_and_mesh_label_names(
            structure_type, patch_mode, len(patch_size)
        )
        self.structure_type = structure_type
        self._raw_data_dir = raw_data_dir
        self._preprocessed_data_dir = preprocessed_data_dir
        self._augment = augment
        self._mesh_target_type = mesh_target_type
        self._patch_origin = patch_origin
        self._mc_step_size = kwargs.get('mc_step_size', 1)
        self.trans_affine = []
        self.ndims = len(patch_size)
        self.patch_mode = patch_mode
        self.patch_size = tuple(patch_size)
        self.select_patch_size = select_patch_size if (
            select_patch_size is not None) else patch_size
        self.n_m_classes = len(self.mesh_label_names)
        assert self.n_m_classes == kwargs.get(
            "n_m_classes", len(self.mesh_label_names)
        ), "Number of mesh classes incorrect."
        self.provide_curvatures = provide_curvatures
        assert seg_ground_truth in ('voxelized_meshes', 'aseg')
        self.seg_ground_truth = seg_ground_truth
        self.n_ref_points_per_structure = n_ref_points_per_structure
        self.n_structures = len(self.mesh_label_names)
        self.mesh_type = mesh_type
        self.centers = None
        self.radii = None
        self.radii_x = None
        self.radii_y = None
        self.radii_z = None
        self.n_min_vertices, self.n_max_vertices = None, None
        # +1 for background
        self.n_v_classes = len(self.seg_label_names) + 1 if (
            self.seg_ground_truth == 'aseg'
        ) else len(self.voxelized_mesh_label_names) + 1

        # Sanity checks to make sure data is transformed correctly
        self.sanity_checks = kwargs.get("sanity_check_data", True)

        # Freesurfer meshes of desired resolution
        if reduced_freesurfer is not None:
            if reduced_freesurfer != 1.0:
                self.mesh_label_names = [
                    mn + "_reduced_" + str(reduced_freesurfer)
                    for mn in self.mesh_label_names
                ]

        if self.ndims == 3:
            self._prepare_data_3D()
        else:
            raise ValueError("Unknown number of dimensions ", self.ndims)

        # NORMALIZE images
        for i, img in enumerate(self.images):
            self.images[i] = normalize_min_max(img)

        # Do not store meshes in train split
        remove_meshes = self._mode == DataModes.TRAIN
        # Point, normal, and potentially curvature labels
        self.point_labels,\
                self.normal_labels,\
                self.curvatures = self._load_ref_points_all(remove_meshes)

        assert self.__len__() == len(self.images)
        assert self.__len__() == len(self.voxel_labels)
        assert self.__len__() == len(self.mesh_labels)
        assert self.__len__() == len(self.point_labels)
        assert self.__len__() == len(self.normal_labels)
        if self.ndims == 3:
            assert self.__len__() == len(self.trans_affine)
        if self.provide_curvatures:
            assert self.__len__() == len(self.curvatures)

        if self._augment:
            self.check_augmentation_normals()

    def _prepare_data_3D(self):
        """ Load 3D data """

        # Image data
        self.images, img_transforms = self._load_data3D_and_transform(
            self.img_filename, is_label=False
        )
        self.voxel_labels = None
        self.voxelized_meshes = None

        # Voxel labels
        if self.sanity_checks or self.seg_ground_truth == 'aseg':
            # Load 'aseg' segmentation maps from FreeSurfer
            self.voxel_labels, _ = self._load_data3D_and_transform(
                self.label_filename, is_label=True
            )
            # Combine labels as specified by groups (see
            # _get_seg_and_mesh_label_names)
            combine = lambda x: torch.sum(
                torch.stack([combine_labels(x, group, val) for val, group in
                enumerate(self.seg_label_names, 1)]),
                dim=0
            )
            self.voxel_labels = list(map(combine, self.voxel_labels))

        # Voxelized meshes
        if self.sanity_checks or self.seg_ground_truth == 'voxelized_meshes':
            try:
                self.voxelized_meshes = self._read_voxelized_meshes()
            except FileNotFoundError:
                self.voxelized_meshes = None # Compute later

        # Meshes
        self.mesh_labels = self._load_dataMesh_raw(meshnames=self.mesh_label_names)
        self._transform_meshes_as_images(img_transforms)

        # Voxelize meshes if voxelized meshes have not been created so far
        # and they are required (for sanity checks or as labels)
        if (self.voxelized_meshes is None and (
            self.sanity_checks or self.seg_ground_truth == 'voxelized_meshes')):
            self.voxelized_meshes = self._create_voxel_labels_from_meshes(
                self.mesh_labels
            )

        # Assert conformity of voxel labels and voxelized meshes
        if self.sanity_checks:
            for i, (vl, vm) in enumerate(zip(self.voxel_labels,
                                             self.voxelized_meshes)):
                iou = Jaccard(vl.cuda(), vm.cuda(), 2)
                if iou < 0.85:
                    out_fn = self._files[i].replace("/", "_")
                    show_difference(
                        vl,  vm,
                        f"../to_check/diff_mesh_voxel_label_{out_fn}.png"
                    )
                    print(f"[Warning] Small IoU ({iou}) of voxel label and"
                          " voxelized mesh label, check files at ../to_check/")

        # Use voxelized meshes as voxel ground truth
        if self.seg_ground_truth == 'voxelized_meshes':
            self.voxel_labels = self.voxelized_meshes

        (self.centers,
         self.radii,
         self.radii_x,
         self.radii_y,
         self.radii_z) = self._get_centers_and_radii(
            self.mesh_labels
         )

        self.thickness_per_vertex = self._get_per_vertex_label("thickness",
                                                               subfolder="")


    def _transform_meshes_as_images(self, img_transforms):
        """ Transform meshes according to image transformations
        (crops, resize) and normalize
        """
        for i, (m, t) in tqdm(
            enumerate(zip(self.mesh_labels, img_transforms)),
            position=0, leave=True, desc="Transform meshes accordingly..."
        ):
            new_vertices, new_faces = transform_mesh_affine(
                m.vertices, m.faces, t
            )
            new_vertices, norm_affine = normalize_vertices_per_max_dim(
                new_vertices.view(-1, self.ndims),
                self.patch_size,
                return_affine=True
            )
            new_vertices = new_vertices.view(self.n_m_classes, -1, self.ndims)
            new_normals = Meshes(
                new_vertices, new_faces
            ).verts_normals_padded()

            # Replace mesh with transformed one
            self.mesh_labels[i] = Mesh(
                new_vertices, new_faces, new_normals, m.features
            )

            # Store affine transformations
            self.trans_affine[i] = norm_affine @ t @ self.trans_affine[i]

    def mean_area(self):
        """ Average surface area of meshes. """
        areas = []
        ndims = len(self.patch_size)
        for m in self.mesh_labels:
            m_unnorm = Mesh(unnormalize_vertices_per_max_dim(
                m.vertices.view(-1, ndims), self.patch_size),
                m.faces.view(-1, ndims)
            )
            areas.append(m_unnorm.to_trimesh().area)

        return np.mean(areas)

    def mean_edge_length(self):
        """ Average edge length in dataset.

        Code partly from pytorch3d.loss.mesh_edge_loss.
        """
        edge_lengths = []
        for m in self.mesh_labels:
            m_ = m.to_pytorch3d_Meshes()
            if self.ndims == 3:
                edges_packed = m_.edges_packed()
            else:
                raise ValueError("Only 3D possible.")
            verts_packed = m_.verts_packed()

            verts_edges = verts_packed[edges_packed]
            v0, v1 = verts_edges.unbind(1)
            edge_lengths.append(
                (v0 - v1).norm(dim=1, p=2).mean().item()
            )

        return torch.tensor(edge_lengths).mean()

    def store_sphere_template(self, path, level):
        """ Template for dataset. This can be stored and later used during
        training.
        """
        if self.centers is not None and self.radii is not None:
            template = generate_sphere_template(self.centers,
                                                self.radii,
                                                level=level)
            template.export(path)
        else:
            raise RuntimeError("Centers and/or radii are unknown, template"
                               " cannnot be created. ")
        return path

    def store_ellipsoid_template(self, path, level):
        """ Template for dataset. This can be stored and later used during
        training.
        """
        if (self.centers is not None and
            self.radii_x is not None and
            self.radii_y is not None and
            self.radii_z is not None):
            template = generate_ellipsoid_template(
                self.centers,
                self.radii_x, self.radii_y, self.radii_z,
                level=level
            )
            template.export(path)
        else:
            raise RuntimeError("Centers and/or radii are unknown, template"
                               " cannnot be created. ")
        return path

    def store_index0_template(self, path, n_max_points=41000):
        """ This template is the structure of dataset element at index 0,
        potentially mirrored at the hemisphere plane. """
        template = Scene()
        if len(self.mesh_label_names) == 2:
            label_1, label_2 = self.mesh_label_names
        else:
            label_1 = self.mesh_labels[0]
            label_2 = None
        # Select mesh to generate the template from
        vertices = self.mesh_labels[0].vertices[0]
        faces = self.mesh_labels[0].faces[0]

        # Remove padded vertices and faces
        vertices_ = vertices[~torch.isclose(vertices, torch.Tensor([-1.0])).all(dim=1)]
        faces_ = faces[~(faces == -1).any(dim=1)]

        structure_1 = Trimesh(vertices_, faces_, process=False)

        # Increase granularity until desired number of points is reached
        while structure_1.subdivide().vertices.shape[0] < n_max_points:
            structure_1 = structure_1.subdivide()

        assert structure_1.is_watertight, "Mesh template should be watertight."
        print(f"Template structure has {structure_1.vertices.shape[0]}"
              " vertices.")
        template.add_geometry(structure_1, geom_name=label_1)

        # Second structure = mirror of first structure
        if label_2 is not None:
            plane_normal = np.array(self.centers[label_2] - self.centers[label_1])
            plane_point = 0.5 * np.array((self.centers[label_1] +
                                          self.centers[label_2]))
            structure_2 = mirror_mesh_at_plane(structure_1, plane_normal,
                                              plane_point)
            template.add_geometry(structure_2, geom_name=label_2)

        template.export(path)

        return path

    @staticmethod
    def split(raw_data_dir,
              augment_train,
              save_dir,
              dataset_seed=0,
              dataset_split_proportions=None,
              fixed_split: Union[dict, bool, Sequence]=False,
              overfit=False,
              load_only=('train', 'validation', 'test'),
              **kwargs):
        """ Create train, validation, and test split of the cortex data"

        :param str raw_data_dir: The raw base folder, contains a folder for each
        sample
        :param dataset_seed: A seed for the random splitting of the dataset.
        :param dataset_split_proportions: The proportions of the dataset
        splits, e.g. (80, 10, 10)
        :param augment_train: Augment training data.
        :param save_dir: A directory where the split ids can be saved.
        :param fixed_split: A dict containing file ids for 'train',
        'validation', and 'test'. If specified, values of dataset_seed,
        overfit, and dataset_split_proportions will be ignored. If only 'True',
        the fixed split IDs are read from a file.
        :param overfit: Create small datasets for overfitting if this parameter
        is > 0.
        :param load_only: Only return the splits specified (in the order train,
        validation, test, while missing splits will be None)
        :param kwargs: Dataset parameters.
        :return: (Train dataset, Validation dataset, Test dataset)
        """

        # Decide between fixed and random split
        if fixed_split:
            if isinstance(fixed_split, dict):
                files_train = fixed_split['train']
                files_val = fixed_split['validation']
                files_test = fixed_split['test']
            elif isinstance(fixed_split, Sequence):
                assert len(fixed_split) == 3,\
                        "Should contain one file per split"
                convert = lambda x: x[:-1] # 'x\n' --> 'x'
                train_split = os.path.join(raw_data_dir, fixed_split[0])
                try:
                    files_train = list(map(convert, open(train_split, 'r').readlines()))
                except:
                    files_train = []
                    print("[Warning] No training files.")
                val_split = os.path.join(raw_data_dir, fixed_split[1])
                try:
                    files_val = list(map(convert, open(val_split, 'r').readlines()))
                except:
                    files_val = []
                    print("[Warning] No validation files.")
                test_split = os.path.join(raw_data_dir, fixed_split[2])
                try:
                    files_test = list(map(convert, open(test_split, 'r').readlines()))
                except:
                    files_test = []
                    print("[Warning] No test files.")

            else:
                raise TypeError("Wrong type of parameter 'fixed_split'."
                                f" Got {type(fixed_split)} but should be"
                                "'Sequence' or 'dict'")
        else: # Random split
            # Available files
            all_files = valid_ids(raw_data_dir) # Remove invalid

            # Shuffle with seed
            random.Random(dataset_seed).shuffle(all_files)

            # Split according to proportions
            assert np.sum(dataset_split_proportions) == 100, "Splits need to sum to 100."
            indices_train = slice(0, dataset_split_proportions[0] * len(all_files) // 100)
            indices_val = slice(indices_train.stop,
                                indices_train.stop +\
                                    (dataset_split_proportions[1] * len(all_files) // 100))
            indices_test = slice(indices_val.stop, len(all_files))

            files_train = all_files[indices_train]
            files_val = all_files[indices_val]
            files_test = all_files[indices_test]

        if overfit:
            # Consider the same splits for train validation and test
            files_train = files_train[:overfit]
            files_val = files_train[:overfit]
            files_test = files_train[:overfit]

        # Save ids to file
        DatasetHandler.save_ids(files_train, files_val, files_test, save_dir)

        assert (len(set(files_train) & set(files_val) & set(files_test)) == 0
                or overfit),\
                "Train, validation, and test set should not intersect!"

        # Create train, validation, and test datasets
        if 'train' in load_only:
            train_dataset = Cortex(files_train,
                                   DataModes.TRAIN,
                                   raw_data_dir,
                                   augment=augment_train,
                                   **kwargs)
        else:
            train_dataset = None
        if 'validation' in load_only:
            val_dataset = Cortex(files_val,
                                 DataModes.VALIDATION,
                                 raw_data_dir,
                                 augment=False,
                                 **kwargs)
        else:
            val_dataset = None
        if 'test' in load_only:
            test_dataset = Cortex(files_test,
                                  DataModes.TEST,
                                  raw_data_dir,
                                  augment=False,
                                  **kwargs)
        else:
            test_dataset = None

        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        return len(self._files)

    def resample_surface_points(self):
        """ Resample the surface points of the meshes. """
        self.point_labels,\
                self.normal_labels,\
                self.curvatures = self._load_ref_points()

    @measure_time
    def get_item_from_index(self, index: int, mesh_target_type: str=None,
                            *args, **kwargs):
        """
        One data item has the form
        (image, voxel label, points, faces, normals)
        with types all of type torch.Tensor
        """
        # Use mesh target type of object if not specified
        if mesh_target_type is None:
            mesh_target_type = self._mesh_target_type

        # Raw data
        img = self.images[index]
        voxel_label = self.voxel_labels[index]
        (target_points,
         target_faces,
         target_normals,
         target_curvs) = self._get_mesh_target(index, mesh_target_type)

        # Potentially augment
        if self._augment:
            assert all(
                (np.array(img.shape) - np.array(self.patch_size)) % 2 == 0
            ), "Padding must be symmetric for augmentation."

            # Mesh coordinates --> image coordinates
            target_points = unnormalize_vertices_per_max_dim(
                target_points.view(-1, 3), self.patch_size
            ).view(self.n_m_classes, -1, 3)
            # Augment
            (img,
             voxel_label,
             target_points,
             target_normals) = self.augment_data(
                 img.numpy(),
                 voxel_label.numpy(),
                 target_points,
                 target_normals
             )
            # Image coordinates --> mesh coordinates
            target_points = normalize_vertices_per_max_dim(
                target_points.view(-1, 3), self.patch_size
            ).view(self.n_m_classes, -1, 3)

            img = torch.from_numpy(img)
            voxel_label = torch.from_numpy(voxel_label)


        # Channel dimension
        img = img[None]

        logging.getLogger(ExecModes.TRAIN.name).debug("Dataset file %s",
                                                      self._files[index])

        return (img,
                voxel_label,
                target_points,
                target_faces,
                target_normals,
                target_curvs)

    def _get_mesh_target(self, index, target_type):
        """ Ground truth points and optionally normals """
        if target_type == 'pointcloud':
            points = self.point_labels[index]
            normals = np.array([]) # Empty, not used
            faces = np.array([]) # Empty, not used
            curvs = np.array([]) # Empty, not used
        elif target_type == 'mesh':
            points, normals, curvs = self._get_ref_points_from_index(index)
            faces = np.array([]) # Empty, not used
        elif target_type == 'full_mesh':
            points = self.mesh_labels[index].vertices
            normals = self.mesh_labels[index].normals
            faces = self.mesh_labels[index].faces
            mesh = self.mesh_labels[index].to_pytorch3d_Meshes()
            curvs = curv_from_cotcurv_laplacian(
                mesh.verts_packed(),
                mesh.faces_packed()
            ).view(self.n_m_classes, -1, 1)
        else:
            raise ValueError("Invalid mesh target type.")

        return points, faces, normals, curvs

    def get_item_and_mesh_from_index(self, index):
        """ Get image, segmentation ground truth and reference mesh"""
        (img,
         voxel_label,
         vertices,
         faces,
         normals, _) = self.get_item_from_index(
            index, mesh_target_type='full_mesh'
        )
        thickness = self.get_thickness_from_index(index)
        mesh_label = Mesh(vertices, faces, normals, thickness)
        trans_affine_label = self.trans_affine[index]

        return img, voxel_label, mesh_label, trans_affine_label

    def get_thickness_from_index(self, index: int):
        """ Return per-vertex thickness of the ith dataset element if possible."""
        return self.thickness_per_vertex[index] if (
            self.thickness_per_vertex is not None) else None

    def _read_voxelized_meshes(self):
        """ Read voxelized meshes stored in nifity files and set voxel classes
        as specified by _get_seg_and_mesh_label_names. """
        data = []
        # Iterate over sample ids
        for i, sample_id in tqdm(enumerate(self._files), position=0, leave=True,
                       desc="Loading voxelized meshes..."):
            voxelized_mesh_label = None
            # Iterate over voxel classes
            for group_id, voxel_group in enumerate(
                self.voxelized_mesh_label_names, 1
            ):
                # Iterate over files
                for vmln in voxel_group:
                    vm_file = os.path.join(
                        self._raw_data_dir, sample_id, vmln + ".nii.gz"
                    )
                    img = nib.load(vm_file).get_fdata().astype(int) * group_id
                    if voxelized_mesh_label is None:
                        voxelized_mesh_label = img
                    else:
                        np.putmask(voxelized_mesh_label, img>0, img)

            # Correct patch size
            voxelized_mesh_label, _ = self._get_single_patch(
                voxelized_mesh_label, is_label=True
            )
            data.append(voxelized_mesh_label)

        return data

    def _load_data3D_raw(self, filename: str):
        data = []
        for fn in tqdm(self._files, position=0, leave=True,
                       desc="Loading images..."):
            img = nib.load(os.path.join(self._raw_data_dir, fn, filename))

            d = img.get_fdata()
            data.append(d)

        return data

    def _load_data3D_and_transform(self, filename: str, is_label: bool):
        """ Load data and transform to correct patch size. """
        data = []
        transformations = []
        for fn in tqdm(self._files, position=0, leave=True,
                       desc="Loading images..."):
            img = nib.load(os.path.join(self._raw_data_dir, fn, filename))

            img_data = img.get_fdata()
            img_data, trans_affine = self._get_single_patch(img_data, is_label)
            data.append(img_data)
            transformations.append(trans_affine)

        return data, transformations

    def _get_single_patch(self, img, is_label):
        """ Extract a single patch from an image. """

        assert (tuple(self._patch_origin) == (0,0,0)
                or self.patch_mode != "no"),\
                "If patch mode is 'no', patch origin should be (0,0,0)"

        # Limits for patch selection
        lower_limit = np.array(self._patch_origin, dtype=int)
        upper_limit = np.array(self._patch_origin, dtype=int) +\
                np.array(self.select_patch_size, dtype=int)

        assert img.shape == (182, 218, 182),\
                "Our mesh templates were created based on this shape."
        # Select patch from whole image
        img_patch, trans_affine_1 = img_with_patch_size(
            img, self.select_patch_size, is_label=is_label, mode='crop',
            crop_at=(lower_limit + upper_limit) // 2
        )
        # Zoom to certain size
        if self.patch_size != self.select_patch_size:
            img_patch, trans_affine_2 = img_with_patch_size(
                img_patch, self.patch_size, is_label=is_label, mode='interpolate'
            )
        else:
            trans_affine_2 = np.eye(self.ndims + 1) # Identity

        trans_affine = trans_affine_2 @ trans_affine_1

        return img_patch, trans_affine

    def _create_voxel_labels_from_meshes(self, mesh_labels):
        """ Return the voxelized meshes as 3D voxel labels. Here, individual
        structures are not distinguished. """
        data = []
        for m in tqdm(mesh_labels, position=0, leave=True,
                      desc="Voxelize meshes..."):
            vertices = m.vertices.view(self.n_m_classes, -1, 3)
            faces = m.faces.view(self.n_m_classes, -1, 3)
            voxel_label = voxelize_mesh(
                vertices, faces, self.patch_size, self.n_m_classes
            ).sum(0).bool().long() # Treat as one class

            data.append(voxel_label)

        return data

    def _get_centers_and_radii(self, meshes: list):
        """ Return average centers and radii of all meshes provided. """

        centers_per_structure = {mn: [] for mn in self.mesh_label_names}
        radii_per_structure = {mn: [] for mn in self.mesh_label_names}
        radii_x_per_structure = {mn: [] for mn in self.mesh_label_names}
        radii_y_per_structure = {mn: [] for mn in self.mesh_label_names}
        radii_z_per_structure = {mn: [] for mn in self.mesh_label_names}

        for m in meshes:
            for verts_, mn in zip(m.vertices, self.mesh_label_names):
                # Remove padded vertices
                verts = verts_[~torch.isclose(verts_, torch.Tensor([-1.0])).all(dim=1)]
                # Centroid of vertex coordinates
                center = verts.mean(dim=0)
                centers_per_structure[mn].append(center)
                # Average radius --> "sphere"
                radius = torch.sqrt(torch.sum((verts - center)**2, dim=1)).mean(dim=0)
                radii_per_structure[mn].append(radius)
                # Max. radius across dimensions --> "ellipsoid"
                (radius_x,
                 radius_y,
                 radius_z) = torch.max(torch.abs(verts - center), dim=0)[0].unbind()
                radii_x_per_structure[mn].append(radius_x)
                radii_y_per_structure[mn].append(radius_y)
                radii_z_per_structure[mn].append(radius_z)

        # Average radius per structure
        if self.__len__() > 0:
            centroids = {k: torch.mean(torch.stack(v), dim=0)
                         for k, v in centers_per_structure.items()}
            radii = {k: torch.mean(torch.stack(v), dim=0)
                     for k, v in radii_per_structure.items()}
            radii_x = {k: torch.mean(torch.stack(v), dim=0)
                     for k, v in radii_x_per_structure.items()}
            radii_y = {k: torch.mean(torch.stack(v), dim=0)
                     for k, v in radii_y_per_structure.items()}
            radii_z = {k: torch.mean(torch.stack(v), dim=0)
                     for k, v in radii_z_per_structure.items()}
        else:
            centroids = None
            radii = None
            radii_x = None
            radii_y = None
            radii_z = None

        return centroids, radii, radii_x, radii_y, radii_z

    def _load_dataMesh_raw(self, meshnames):
        """ Load mesh such that it's registered to the respective 3D image. If
        a mesh cannot be found, a dummy is inserted if it is a test split.
        """
        data = []
        assert len(self.trans_affine) == 0, "Should be empty."
        for fn in tqdm(self._files, position=0, leave=True,
                       desc="Loading meshes..."):
            # Voxel coords
            orig = nib.load(os.path.join(self._raw_data_dir, fn,
                                         self.img_filename))
            vox2world_affine = orig.affine
            world2vox_affine = np.linalg.inv(vox2world_affine)
            self.trans_affine.append(world2vox_affine)
            file_vertices = []
            file_faces = []
            for mn in meshnames:
                try:
                    mesh = trimesh.load_mesh(os.path.join(
                        self._raw_data_dir, fn, mn + ".stl"
                    ))
                except ValueError:
                    try:
                        mesh = trimesh.load_mesh(os.path.join(
                            self._raw_data_dir, fn, mn + ".ply"
                        ))
                    except Exception as e:
                        # Insert a dummy if dataset is test split
                        if self._mode != DataModes.TEST:
                            raise e
                        print(f"[Warning] No mesh for file {fn}/{mn},"
                              " inserting dummy.")
                        mesh = trimesh.creation.icosahedron()
                # World --> voxel coordinates
                voxel_verts, voxel_faces = transform_mesh_affine(
                    mesh.vertices, mesh.faces, world2vox_affine
                )
                # Store min/max number of vertices
                self.n_max_vertices = np.maximum(
                    voxel_verts.shape[0], self.n_max_vertices) if (
                self.n_max_vertices is not None) else voxel_verts.shape[0]
                self.n_min_vertices = np.minimum(
                    voxel_verts.shape[0], self.n_min_vertices) if (
                self.n_min_vertices is not None) else voxel_verts.shape[0]
                # Add to structures of file
                file_vertices.append(torch.from_numpy(voxel_verts))
                file_faces.append(torch.from_numpy(voxel_faces))

            # First treat as a batch of multiple meshes and then combine
            # into one mesh
            mesh_batch = Meshes(file_vertices, file_faces)
            mesh_single = Mesh(
                mesh_batch.verts_padded().float(),
                mesh_batch.faces_padded().long(),
            )
            data.append(mesh_single)

        return data

    def _load_mc_dataMesh(self, voxel_labels):
        """ Create ground truth meshes from voxel labels."""
        data = []
        for vl in voxel_labels:
            assert tuple(vl.shape) == tuple(self.patch_size),\
                    "Voxel label should be of correct size."
            mc_mesh = create_mesh_from_voxels(
                vl, mc_step_size=self._mc_step_size,
            ).to_pytorch3d_Meshes()
            data.append(Mesh(
                mc_mesh.verts_padded(),
                mc_mesh.faces_padded(),
                mc_mesh.verts_normals_padded()
            ))

        return data

    def _get_ref_points_from_index(self, index):
        """ Choose n reference points together with their normals and
        curvature. """
        # Points/vertices
        p, idx = choose_n_random_points(
            self.point_labels[index],
            self.n_ref_points_per_structure,
            return_idx=True,
            # Ignoring padded vertices is only possible if the
            # number of required vertices is smaller than the
            # minimum number of vertices in the dataset
            ignore_padded=(self.n_ref_points_per_structure <
                           self.n_min_vertices)
        )
        # Normals
        n = self.normal_labels[index][idx.unbind(1)].view(
            self.n_m_classes, -1, 3
        )
        # Curvature
        if self.provide_curvatures:
            c = self.curvatures[index][idx.unbind(1)].view(
                self.n_m_classes, -1, 1
            )
        else:
            c = np.array([])

        return p, n, c


    def _load_ref_points_all(self, remove_meshes=False):
        """ Sample surface points from meshes and optionally remove them (to
        save memory)"""
        points, normals = [], []
        curvs = [] if self.provide_curvatures else None
        for i, m in tqdm(enumerate(self.mesh_labels), leave=True, position=0,
                      desc="Get point labels from meshes..."):
            if self.provide_curvatures:
                # Choose a certain number of vertices
                m_ = m.to_pytorch3d_Meshes()
                p = m_.verts_padded()
                # Choose normals with the same indices as vertices
                n = m_.verts_normals_padded()
                # Choose curvatures with the same indices as vertices
                c = curv_from_cotcurv_laplacian(
                    m_.verts_packed(), m_.faces_packed()
                ).view(self.n_m_classes, -1, 1)
                # Sanity check
                assert torch.allclose(
                    m_.verts_normals_padded(), m.normals, atol=1e-05
                ), "Inconsistent normals!"

                # Remove to save memory
                if remove_meshes:
                    self.mesh_labels[i] = None

            else: # No curvatures
                # Sample from mesh surface, probably less memory efficient
                # than with curvatures
                p, n = sample_points_from_meshes(
                    m.to_pytorch3d_Meshes(),
                    self.n_ref_points_per_structure,
                    return_normals=True
                )

            points.append(p)
            normals.append(n)
            if self.provide_curvatures:
                curvs.append(c)

        return points, normals, curvs

    def augment_data(self, img, label, coordinates, normals):
        assert self._augment, "No augmentation in this dataset."
        return flip_img(img, label, coordinates, normals)

    def check_augmentation_normals(self):
        """ Assert correctness of the transformation of normals during
        augmentation.
        """
        py3d_mesh = self.mesh_labels[0].to_pytorch3d_Meshes()
        img_f, label_f, coo_f, normals_f = self.augment_data(
            self.images[0].numpy(), self.voxel_labels[0].numpy(),
            py3d_mesh.verts_padded(), py3d_mesh.verts_normals_padded()
        )
        py3d_mesh_aug = Meshes(coo_f, py3d_mesh.faces_padded())
        # Assert up to sign of direction
        assert (
            torch.allclose(normals_f, py3d_mesh_aug.verts_normals_padded(),
                           atol=7e-03)
            or torch.allclose(-normals_f, py3d_mesh_aug.verts_normals_padded(),
                             atol=7e-03)
        )

    def _get_per_vertex_label(self, morphology, subfolder="surf"):
        """ Load per-vertex labels, e.g., thickness values, from a freesurfer
        morphology (curv) file in the preprocessed data (FS) directory for each
        dataset sample.
        :param morphology: The morphology to load, e.g., 'thickness'
        :param subfolder: The subfolder of a the sample folder where the
        morphology file could be found.
        :return: List of len(self) containing per-vertex morphology values for
        each sample.
        """
        if self._preprocessed_data_dir is None:
            return None

        morph_labels = []
        for fn in self._files:
            file_dir = os.path.join(
                self._preprocessed_data_dir, fn, subfolder
            )
            file_labels = []
            n_max = 0
            for mn in self.mesh_label_names:
                # Filenames have form 'lh_white_reduced_0.x.thickness'
                morph_fn = mn + "." + morphology
                morph_fn = os.path.join(file_dir, morph_fn)
                try:
                    morph_label = nib.freesurfer.io.read_morph_data(morph_fn)
                except Exception as e:
                    # Insert dummy if file could not
                    # be found
                    print(f"[Warning] File {morph_fn} could not be found,"
                          " inserting dummy.")
                    morph_label = np.zeros(self.mesh_labels[
                        self._files.index(fn)
                    ].vertices[self.mesh_label_names.index(mn)].shape[0])

                file_labels.append(
                    torch.from_numpy(morph_label.astype(np.float32))
                )
                if morph_label.shape[0] > n_max:
                    n_max = morph_label.shape[0]

            # Pad values with 0
            file_labels_padded = []
            for fl in file_labels:
                file_labels_padded.append(
                    F.pad(fl, (0, n_max - fl.shape[0]))
                )

            morph_labels.append(torch.stack(file_labels_padded))

        return morph_labels
