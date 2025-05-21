""" Cortex dataset handler """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import random
import warnings
import collections.abc as abc
from typing import Union, Sequence
from abc import ABC, abstractmethod

import torch
import torchio as tio
import numpy as np
import nibabel as nib
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import transform_mesh_affine
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

import logger
from utils.visualization import show_difference
from utils.eval_metrics import Jaccard
from utils.modes import DataModes, ExecModes
from utils.mesh import Mesh, curv_from_cotcurv_laplacian
from utils.template import TEMPLATE_SPECS, TEMPLATE_PATH

from utils.utils import voxelize_mesh, normalize_min_max
from utils.coordinate_transform import (
    normalize_vertices,
)
from data.dataset import (
    DatasetHandler,
    flip_img,
    img_with_patch_size,
)


log = logger.get_std_logger(__name__)


def combine_labels(label, label_groups, LabelMap):
    """Combine/remap labels as specified by groups
    (see _get_seg_and_mesh_file_names). This is a bit messy but gives a lot of
    flexibility.
    """
    # Find relevant labels and set the rest to 0 (background)
    relevant_labels = torch.tensor(
        [LabelMap[l_name] for group in label_groups.values() for l_name in group]
    ).long()
    new_label = torch.zeros_like(label)

    # Remap label ids based on groups (ignore background here)
    for i, group in enumerate(list(label_groups.values()), start=1):
        for l_name in group:
            new_label[label == LabelMap[l_name]] = i

    return new_label


class ImageAndMeshDataset(DatasetHandler, ABC):
    """Base class for dataset handlers consisting of images, meshes, and
    segmentations.

    It loads all data specified by 'ids' directly into memory. The
    corresponding raw directory should contain a folder for each ID with
    image and mesh data.


    For initialization, it is generally recommended to use the split()
    function, which directly gives a train, validation, and test dataset.
    """

    # Generic names, are usually overridden by subclasses
    image_file_name = "mri.nii.gz"
    seg_file_name = "aseg.nii.gz"
    voxel_label_names = {"foreground": ("foreground",)}
    mesh_label_names = {"foreground": "foreground"}

    # Default value used for padding images
    PAD_VALUE = 0

    @classmethod
    @abstractmethod
    def _get_seg_and_mesh_label_names(cls, structure_type):
        """Helper function to map the structure type, i.e., a generic name
        like "white_matter", to the correct segmentation and mesh label names.

        This function should be overridden by each subclass, see for example
        the implementation in data.cortex.CortexDataset.
        """
        pass

    def __init__(
        self,
        ids: Sequence,
        mode: DataModes,
        raw_data_dir: str,
        patch_size,
        n_ref_points_per_structure: int,
        image_file_name: str,
        mesh_file_names: str,
        seg_file_name: Sequence[str] = None,
        voxelized_mesh_file_names: Sequence[str] = None,
        augment: bool = False,
        patch_origin: Sequence[int] = (0, 0, 0),
        select_patch_size: Sequence[int] = None,
        seg_ground_truth: str = "voxelized_meshes",
        check_dir: str = "../to_check",
        sanity_check_data: bool = True,
        **kwargs
    ):
        super().__init__(ids, mode)

        if seg_ground_truth not in ("voxelized_meshes", "voxel_seg"):
            raise ValueError(f"Unknown seg_ground_truth {seg_ground_truth}")

        self._orig_img_size = None
        self._check_dir = check_dir
        self._raw_data_dir = raw_data_dir
        self._augment = augment
        self._patch_origin = patch_origin
        self.trans_affine = []
        self.mesh_targets = None
        self.ndims = len(patch_size)
        self.patch_size = tuple(patch_size)
        # If not specified, select_patch_size is equal to patch_size
        self.select_patch_size = (
            select_patch_size if (select_patch_size is not None) else patch_size
        )
        self.n_m_classes = len(mesh_file_names)
        self.seg_ground_truth = seg_ground_truth
        self.n_ref_points_per_structure = n_ref_points_per_structure
        self.n_min_vertices, self.n_max_vertices = None, None
        self.n_v_classes = len(self.voxel_label_names) + 1  # +1 for background

        # Sanity checks to make sure data is transformed correctly
        self.sanity_checks = sanity_check_data
        # Load/prepare data
        self._prepare_data_3D(
            image_file_name,
            seg_file_name,
            mesh_file_names,
            voxelized_mesh_file_names,
        )

        # NORMALIZE images
        if not hasattr(self, "img_norm"):
            self.img_norm = normalize_min_max
        log.info("Normalizing images with " + str(self.img_norm))
        for i, img in enumerate(self.images):
            self.images[i] = self.img_norm(img)

        assert self.__len__() == len(self.images)
        assert self.__len__() == len(self.voxel_labels)
        assert self.__len__() == len(self.mesh_labels)
        assert self.__len__() == len(self.trans_affine)

        # if self._augment:
        #    self.check_augmentation_normals()

    def image_affine(self, index: int):
        return nib.load(
            os.path.join(self._raw_data_dir, self.ids[index], self.image_file_name)
        ).affine

    def _prepare_data_3D(
        self,
        image_file_name,
        seg_file_name,
        mesh_file_names,
        voxelized_mesh_file_names,
    ):
        """Load 3D data"""

        # Image data
        self.images, img_transforms = self._load_data3D_and_transform(
            image_file_name, is_label=False
        )
        self.voxel_labels = None
        self.voxelized_meshes = None

        # Voxel labels
        if self.sanity_checks or self.seg_ground_truth == "voxel_seg":
            self.voxel_labels, _ = self._load_data3D_and_transform(
                seg_file_name, is_label=True
            )
            self.voxel_labels = [
                combine_labels(vl, self.voxel_label_names, self.LabelMap)
                for vl in self.voxel_labels
            ]

        # Meshes
        self.mesh_labels = self._load_dataMesh_raw(meshnames=mesh_file_names)
        self._transform_meshes_as_images(img_transforms)

        # Voxelize meshes if voxelized meshes have not been created so far
        # and they are required (for sanity checks or as labels)
        if self.voxelized_meshes is None and (
            self.sanity_checks or self.seg_ground_truth == "voxelized_meshes"
        ):
            self.voxelized_meshes = self._create_voxel_labels_from_meshes(
                self.mesh_labels
            )

        # Assert conformity of voxel labels and voxelized meshes
        if self.sanity_checks:
            for i, (vl, vm) in enumerate(zip(self.voxel_labels, self.voxelized_meshes)):
                iou = Jaccard(
                    vl.bool().long().cuda(),  # Combine labels
                    vm.bool().long().cuda(),  # Combine labels
                    2,
                )
                out_fn = self.ids[i].replace("/", "_")
                show_difference(
                    vl.bool().long(),
                    vm.bool().long(),
                    os.path.join(
                        self._check_dir, f"diff_mesh_voxel_label_{out_fn}.png"
                    ),
                )
                if iou < 0.9:
                    log.warning(
                        f"Small IoU ({iou}) of voxel label and voxelized mesh"
                        f" label {self.ids[i]}, check files at {self._check_dir}"
                    )
                    img = nib.Nifti1Image(vl.squeeze().cpu().numpy(), np.eye(4))
                    nib.save(
                        img, os.path.join(self._check_dir, "data_voxel_label.nii.gz")
                    )
                    img = nib.Nifti1Image(vm.squeeze().cpu().numpy(), np.eye(4))
                    nib.save(
                        img, os.path.join(self._check_dir, "data_mesh_label.nii.gz")
                    )
                    img = nib.Nifti1Image(
                        self.images[i].squeeze().cpu().numpy(), np.eye(4)
                    )
                    nib.save(img, os.path.join(self._check_dir, "data_img.nii.gz"))

        # Use voxelized meshes as voxel ground truth
        if self.seg_ground_truth == "voxelized_meshes":
            self.voxel_labels = self.voxelized_meshes

    def _transform_meshes_as_images(self, img_transforms):
        """Transform meshes according to image transformations
        (crops, resize) and normalize
        """
        for i, (m, t) in tqdm(
            enumerate(zip(self.mesh_labels, img_transforms)),
            position=0,
            leave=True,
            desc="Transform meshes accordingly...",
        ):
            # Transform vertices and potentially faces (to preserve normal
            # convention)
            new_vertices, new_faces = [], []
            for v, f in zip(m.verts_list(), m.faces_list()):
                new_v, new_f = transform_mesh_affine(
                    v, f, torch.tensor(t, dtype=v.dtype)
                )
                _, _, norm_affine = normalize_vertices(
                    new_v, self.patch_size, new_f, return_affine=True
                )
                new_v, new_f = transform_mesh_affine(
                    new_v, new_f, torch.tensor(norm_affine, dtype=v.dtype)
                )
                new_vertices.append(new_v)
                new_faces.append(new_f)

            # Replace mesh with transformed one
            self.mesh_labels[i] = Meshes(new_vertices, new_faces)
            # Store affine transformations
            # TODO: maybe do this outside of this function
            self.trans_affine[i] = norm_affine @ t @ self.trans_affine[i]

    def mean_edge_length(self):
        """Average edge length in dataset.

        Code partly from pytorch3d.loss.mesh_edge_loss.
        """
        edge_lengths = []
        for m in self.mesh_labels:
            if self.ndims == 3:
                edges_packed = m.edges_packed()
            else:
                raise ValueError("Only 3D possible.")
            verts_packed = m.verts_packed()

            verts_edges = verts_packed[edges_packed]
            v0, v1 = verts_edges.unbind(1)
            edge_lengths.append((v0 - v1).norm(dim=1, p=2).mean().item())

        return torch.tensor(edge_lengths).mean()

    @classmethod
    def split(
        cls,
        raw_data_dir,
        save_dir,
        augment_train: bool = False,
        dataset_seed: int = 0,
        all_ids_file: str = None,
        dataset_split_proportions: Sequence[int] = None,
        fixed_split: Union[dict, Sequence[str]] = None,
        overfit: int = None,
        load_only: Union[str, Sequence[str]] = ("train", "validation", "test"),
        **kwargs,
    ):
        """Create train, validation, and test split of data"

        :param str raw_data_dir: The raw base folder; should contain a folder for each
        ID
        :param save_dir: A directory where the split ids can be saved.
        :param augment_train: Augment training data.
        :param dataset_seed: A seed for the random splitting of the dataset.
        :param all_ids_file: A file that contains all IDs that should be taken
        into consideration.
        :param dataset_split_proportions: The proportions of the dataset
        splits, e.g. (80, 10, 10)
        :param fixed_split: A dict containing file ids for 'train',
        'validation', and 'test'. If specified, values of dataset_seed,
        overfit, and dataset_split_proportions will be ignored. Alternatively,
        a sequence of files containing ids can be given.
        :param overfit: Create small datasets for overfitting if this parameter
        is > 0.
        :param load_only: Only return the splits specified (in the order train,
        validation, test, while missing splits will be None). This is helpful
        to save RAM.
        :param kwargs: Parameters of ImageAndMeshDataset + subclass-specific
        parameters.
        :return: (Train dataset, Validation dataset, Test dataset)
        """

        # Decide between fixed and random split
        if fixed_split is not None:
            if isinstance(fixed_split, dict):
                files_train = fixed_split["train"]
                files_val = fixed_split["validation"]
                files_test = fixed_split["test"]
            elif isinstance(fixed_split, abc.Sequence):
                assert len(fixed_split) == 3, "Should contain one file per split"
                convert = lambda x: x[:-1]  # 'x\n' --> 'x'
                train_split = os.path.join(raw_data_dir, fixed_split[0])
                try:
                    files_train = list(map(convert, open(train_split, "r").readlines()))
                except FileNotFoundError:
                    files_train = []
                    log.warning("No training files.")
                val_split = os.path.join(raw_data_dir, fixed_split[1])
                try:
                    files_val = list(map(convert, open(val_split, "r").readlines()))
                except FileNotFoundError:
                    files_val = []
                    log.warning("No validation files.")
                test_split = os.path.join(raw_data_dir, fixed_split[2])
                try:
                    files_test = list(map(convert, open(test_split, "r").readlines()))
                except FileNotFoundError:
                    files_test = []
                    log.warning("No test files.")
            else:
                raise TypeError(
                    "Wrong type of parameter 'fixed_split'."
                    f" Got {type(fixed_split)} but should be"
                    "'Sequence' or 'dict'"
                )
        else:
            # Random split
            assert (
                np.sum(dataset_split_proportions) == 100
            ), "Splits need to sum to 100."
            allids = [line.rstrip()
                         for line in open(all_ids_file, 'r').readlines()]
            random.Random(dataset_seed).shuffle(allids)
            indices_train = slice(
                0, dataset_split_proportions[0] * len(allids) // 100
            )
            indices_val = slice(
                indices_train.stop,
                indices_train.stop
                + (dataset_split_proportions[1] * len(allids) // 100),
            )
            indices_test = slice(indices_val.stop, len(allids))
            files_train = allids[indices_train]
            files_val = allids[indices_val]
            files_test = allids[indices_test]

        if overfit:
            # Consider the same splits for train validation and test
            files_train = files_train[:overfit]
            files_val = files_train[:overfit]
            files_test = files_train[:overfit]

        # Save ids to file
        DatasetHandler.save_ids(files_train, files_val, files_test, save_dir)

        # Create train, validation, and test datasets
        if "train" in load_only:
            train_dataset = cls(
                ids=files_train,
                mode=DataModes.TRAIN,
                raw_data_dir=raw_data_dir,
                augment=augment_train,
                **kwargs,
            )
        else:
            train_dataset = None

        # For evaluation, the original FS outputs should be used
        if 'reduced_gt' in kwargs:
            kwargs.pop('reduced_gt')
        if 'registered_gt_meshes' in kwargs:
            kwargs.pop('registered_gt_meshes')

        if "validation" in load_only:
            val_dataset = cls(
                ids=files_val,
                mode=DataModes.VALIDATION,
                raw_data_dir=raw_data_dir,
                augment=False,
                reduced_gt=False,
                **kwargs,
            )
        else:
            val_dataset = None

        if "test" in load_only:
            test_dataset = cls(
                ids=files_test,
                mode=DataModes.TEST,
                raw_data_dir=raw_data_dir,
                augment=False,
                reduced_gt=False,
                **kwargs,
            )
        else:
            test_dataset = None

        return train_dataset, val_dataset, test_dataset

    @logger.measure_time
    def get_item_from_index(self, index: int):
        """
        One data item for training.
        """
        # Raw data
        img = self.images[index]
        voxel_label = self.voxel_labels[index]
        mesh_label = list(self._get_mesh_target_no_faces(index))
        img = img.unsqueeze(0)
        # Potentially augment
        if self._augment:
            # image level augmentation:
            biasfield = tio.RandomBiasField()
            gamma = tio.RandomGamma()
            noise = tio.RandomNoise(mean=0, std=(0, 0.125))

            img = gamma(img)
            img = biasfield(img)
            img = noise(img)

        log.debug("Dataset file %s", self.ids[index])

        return (img, voxel_label, *mesh_label)

    def _get_mesh_target_no_faces(self, index):
        return [target[index] for target in self.mesh_targets]

    def get_data_element(self, index):
        """Get image, segmentation ground truth and full reference mesh. In
        contrast to 'get_item_from_index', this function is not designed to be
        wrapped by a dataloader.
        """
        img = self.images[index][None]
        voxel_label = self.voxel_labels[index]
        mesh_label = self.mesh_labels[index]
        trans_affine_label = self.trans_affine[index]

        return {
            "img": img,
            "voxel_label": voxel_label,
            "mesh_label": mesh_label,
            "trans_affine_label": trans_affine_label,
        }

    def _read_voxelized_meshes(self, voxelized_mesh_file_names):
        """Read voxelized meshes stored in nifity files and set voxel classes
        according to groups (similar to voxel_label_names)."""
        data = []
        # Iterate over sample ids
        for _, sample_id in tqdm(
            enumerate(self.ids),
            position=0,
            leave=True,
            desc="Loading voxelized meshes...",
        ):
            voxelized_mesh_label = None
            # Iterate over voxel classes
            for group_id, voxel_group in enumerate(voxelized_mesh_file_names, 1):
                # Iterate over files
                for vmln in voxel_group:
                    vm_file = os.path.join(
                        self._raw_data_dir, sample_id, vmln + ".nii.gz"
                    )
                    img = nib.load(vm_file).get_fdata()
                    img, _ = self._get_single_patch(img, is_label=True)
                    # Assign a group id
                    img = img.bool().long() * group_id
                    if voxelized_mesh_label is None:
                        voxelized_mesh_label = img.numpy()
                    else:
                        np.putmask(voxelized_mesh_label, img.numpy() > 0, img.numpy())

            # Correct patch size
            voxelized_mesh_label, _ = self._get_single_patch(
                voxelized_mesh_label, is_label=True
            )
            data.append(voxelized_mesh_label)

        return data

    def _load_data3D_and_transform(self, filename: str, is_label: bool):
        """Load data and transform to correct patch size."""
        data = []
        transformations = []
        for fn in tqdm(self.ids, position=0, leave=True, desc="Loading images..."):
            img = nib.load(os.path.join(self._raw_data_dir, fn, filename))

            img_data = img.get_fdata()
            if self._orig_img_size is None:
                self._orig_img_size = img_data.shape
            else:
                assert np.array_equal(
                    np.array(img_data.shape), np.array(self._orig_img_size)
                ), "All images should be of equal size"
            img_data, trans_affine = self._get_single_patch(img_data, is_label)
            data.append(img_data)
            transformations.append(trans_affine)

        return data, transformations

    def _get_single_patch(self, img, is_label):
        """Extract a single patch from an image."""

        # Limits for patch selection
        lower_limit = np.array(self._patch_origin, dtype=int)
        upper_limit = np.array(self._patch_origin, dtype=int) + np.array(
            self.select_patch_size, dtype=int
        )

        # Select patch from whole image
        img_patch, trans_affine_1 = img_with_patch_size(
            img,
            self.select_patch_size,
            is_label=is_label,
            mode="crop",
            crop_at=(lower_limit + upper_limit) // 2,
            pad_value=(0 if is_label else self.PAD_VALUE),
        )
        # Zoom to certain size
        if self.patch_size != self.select_patch_size:
            img_patch, trans_affine_2 = img_with_patch_size(
                img_patch, self.patch_size, is_label=is_label, mode="interpolate"
            )
        else:
            trans_affine_2 = np.eye(self.ndims + 1)  # Identity

        trans_affine = trans_affine_2 @ trans_affine_1

        return img_patch, trans_affine

    def label_to_original_size(self, img, is_label=True):
        """Transform an image/label back to the original image size"""
        # Zoom back to original resolution
        img_zoom, _ = img_with_patch_size(
            img, self.select_patch_size, is_label=is_label, mode="interpolate"
        )
        # Invert cropping: we compute the coordinates of the original image
        # center in the coordinate frame of the cropped image
        center_cropped_cropped_frame = np.array(self.patch_size) // 2
        lower_limit = np.array(self._patch_origin, dtype=int)
        upper_limit = np.array(self._patch_origin, dtype=int) + np.array(
            self.select_patch_size, dtype=int
        )
        center_cropped_orig_frame = (lower_limit + upper_limit) // 2
        center_orig_orig_frame = np.array(self._orig_img_size) // 2
        delta_centers = center_orig_orig_frame - center_cropped_orig_frame
        center_orig_cropped_frame = center_cropped_cropped_frame + delta_centers
        img_orig, _ = img_with_patch_size(
            img_zoom.cpu().numpy(),
            self._orig_img_size,
            is_label=is_label,
            mode="crop",
            crop_at=center_orig_cropped_frame,
            pad_value=(0 if is_label else self.PAD_VALUE),
        )

        return img_orig

    def _create_voxel_labels_from_meshes(self, mesh_labels):
        """Return the voxelized meshes as 3D voxel labels."""
        data = []
        # Iterate over subjects
        for i, m in tqdm(
            enumerate(mesh_labels), position=0, leave=True, desc="Voxelize meshes..."
        ):
            voxelized = torch.zeros(self.patch_size, dtype=torch.long)
            # Iterate over meshes
            for j, (v, f) in enumerate(zip(m.verts_list(), m.faces_list()), start=1):
                voxel_label = voxelize_mesh(v, f, self.patch_size)
                voxelized[voxel_label != 0] = j

            data.append(voxelized)

        return data

    def _load_inputSegmentation(self, template_path, template_name):
        data = []
        for i, fn in enumerate(
            tqdm(
                self.ids,
                position=0,
                leave=True,
                desc="Loading input segmentations...",
            )
        ):
            input_segm_fn = os.path.join(
                template_path, fn + "_" + template_name + "_segmentation.nii.gz"
            )
            segm = nib.load(input_segm_fn).get_data()
            data.append(torch.from_numpy(segm))
        return data

    def _load_dataMesh_raw(self, meshnames):
        """Load mesh such that it's registered to the respective 3D image. If
        a mesh cannot be found, a dummy is inserted if it is a test split.
        """
        data = []
        assert len(self.trans_affine) == 0, "Should be empty."
        for fn in tqdm(self.ids, position=0, leave=True, desc="Loading meshes..."):
            # Voxel coords
            orig = nib.load(os.path.join(self._raw_data_dir, fn, self.image_file_name))
            vox2world_affine = orig.affine
            world2vox_affine = np.linalg.inv(vox2world_affine)
            self.trans_affine.append(world2vox_affine)
            file_vertices = []
            file_faces = []
            for mn in meshnames:
                try:
                    mesh = trimesh.load_mesh(
                        os.path.join(self._raw_data_dir, fn, mn + ".stl")
                    )
                except ValueError:
                    try:
                        mesh = trimesh.load_mesh(
                            os.path.join(self._raw_data_dir, fn, mn + ".ply"),
                            process=False,
                        )
                    except Exception as e:
                        # Insert a dummy if dataset is test split
                        if self.mode != DataModes.TEST:
                            raise e
                        mesh = trimesh.creation.icosahedron()
                        log.warning(f"No mesh for file {fn}/{mn}," " inserting dummy.")
                # World --> voxel coordinates
                mesh.apply_transform(world2vox_affine)
                # Store min/max number of vertices
                self.n_max_vertices = (
                    np.maximum(mesh.vertices.shape[0], self.n_max_vertices)
                    if (self.n_max_vertices is not None)
                    else mesh.vertices.shape[0]
                )
                self.n_min_vertices = (
                    np.minimum(mesh.vertices.shape[0], self.n_min_vertices)
                    if (self.n_min_vertices is not None)
                    else mesh.vertices.shape[0]
                )
                # Add to structures of file
                file_vertices.append(torch.from_numpy(mesh.vertices).float())
                file_faces.append(torch.from_numpy(mesh.faces).long())

            # Treat as a batch of meshes
            mesh = Meshes(file_vertices, file_faces)
            data.append(mesh)

        return data

    def create_training_targets(self, remove_meshes=False):
        """Sample surface points, normals and curvaturs from meshes."""
        if self.mesh_labels[0] is None:
            warnings.warn(
                "Mesh labels do not exist (anymore) and no new training"
                " targets can be created."
            )
            return self.mesh_labels

        points, normals, curvs = [], [], []

        if self.registered_gt_meshes:
            log.info("Using vertices from registered meshes as reference points.")
        else:
            log.info("Sampling reference points from mesh.")

        # Iterate over mesh labels
        for i, m in tqdm(
            enumerate(self.mesh_labels),
            leave=True,
            position=0,
            desc="Get point labels from meshes...",
        ):
            # Create meshes with curvature
            curv_list = [
                curv_from_cotcurv_laplacian(v, f).unsqueeze(-1)
                for v, f in zip(m.verts_list(), m.faces_list())
            ]
            m_new = Meshes(m.verts_list(), m.faces_list(), verts_features=curv_list)
            if self.registered_gt_meshes:
                # Use vertices as points
                p = m_new.verts_padded()
                n = m_new.verts_normals_padded()
                c = m_new.verts_features_padded()
            else:
                # Sample points from mesh
                p, n, c = sample_points_from_meshes(
                    m_new,
                    self.n_ref_points_per_structure,
                    return_normals=True,
                    interpolate_features="barycentric",
                )

            points.append(p)
            normals.append(n)
            curvs.append(c)

            # Remove meshes to save memory
            if remove_meshes:
                self.mesh_labels[i] = None
            else:
                self.mesh_labels[i] = m_new

        # Placeholder for point labels
        point_classes = [torch.zeros_like(curvs[0])] * len(curvs)

        self.mesh_targets = (points, normals, curvs, point_classes)

        return self.mesh_targets

    def augment_data(self, img, label, coordinates, normals):
        assert self._augment, "No augmentation in this dataset."
        return flip_img(img, label, coordinates, normals)

    def check_augmentation_normals(self):
        """Assert correctness of the transformation of normals during
        augmentation.
        """
        py3d_mesh = self.mesh_labels[0]
        _, _, coo_f, normals_f = self.augment_data(
            self.images[0].numpy(),
            self.voxel_labels[0].numpy(),
            py3d_mesh.verts_padded(),
            py3d_mesh.verts_normals_padded(),
        )
        py3d_mesh_aug = Meshes(coo_f, py3d_mesh.faces_padded())
        # Assert up to sign of direction
        assert torch.allclose(
            normals_f, py3d_mesh_aug.verts_normals_padded(), atol=7e-03
        ) or torch.allclose(
            -normals_f, py3d_mesh_aug.verts_normals_padded(), atol=7e-03
        )
