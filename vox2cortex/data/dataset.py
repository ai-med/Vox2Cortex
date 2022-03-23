
""" Making datasets accessible

The file contains one base class for all datasets and a separate subclass for
each used dataset.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import os

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm

from utils.modes import DataModes
from utils.mesh import Mesh
from utils.eval_metrics import Jaccard
from utils.logging import (
    write_scatter_plot_if_debug,
)
from utils.utils import (
    voxelize_mesh,
    sample_outer_surface_in_voxel,
    sample_inner_volume_in_voxel,
)
from utils.coordinate_transform import (
    unnormalize_vertices_per_max_dim,
    normalize_vertices_per_max_dim
)

def _box_in_bounds(box, image_shape):
    """ From https://github.com/cvlab-epfl/voxel2mesh """
    newbox = []
    pad_width = []

    for box_i, shape_i in zip(box, image_shape):
        pad_width_i = (max(0, -box_i[0]), max(0, box_i[1] - shape_i))
        newbox_i = (max(0, box_i[0]), min(shape_i, box_i[1]))

        newbox.append(newbox_i)
        pad_width.append(pad_width_i)

    needs_padding = any(i != (0, 0) for i in pad_width)

    return newbox, pad_width, needs_padding

def crop_indices(image_shape, patch_shape, center):
    """ From https://github.com/cvlab-epfl/voxel2mesh """
    box = [(i - ps // 2, i - ps // 2 + ps) for i, ps in zip(center, patch_shape)]
    box, pad_width, needs_padding = _box_in_bounds(box, image_shape)
    slices = tuple(slice(i[0], i[1]) for i in box)
    return slices, pad_width, needs_padding

def crop(image, patch_shape, center, mode='constant'):
    """ From https://github.com/cvlab-epfl/voxel2mesh """
    slices, pad_width, needs_padding = crop_indices(image.shape, patch_shape, center)
    patch = image[slices]

    if needs_padding and mode != 'nopadding':
        if isinstance(image, np.ndarray):
            if len(pad_width) < patch.ndim:
                pad_width.append((0, 0))
            patch = np.pad(patch, pad_width, mode=mode)
        elif isinstance(image, torch.Tensor):
            raise NotImplementedError("Check implementation before using.")
            # assert len(pad_width) == patch.dim(), "not supported"
            # patch = F.pad(patch, tuple([int(element) for element in np.flip(np.array(pad_width), axis=0).flatten()]), mode=mode)

    return patch

def img_with_patch_size(img: np.ndarray, patch_size: int, is_label: bool,
                        mode='crop', crop_at=None) -> torch.tensor:
    """ Pad/interpolate an image such that it has a certain shape.

    :param img: The image to adapt.
    :param patch_size: The desired output patch size.
    :param is_label: Whether the image is a label map or not.
    :param mode: Either 'crop' or 'interpolate'.
    :param crop_at: The center of the patches. Only relevant if mode='crop'.

    :returns: The input image with adapted shape and an affine transformation
    matrix that can be used to adapt image coordinates (of vertices).
    """

    D, H, W = img.shape
    D_new, H_new, W_new = patch_size

    transform_affine = np.eye(4)

    if mode == 'crop':
        if crop_at is None:
            center_z, center_y, center_x = D // 2, H // 2, W // 2
        else:
            center_z, center_y, center_x = crop_at
        img = crop(img, (D_new, H_new, W_new), (center_z, center_y, center_x))
        if isinstance(img, np.ndarray):
            if is_label:
                img = torch.from_numpy(img).long()
            else:
                img = torch.from_numpy(img).float()
        transform_affine[:-1,-1] = offset_due_to_padding_and_shift(
            (center_z, center_y, center_x), patch_size
        )

    elif mode == 'interpolate':
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        transform_affine[np.diag_indices(3)] =\
                np.array(patch_size, dtype=float) / np.array(img.shape, dtype=float)
        if is_label:
            img = F.interpolate(img[None, None].float(),
                                patch_size,
                                mode='nearest')[0, 0].long()
        else:
            img = F.interpolate(img[None, None],
                                patch_size,
                                mode='trilinear',
                                align_corners=False)[0, 0]

    else:
        raise ValueError("Unknown mode.")

    assert img.shape == torch.Size(patch_size)

    return img, transform_affine

def offset_due_to_padding_and_shift(crop_at, patch_shape):
    """ Get the voxel coordinate offset due to padding and shift of an image with shape
    'old_shape' such that it has 'new_shape'
    """
    offset = [(i - ps // 2) for i, ps in zip(crop_at, patch_shape)]
    offset = -np.array(offset)

    return offset

def rotate90(img, label):
    """ Rotate an image and the corresponding voxel label.
    """
    if np.random.rand(1) > 0.5:
        img, label = np.rot90(img, 1, [0,1]), np.rot90(label, 1, [0,1])
    if np.random.rand(1) > 0.5:
        img, label = np.rot90(img, 1, [1,2]), np.rot90(label, 1, [1,2])
    if np.random.rand(1) > 0.5:
        img, label = np.rot90(img, 1, [2,0]), np.rot90(label, 1, [2,0])

    return img, label

def flip_img(img, label, coordinates=None, normals=None):
    """ Flip an img and the corresponding voxel label. Optionally, a
    corresponding mesh can also be flipped.

    Note: The coordinates need to be given in the image coordinate system.
    """
    if coordinates is None and normals is None: # No mesh vertices
        if np.random.rand(1) > 0.5:
            img, label = np.flip(img, 0), np.flip(label, 0)
        if np.random.rand(1) > 0.5:
            img, label = np.flip(img, 1), np.flip(label, 1)
        if np.random.rand(1) > 0.5:
            img, label = np.flip(img, 2), np.flip(label, 2)

        return img, label

    # Flipping image in a certain axis is equivalent to multiplication
    # of centered coordinates with (-1). Normals just need to be multiplied
    # with -1 in the respective axis.
    img_shape = img.squeeze().shape
    co_shape = coordinates.shape
    assert normals.shape == coordinates.shape
    coordinates = coordinates.view(-1, 3)
    normals = normals.view(-1, 3)
    if np.random.rand(1) > 0.5:
        img, label = np.flip(img, 0).copy(), np.flip(label, 0).copy()
        coordinates[:,0] = coordinates[:,0] * (-1) + img_shape[0] - 1
        normals[:,0] = normals[:,0] * (-1)
    if np.random.rand(1) > 0.5:
        img, label = np.flip(img, 1).copy(), np.flip(label, 1).copy()
        coordinates[:,1] = coordinates[:,1] * (-1) + img_shape[1] - 1
        normals[:,1] = normals[:,1] * (-1)
    if np.random.rand(1) > 0.5:
        img, label = np.flip(img, 2).copy(), np.flip(label, 2).copy()
        coordinates[:,2] = coordinates[:,2] * (-1) + img_shape[2] - 1
        normals[:,2] = normals[:,2] * (-1)

    # Back to original shape
    coordinates = coordinates.view(co_shape)
    normals = normals.view(co_shape)

    return img, label, coordinates, normals

def sample_surface_points(y_label, n_classes, point_count=3000):
    """ Sample outer surface points from a volume label """
    surface_points_normalized_all = []
    shape = y_label.shape
    for c in range(1, n_classes):
        y_label_outer = sample_outer_surface_in_voxel((y_label==c).long())
        surface_points = torch.nonzero(y_label_outer)
        # Point coordinates
        surface_points_normalized = normalize_vertices_per_max_dim(
            surface_points, shape
        )
        # debug
        write_scatter_plot_if_debug(surface_points_normalized,
                                    "../misc/surface_points.png")
        n_points = len(surface_points_normalized)
        perm = torch.randperm(n_points)
        # randomly pick a maximum of point_count points
        surface_points_normalized = surface_points_normalized[
            perm[:np.min([n_points, point_count])]
        ].cuda()
        # pad s.t. a batch can be created
        if n_points < point_count:
            surface_points_normalized = F.pad(
                surface_points_normalized, (0, 0, 0, point_count-n_points)
            )
        surface_points_normalized_all.append(surface_points_normalized)

    return torch.stack(surface_points_normalized_all)


class DatasetHandler(torch.utils.data.Dataset):
    """
    Base class for all datasets. It implements a map-style dataset, see
    https://pytorch.org/docs/stable/data.html.

    :param list ids: The ids of the files the dataset split should contain
    :param DataModes datamode: TRAIN, VALIDATION, or TEST

    """

    def __init__(self, ids: list, mode: DataModes):
        self._mode = mode
        self._files = ids

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        if isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index {key} is out of range.")
            # get the data from direct index
            return self.get_item_from_index(key)

        raise TypeError("Invalid argument type.")

    def get_file_name_from_index(self, index):
        """ Return the filename corresponding to an index in the dataset """
        return self._files[index]

    @staticmethod
    def save_ids(train_ids, val_ids, test_ids, save_dir):
        """ Save ids to a file """
        filename = os.path.join(save_dir, "dataset_ids.txt")

        with open(filename, 'w') as f:
            f.write("##### Training ids #####\n\n")
            for idx, id_i in enumerate(train_ids):
                f.write(f"{idx}: {id_i}\n")
            f.write("\n\n")
            f.write("##### Validation ids #####\n\n")
            for idx, id_i in enumerate(val_ids):
                f.write(f"{idx}: {id_i}\n")
            f.write("\n\n")
            f.write("##### Test ids #####\n\n")
            for idx, id_i in enumerate(test_ids):
                f.write(f"{idx}: {id_i}\n")
            f.write("\n\n")


    def get_item_and_mesh_from_index(self, index):
        """ Return the 3D data plus a mesh """
        raise NotImplementedError

    def get_item_from_index(self, index: int, *args, **kwargs):
        """
        An item consists in general of (data, labels)

        :param int index: The index of the data to access.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def check_data(self):
        """ Check if voxel and mesh data is consistent """
        for i in tqdm(range(len(self)),
                      desc="Checking IoU of voxel and mesh labels"):
            data = self.get_item_and_mesh_from_index(i)
            voxel_label = data[1]
            mesh = data[2]
            shape = voxel_label.shape
            vertices, faces = mesh.vertices, mesh.faces
            faces = faces.view(self.n_m_classes, -1, 3)
            voxelized_mesh = voxelize_mesh(
                vertices, faces, shape, self.n_m_classes
            ).cuda().sum(0).bool().long() # Treat as one class

            j_vox = Jaccard(voxel_label.cuda(), voxelized_mesh.cuda(), 2)

            if j_vox < 0.85:
                img = nib.Nifti1Image(voxel_label.squeeze().cpu().numpy(), np.eye(4))
                nib.save(img, "../to_check/data_voxel_label" + self._files[i] + ".nii.gz")
                img = nib.Nifti1Image(voxelized_mesh.squeeze().cpu().numpy(), np.eye(4))
                nib.save(img, "../to_check/data_mesh_label" + self._files[i] + ".nii.gz")
                print(f"[Warning] Small IoU ({j_vox}) of voxel label and"
                      " voxelized mesh label, check files at ../to_check/")
