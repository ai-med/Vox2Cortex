
""" Create an organ mesh template from all segementation masks in the training
set. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import nibabel as nib
import numpy as np
import trimesh
from tqdm import tqdm
from scipy import ndimage
from skimage import measure

DATA_PATH = "/path/to/AbdomenCT-1K"
OUT_PATH = "/home/fabianb/work/vox2organ/supplementary_material/abdomen_template/avg_template_abdomenct1k_registered"
TRAIN_SPLIT_FILE = "/path/to/train.txt"
LABEL_FILENAME = "registered_label.nii.gz"
# Segmentation mask labels
LABEL_NAMES = {"background": 0, "liver": 1, "kidney": 2, "spleen": 3, "pancreas": 4}
ORGANS = ("liver", "kidney_right", "kidney_left", "spleen", "pancreas")
# Threshold when to accept the voxel to be part of the 'mean' shape
THRESHOLD = 0.3


def create_mesh_from_voxels(volume, mc_step_size=1):
    """ Convert a voxel volume to mesh using marching cubes

    :param volume: The voxel volume.
    :param mc_step_size: The step size for the marching cubes algorithm.
    :return: The generated mesh.
    """
    vertices_mc, faces_mc, _, _ = measure.marching_cubes(
        volume, 0, step_size=mc_step_size, allow_degenerate=False
    )

    # measure.marching_cubes uses left-hand rule for normal directions, our
    # convention is right-hand rule
    faces_mc = np.flip(faces_mc, axis=[1])

    return trimesh.Trimesh(vertices_mc, faces_mc, process=False)


ids = [l.rstrip('\n') for l in open(TRAIN_SPLIT_FILE, 'r').readlines()]
mean_segs = {name: np.zeros_like(
    nib.load(os.path.join(DATA_PATH, ids[0], LABEL_FILENAME)).get_fdata(),
    dtype=np.float32
) for name in ORGANS}
affine = nib.load(os.path.join(DATA_PATH, ids[0], LABEL_FILENAME)).affine

for i in tqdm(ids, desc="Creating mean template..."):
    label_file = os.path.join(DATA_PATH, i, "registered_label.nii.gz")
    label = nib.load(label_file).get_fdata()

    for label_name, label_id in LABEL_NAMES.items():
        # Ignore background
        if label_name == "background":
            continue
        label_binary = (label == label_id).astype(np.int32)
        # Split kidneys
        if label_name == "kidney":
            # Opening/closing
            new_binary = ndimage.binary_opening(
                ndimage.binary_closing(
                    label_binary,
                    structure=np.ones((5,5,5)),
                ),
                structure=np.ones((5,5,5))
            )
            split_map, n = ndimage.label(new_binary)
            if n > 2:
                print("More than two kidney labels, please check morphology operations.")
                continue

            x1, _, _ = np.where(split_map == 1)
            x2, _, _ = np.where(split_map == 2)

            # Find left and right kidney according to their x coordinate
            if np.average(x1) > split_map.shape[0] / 2:
                right_kidney_map = (split_map == 1).astype(np.int32)
                left_kidney_map = (split_map == 2).astype(np.int32)
            else:
                right_kidney_map = (split_map == 2).astype(np.int32)
                left_kidney_map = (split_map == 1).astype(np.int32)

            mean_segs["kidney_right"] += right_kidney_map
            mean_segs["kidney_left"] += left_kidney_map

        else:
            mean_segs[label_name] += label_binary

# Extract mesh and store
for name, mean_seg in mean_segs.items():
    th_mean_seg = (mean_seg > THRESHOLD * len(ids)).astype(np.int32)
    mesh = create_mesh_from_voxels(th_mean_seg)
    mesh.apply_transform(affine)
    mesh.export(os.path.join(OUT_PATH, name + ".ply"))
