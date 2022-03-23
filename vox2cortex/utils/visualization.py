""" Visualization of data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import os
from typing import Union
from collections.abc import Sequence

import numpy as np
# import open3d as o3d # Leads to double logging, uncomment if needed
import nibabel as nib
import matplotlib.pyplot as plt
import trimesh
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import find_contours

from data.cortex_labels import combine_labels
from utils.coordinate_transform import normalize_vertices_per_max_dim


def show_slices(slices, labels=None, save_path=None, label_mode='contour'):
    """
    Visualize image slices in a row.

    :param array-like slices: The image slices to visualize.
    :param array-like labels (optional): The image segmentation label slices.
    """

    assert label_mode in ('contour', 'fill')
    colors = ('blue', 'green', 'cyan', 'yellow')

    _, axs = plt.subplots(1, len(slices))
    if len(slices) == 1:
        axs = [axs]

    for i, s in enumerate(slices):
        axs[i].imshow(s, cmap="gray")

    if labels is not None:
        for i, l in enumerate(labels):
            if not isinstance(l, Sequence):
                l_ = [l]
            else:
                l_ = l

            for ll, col in zip(l_, colors):
                if label_mode == 'fill':
                    axs[i].imshow(ll, cmap="OrRd", alpha=0.3)
                else:
                    contours = find_contours(ll, np.max(ll)/2)
                    for c in contours:
                        axs[i].plot(c[:, 1], c[:, 0], linewidth=0.5,
                                    color=col)

    plt.suptitle("Image Slices")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def show_difference(img_1, img_2, save_path=None):
    """
    Visualize the difference of two 3D images in the center axes.

    :param array-like img_1: The first image
    :param array-like img_2: The image that should be compared to the first one
    :param save_path: Where the image is exported to
    """
    shape_1 = img_1.shape
    img_1_slices = [img_1[shape_1[0]//2, :, :],
                    img_1[:, shape_1[1]//2, :],
                    img_1[:, :, shape_1[2]//2]]
    shape_2 = img_2.shape
    assert shape_1 == shape_2, "Compared images should be of same shape."
    img_2_slices = [img_2[shape_2[0]//2, :, :],
                    img_2[:, shape_2[1]//2, :],
                    img_2[:, :, shape_2[2]//2]]
    diff = [(i1 != i2).long() for i1, i2 in zip(img_1_slices, img_2_slices)]

    fig, axs = plt.subplots(1, len(img_1_slices))
    if len(img_1_slices) == 1:
        axs = [axs]

    for i, s in enumerate(img_1_slices):
        axs[i].imshow(s, cmap="gray")

    for i, (l, ax) in enumerate(zip(diff, axs)):
        im = ax.imshow(l, cmap="OrRd", alpha=0.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    fig.tight_layout()

    plt.suptitle("Difference")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
