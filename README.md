# Vox2Cortex

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Preprint](https://img.shields.io/badge/arXiv-2203.09446-b31b1b)](https://arxiv.org/abs/2203.09446)

This repository implements *Vox2Cortex* (preprint is [here](https://arxiv.org/abs/2203.09446)), a fast deep learning-based method for reconstruction of cortical surfaces from MRI.

![Alt Text](https://github.com/ai-med/Vox2Cortex/blob/main/demo/cortex_surfaces.gif)

If you find this work useful, please cite:
```
@InProceedings{Bongratz_2022_CVPR,
    author    = {Bongratz, Fabian and Rickmann, Anne-Marie and P\"olsterl, Sebastian and Wachinger, Christian},
    title     = {Vox2Cortex: Fast Explicit Reconstruction of Cortical Surfaces From 3D MRI Scans With Geometric Deep Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {20773-20783}
}
```

## Installation
1. Make sure you use python 3.8
2. Install this (Vox2Cortex) repo using pip
```
    git clone https://github.com/ai-med/Vox2Cortex.git
    cd Vox2Cortex && pip install -e .
```
3. Clone and install [this](https://github.com/fabibo3/pytorch3d/tree/vox2cortex_cvpr2022) fork of PyTorch3d analogously, i.e.,
```
    git clone https://github.com/fabibo3/pytorch3d.git
    cd pytorch3d
    git checkout tags/vox2cortex_cvpr2022 -b vox2cortex_pytorch3d
    pip install -e .
```

## Usage
You can include new cortex datasets directly in `vox2cortex/data/supported_datasets.py` and `vox2cortex/data/dataset_handler.py`. It is generally assumed that the cortex data is stored in the form `data-raw-directory/sample-ID/sample-data`, where `sample-data` includes MRI scans and ground-truth surfaces. See `vox2cortex/scripts/pre_process_oasis.py` for our preprocessing routine.

A training with subsequent model testing can be started with
```
    cd vox2cortex/
    python3 main.py --train --test
```
For further information about command-line options see
```
    python3 main.py --help
```
Model parameters (and also parameters for optimization, testing, tuning, etc.) are set in `vox2cortex/utils/params.py` and overwritten by `vox2cortex/main.py`.

In order to evaluate predicted meshes created with `--test`, please refer to `vox2cortex/scripts/eval_meshes.py`.

We provide an exemplary template with 42016 vertices per surface in `supplementary_material/templates/`. Note that this template is stored in a normalized format and is only applicable to images of size 182x218x182 (this is unfortunately not very convenient and we plan to change it in a future version). If you want to create your own template, you can use `vox2cortex/scripts/create_template_and_store.py`. Note that we used an extensively smoothed FreeSurfer mesh as starting template. The simplest workflow is probably to create an individual dataset containing the smoothed surfaces as ground truth meshes.

## Coordinate convention
We assume that the meshes are stored in world coordinates, i.e., they can be transformed via the inverse nifty header to the mri voxel space.

## Normal convention
The normal convention follows the convention used in most libraries like
pytorch3d or trimesh. That is, the face indices are ordered such that the face
normal of a face with vertex indices (i, j, k) calculates as (vj - vi) x (vk - vi).
