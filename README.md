# Vox2Cortex and related methods

*Note*: This repository has been refactored entirely, you can find the old Vox2Cortex repo [here](https://github.com/ai-med/Vox2Cortex/releases/tag/v0.1.0).

This repository implements several mesh-based segmentation methods for the cortex abdominal organs, namely:
- [Vox2Cortex](https://arxiv.org/abs/2203.09446)
- [V2C-Flow](https://arxiv.org/abs/2401.12938)
- [UNetFlow](https://arxiv.org/abs/2306.15515)

<p float="left">
 <img src="/media/lh_pial_deform.gif" width="300" />   <img src="/media/rh_pial_deform.gif" width="300" />
</p>
<p float="left">
<img src="/media/deform_avg_abdomen.gif" width="600" /> 
</p>

## Installation
1. Make sure you use python 3.9
2. Clone this (Vox2Cortex) repo
```
    git clone git@github.com:ai-med/Vox2Cortex.git
    cd Vox2Cortex/
```
3. Create conda environment
```
    conda env create -f requirements.yml
    conda activate vox2organ
```
4. Clone and install our [pytorch3d fork](https://github.com/fabibo3/pytorch3d) as described therein (basically running `pip install -e .` in the cloned pytorch3d repo).

## Usage
You can include new datasets directly in `vox2organ/data/supported_datasets.py`. It is generally assumed that subject data (comprising image, meshes, and segmentation maps) is stored in the form `data-raw-directory/sample-ID/subject-data`. Currently, it is required that the mapping from world to image space is equal for all images, which can be achieved by affine registration of all input images to a common template, e.g., with [niftyreg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg), and applying the computed affine transformation to the respective meshes. See the `preprocessing/` directory for preprocessing scripts.

### Inference
We provide a pre-trained V2C-Flow-S model in `vox2organ/pretrained_models/V2C-Flow-S-ADNI`. For inference with this model, we recommend copying it to an experiment dir first.
```
mkdir experiments
cp -r vox2organ/pretrained_models/V2C-Flow-S-ADNI experiments/V2C-Flow-S-ADNI
cd vox2organ
python main.py --test -n V2C-Flow-S-ADNI --dataset YOUR_DATASET
```

### Training
A V2C-Flow training on a new dataset with subsequent model testing can be started with
```
    cd vox2organ/
    python3 main.py --train --test --group "V2C-Flow-S" --dataset YOUR_DATASET
```
We recommend using the pre-trained V2C-Flow model as a starting point for cortex reconstruction to shorten training time and save resources, i.e.,
```
    python3 main.py --train --test --group "V2C-Flow-S" --dataset YOUR_DATASET --pretrained_model pretrained_models/V2C-Flow-S-ADNI/best.pt
```
For information about command-line options see
```
    python3 main.py --help
```

### Models and parameters
Training a UNetFlow model works similarly, see `vox2organ/params/groups.py` for implemented models. A list of all available parameters and their default values is in `vox2organ/params/default.py`. Parameters are overwritten in the following sequential manner: `CLI` -> `vox2organ/main.py` -> `vox2organ/params/groups.py` -> `vox2organ/params/default.py`. That is, a parameter specified in `main.py` overwrites parameter groups and default parameters etc.

### Templates
A couple of mesh templates for the cortex and the abdomen are in `supplementary_material/`; new ones can also be added, of course.

### Docker
We provide files for creating a docker image in the `docker/` directory.

### Debugging
For debugging, it is usually helpful to start training/testing on a few samples (N) with the command-line arguments `-n debug --overfit [N]`. This omits logging in wandb and writes output to a "debug" experiment.

## Coordinate convention
The coordinate convention is the following:
- Input/output meshes should be stored in scanner RAS coordinates. A simple check can be performed by loading an image/segmentation and corresponding meshes via [3D slicer](https://www.slicer.org/), selecting "RAS" as the coordinate convention for the meshes. FreeSurfer surfaces are, by default, stored in tkrRAS coordinates, see for example [this link](https://www.fieldtriptoolbox.org/faq/coordsys/); conversion from tkrRAS to scanner RAS can be done by `mris_convert --to-scanner input-surf output-surf`
- Internally, mesh coordinates are converted to image coordinates normalized by image dimensions so that they fit the requirements of [`torch.nn.functional.grid_sample`](https://pytorch.org/docs/stable/nn.functional.html?highlight=grid_sample#torch.nn.functional.grid_sample). A sample code snipped documenting this convention is also provided below.
```
import torch
import torch.nn.functional as F
a = torch.tensor([[[0,0,0],[0,0,1],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]).float()
c = torch.nonzero(a).float() - 1 # coords in [-1,1]
c = torch.flip(c, dims=[1]) # z,y,x --> x,y,z
a = a[None][None]
c = c[None][None][None]
print(F.grid_sample(a, c, align_corners=True))
```
Output:
```
tensor([[[[[1.]]]]])
```

## Normal convention
The normal convention of input meshes should follow the convention used in most libraries like
pytorch3d or trimesh. That is, the face indices are ordered such that the face
normal of a face with vertex indices (i, j, k) calculates as (vj - vi) x (vk - vi).

## Citation
If you find this work useful, please cite (depending on the used model):
```
@InProceedings{Bongratz2022Vox2Cortex,
	author    = {Bongratz, Fabian and Rickmann, Anne-Marie and P\"olsterl, Sebastian and Wachinger, Christian},
	title     = {Vox2Cortex: Fast Explicit Reconstruction of Cortical Surfaces From 3D MRI Scans With Geometric Deep Neural Networks},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month     = {June},
	year      = {2022},
	pages     = {20773-20783}
}
@article{Bongratz2023Abdominal,
	year = {2023},
	month = oct,
	publisher = {Springer Science and Business Media {LLC}},
	volume = {13},
	number = {1},
	author = {Fabian Bongratz and Anne-Marie Rickmann and Christian Wachinger},
	title = {Abdominal organ segmentation via deep diffeomorphic mesh deformations},
	journal = {Scientific Reports}
}
@article{Bongratz2024V2CFlow,
	title = {Neural deformation fields for template-based reconstruction of cortical surfaces from MRI},
	volume = {93},
	ISSN = {1361-8415},
	journal = {Medical Image Analysis},
	publisher = {Elsevier BV},
	author = {Bongratz,  Fabian and Rickmann,  Anne-Marie and Wachinger,  Christian},
	year = {2024},
	month = apr,
	pages = {103093}
}
```
