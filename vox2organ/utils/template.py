
""" Utility functions for templates. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from collections.abc import Sequence
from copy import deepcopy

import torch
import torch.nn.functional as F
import trimesh
import numpy as np
import nibabel as nib
from trimesh.scene.scene import Scene
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import MeshesXD, Meshes

import logger
from utils.coordinate_transform import transform_mesh_affine


TEMPLATE_PATH = "../supplementary_material/"

log = logger.get_std_logger(__name__)


# Specification of different templates
TEMPLATE_SPECS = {
    "fsaverage6-smooth-no-parc": {
        "path": os.path.join(TEMPLATE_PATH, "brain_template", "fsaverage6"),
        "mesh_suffix": "_smoothed.ply",
        "parc_labels": False,
        "group_structs": [["lh_white", "rh_white"], ["lh_pial", "rh_pial"]],
        "virtual_edges": [[0, 2], [1, 3]],
    },
    "fsaverage-smooth-no-parc": {
        "path": os.path.join(TEMPLATE_PATH, "brain_template", "fsaverage"),
        "mesh_suffix": "_smoothed.ply",
        "parc_labels": False,
        "group_structs": [["lh_white", "rh_white"], ["lh_pial", "rh_pial"]],
        "virtual_edges": [[0, 2], [1, 3]],
    },
    "fsaverage-no-parc": {
        "path": os.path.join(TEMPLATE_PATH, "brain_template", "fsaverage"),
        "mesh_suffix": ".ply",
        "parc_labels": False,
        "group_structs": [["lh_white", "rh_white"], ["lh_pial", "rh_pial"]],
        "virtual_edges": [[0, 2], [1, 3]],
    },
    "fsaverage6-no-parc": {
        "path": os.path.join(TEMPLATE_PATH, "brain_template", "fsaverage6"),
        "mesh_suffix": ".ply",
        "parc_labels": False,
        "group_structs": [["lh_white", "rh_white"], ["lh_pial", "rh_pial"]],
        "virtual_edges": [[0, 2], [1, 3]],
    },
    "abdomen-spheres": {
        "path": os.path.join(TEMPLATE_PATH, "abdomen_template", "spheres"),
        "mesh_suffix": ".ply",
        "feature_suffix": "",
        "parc_labels": False,
        "group_structs": [
            ["liver"],
            ["kidney_left"],
            ["kidney_right"],
            ["spleen"],
            ["pancreas"]
        ],
        "virtual_edges": None,
    },
    "abdomen-ellipses": {
        "path": os.path.join(TEMPLATE_PATH, "abdomen_template", "ellipses"),
        "mesh_suffix": ".ply",
        "feature_suffix": "",
        "parc_labels": False,
        "group_structs": [
            ["liver"],
            ["kidney_left"],
            ["kidney_right"],
            ["spleen"],
            ["pancreas"]
        ],
        "virtual_edges": None,
    },
    "abdomen-ct-1k": {
        "path": os.path.join(
            TEMPLATE_PATH, "abdomen_template",
            "avg_template_abdomenct1k_registered"
        ),
        "mesh_suffix": ".ply",
        "feature_suffix": "",
        "parc_labels": False,
        "group_structs": [
            ["liver"],
            ["kidney_left"],
            ["kidney_right"],
            ["spleen"],
            ["pancreas"]
        ],
        "virtual_edges": None,
    },
}


def generate_sphere_template(centers: dict, radii: dict, level=6):
    """ Generate a template with spheres centered at centers and corresponding
    radii
    - level 6: 40962 vertices
    - level 7: 163842 vertices

    :param centers: A dict containing {structure name: structure center}
    :param radii: A dict containing {structure name: structure radius}
    :param level: The ico level to use

    :returns: A trimesh.scene.scene.Scene
    """
    if len(centers) != len(radii):
        raise ValueError("Number of centroids and radii must be equal.")

    scene = Scene()
    for (k, c), (_, r) in zip(centers.items(), radii.items()):
        # Get unit sphere
        sphere = ico_sphere(level)
        # Scale adequately
        v = sphere.verts_packed() * r + c

        v = v.cpu().numpy()
        f = sphere.faces_packed().cpu().numpy()

        mesh = trimesh.Trimesh(v, f)

        scene.add_geometry(mesh, geom_name=k)

    return scene

def generate_ellipsoid_template(centers: dict, radii_x: dict, radii_y: dict,
                                radii_z: dict, level=6):
    """ Generate a template with ellipsoids centered at centers and corresponding
    radii
    - level 6: 40962 vertices
    - level 7: 163842 vertices

    :param centers: A dict containing {structure name: structure center}
    :param radii_x: A dict containing {structure name: structure radius}
    :param radii_y: A dict containing {structure name: structure radius}
    :param radii_z: A dict containing {structure name: structure radius}
    :param level: The ico level to use

    :returns: A trimesh.scene.scene.Scene
    """
    if (len(centers) != len(radii_x)
        or len(centers) != len(radii_y)
        or len(centers) != len(radii_z)):
        raise ValueError("Number of centroids and radii must be equal.")

    scene = Scene()
    for (k, c), (_, r_x), (_, r_y), (_, r_z) in zip(
        centers.items(), radii_x.items(), radii_y.items(), radii_z.items()):
        # Get unit sphere
        sphere = ico_sphere(level)
        # Scale adequately
        v = sphere.verts_packed() * torch.tensor((r_x, r_y, r_z)) + c

        v = v.cpu().numpy()
        f = sphere.faces_packed().cpu().numpy()

        mesh = trimesh.Trimesh(v, f)

        scene.add_geometry(mesh, geom_name=k)

    return scene


class MeshTemplate():
    """ Class for mesh template instances.
    """
    def __init__(
        self,
        path: str,
        mesh_label_names: Sequence,
        group_structs: Sequence[Sequence[str]],
        virtual_edges: Sequence[Sequence[int]],
        mesh_suffix: str=None,
        feature_suffix: str="_reduced.aparc.DKTatlas40.annot",
        trans_affine=torch.eye(4),
        parc_labels: bool=True,
    ):

        self._mesh = _load_mesh_template(
            path,
            mesh_label_names,
            group_idx=group_structs,
            mesh_suffix=mesh_suffix,
            feature_suffix=feature_suffix,
            trans_affine=trans_affine,
            parc_labels=parc_labels,
        )
        self._virtual_edges = virtual_edges

    @property
    def mesh(self):
        """ Always return a copy of the template.
        """
        return self._mesh.clone()

    @mesh.setter
    def mesh(self, value):
        raise RuntimeError("Template mesh cannot be exchanged.")

    def create_template_batch_size(
        self,
        batch_size,
        device="cuda:0"
    ) -> MeshesXD:
        """ Create an X-dim. mesh with additional batch dimension
        """
        template = self._mesh
        verts_list = template.verts_list() * batch_size
        faces_list = template.faces_list() * batch_size
        features_list = template.verts_features_list() * batch_size
        C = template.verts_padded().shape[0]
        batch_meshes = MeshesXD(
            verts_list,
            faces_list,
            X_dims=(batch_size, C),
            verts_features=features_list,
            virtual_edges=self._virtual_edges
        ).to(torch.device(device))

        return batch_meshes


def _load_mesh_template(
    path: str,
    mesh_label_names: Sequence,
    group_idx,
    mesh_suffix: str=".ply",
    feature_suffix: str=".annot",
    trans_affine=torch.eye(4),
    parc_labels=True,
) -> MeshesXD:

    vertices = []
    faces = []
    normals = []
    features = []

    n_groups = len(group_idx)

    # Load meshes and parcellation
    for mn in mesh_label_names:
        # Group id of the surface
        group_id = [
            i for i, x in enumerate(group_idx) if any(y in mn for y in x)
        ]
        assert len(group_id) == 1, "Group ID should be unique"
        group_id = torch.tensor(group_id[0])

        m = trimesh.load_mesh(
            os.path.join(path, mn + mesh_suffix),
            process=False
        )
        m.apply_transform(trans_affine)
        vertices.append(torch.from_numpy(m.vertices).float())
        faces.append(torch.from_numpy(m.faces).long())

        # Group ID as per-vertex feature
        surf_features = F.one_hot(
            group_id, num_classes=n_groups
        ).expand(m.vertices.shape[0], -1)

        # Parcellation as per-vertex feature
        if parc_labels:
            ft = torch.from_numpy(
                nib.freesurfer.io.read_annot(
                   os.path.join(path, mn + feature_suffix)
                )[0].astype(np.int64)
            )
            # Combine -1 & 0 into one class
            ft[ft < 0] = 0
            surf_features = torch.cat([surf_features, ft.unsqueeze(1)], dim=1)

        features.append(surf_features.float())

    return Meshes(vertices, faces, verts_features=features)
