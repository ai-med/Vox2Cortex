
""" Utility functions for sphere templates. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from trimesh import Trimesh
from trimesh.scene.scene import Scene
from pytorch3d.utils import ico_sphere
import torch

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

        mesh = Trimesh(v, f)

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

        mesh = Trimesh(v, f)

        scene.add_geometry(mesh, geom_name=k)

    return scene
