
""" Utility functions for cortical thickness analysis. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import numpy as np
import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance

from utils.mesh import Mesh

point_face_distance = _PointFaceDistance.apply

# Mapping a surface index to its partner relevant for computation of cortical
# thickness, order: lh_white, rh_white, lh_pial, rh_pial
# Example: thickness_partner[1] = 3 since the partner of rh_white is rh_pial
thickness_partner_4 = (2, 3, 0, 1)
thickness_partner_2 = (1, 0)

def cortical_thickness(vertices: torch.Tensor, faces: torch.Tensor):
    """ Compute the cortical thickness from vertices and faces for each
    surface.
    Note: for a meaningful thickness value, coordinates should be
    given in the original space.
    :param vertices: A tensor of shape (S, V, 3)
    :param faces: A tensor of shape (S, V, 3)
    :returns: A list of meshes containing the per-vertex cortical thickness as
    features
    """
    assert vertices.ndim == 3, "Vertices should be in padded representation. "
    assert faces.ndim == 3, "Faces should be in padded representation. "
    assert vertices.shape[0] in (2, 4), "2 or 4 surfaces required."
    assert faces.shape[0] in (2, 4), "2 or 4 surfaces required."

    # Iterate over surfaces
    result = []
    n_surfaces = vertices.shape[0]

    if n_surfaces == 4:
        thickness_partner = thickness_partner_4
    if n_surfaces == 2:
        thickness_partner = thickness_partner_2

    for i in range(n_surfaces):
        # The surface for which the thickness is computed
        surface_vertices = vertices[i]
        surface_faces = faces[i]
        # Remove padded
        surface_vertices = surface_vertices[
            ~torch.isclose(
                surface_vertices,
                torch.Tensor([-1.0]).to(surface_vertices.device)
            ).all(dim=1)
        ]
        surface_faces = surface_faces[
            ~(surface_faces == -1).any(dim=1)
        ]
        surface = Pointclouds([surface_vertices])
        # The corresponding partner to which the distance is calculated. Here,
        # it is not necessary to remove padded vertices since they only
        # represent one coordinate in space
        partner_id = thickness_partner[i]
        partner = Meshes([vertices[partner_id]], [faces[partner_id]])

        # Thickness from pointcloud to mesh
        thickness_pred = _point_mesh_face_distance_unidirectional(
            surface,
            partner
        )

        mesh = Mesh(surface_vertices,
                    surface_faces,
                    normals=None,
                    features=thickness_pred)

        result.append(mesh)

    return result # List of meshes

def _point_mesh_face_distance_unidirectional(white_pntcloud: Pointclouds,
                                             pial_mesh: Meshes):
    """ Compute the cortical thickness for every point in 'white_pntcloud' as
    its distance to the surface defined by 'pial_mesh'."""
    # The following is taken from pytorch3d.loss.point_to_mesh_distance
    # Packed representation for white matter pointclouds
    points = white_pntcloud.points_packed()  # (P, 3)
    points_first_idx = white_pntcloud.cloud_to_packed_first_idx()
    max_points = white_pntcloud.num_points_per_cloud().max().item()

    # Packed representation for faces
    verts_packed = pial_mesh.verts_packed()
    faces_packed = pial_mesh.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = pial_mesh.mesh_to_faces_packed_first_idx()

    # Point to face distance: shape # (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    # Take root as point_face_distance returns squared distances
    point_to_face = torch.sqrt(point_to_face)

    return point_to_face
