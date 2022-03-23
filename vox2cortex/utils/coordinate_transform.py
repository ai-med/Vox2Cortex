
""" Transformation of image/mesh coordinates. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from typing import Union, Tuple

import torch
import numpy as np

def normalize_vertices(vertices: Union[torch.Tensor, np.array],
                       shape: Tuple[int, int, int], faces=None):
    """ Normalize vertex coordinates from [0, patch size-1] into [-1, 1]
    treating each dimension separately and flip x- and z-axis.

    Flipping x- and z-axis also changes the direction of normals. If the
    ordering of faces follows a normal convention, they can be also provided
    and will modified appropriately.
    """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    if isinstance(vertices, torch.Tensor):
        shape = torch.tensor(
            shape
        ).float().to(vertices.device).flip(dims=[0]).unsqueeze(0)
        vertices = vertices.flip(dims=[1])
    elif isinstance(vertices, np.ndarray):
        shape = np.flip(np.array(shape, dtype=float), axis=0)[None]
        vertices = np.flip(vertices, axis=1)
    else:
        raise TypeError()

    new_verts = 2*(vertices/(shape-1) - 0.5)

    if faces is None:
        return new_verts

    assert len(faces.shape) == 2, "Faces should be packed."
    assert faces.shape[1] == 3, "Faces should be 3D."
    if isinstance(faces, torch.Tensor):
        new_faces = faces.flip(dims=[1])
    elif isinstance(faces, np.ndarray):
        new_faces = np.flip(faces, axis=1)
    else:
        raise TypeError()

    return new_verts, new_faces

def unnormalize_vertices(vertices: Union[torch.Tensor, np.array],
                         shape: Tuple[int, int, int], faces=None):
    """ Inverse of 'normalize vertices' """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    if isinstance(vertices, torch.Tensor):
        shape = torch.tensor(shape).float().to(vertices.device).unsqueeze(0)
        vertices = vertices.flip(dims=[1])
    elif isinstance(vertices, np.ndarray):
        shape = np.array(shape, dtype=float)[None]
        vertices = np.flip(vertices, axis=1)
    else:
        raise TypeError()

    new_verts = (0.5 * vertices + 0.5) * (shape - 1)

    if faces is None:
        return new_verts

    assert len(faces.shape) == 2, "Faces should be packed."
    assert faces.shape[1] == 3, "Faces should be 3D."
    if isinstance(faces, torch.Tensor):
        new_faces = faces.flip(dims=[1])
    elif isinstance(faces, np.ndarray):
        new_faces = np.flip(faces, axis=1)
    else:
        raise TypeError()

    return new_verts, new_faces

def normalize_vertices_per_max_dim(vertices: Union[torch.Tensor, np.array],
                                   shape: Tuple[int, int, int],
                                   return_affine=False):
    """ Normalize vertex coordinates w.r.t. the maximum input dimension. If
    return_affine is specified, a matrix m is returned such that the transformed
    coordinates are obtained as v_new = (m @ v.T).T
    """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    v_new = 2*(vertices/(np.max(shape)-1) - 0.5)

    if not return_affine:
        return v_new

    mult = 2.0/(np.max(shape)-1)
    add = -1
    return v_new, np.array([[mult, 0, 0, add],
                            [0, mult, 0, add],
                            [0, 0, mult, add],
                            [0, 0, 0, 1]])

def unnormalize_vertices_per_max_dim(vertices: Union[torch.Tensor, np.array],
                                     shape: Tuple[int, int, int]):
    """ Inverse of 'normalize vertices_per_max_dim' """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    return (0.5 * vertices + 0.5) * (np.max(shape) - 1)

def transform_mesh_affine(vertices: Union[np.ndarray, torch.Tensor],
                          faces: Union[np.ndarray, torch.Tensor],
                          transformation_matrix: Union[np.ndarray, torch.Tensor]):
    """ Transform vertices of shape (V, D) or (S, V, D) using a given
    transformation matrix such that v_new = (mat @ v.T).T. """

    ndims = vertices.shape[-1]
    if (tuple(transformation_matrix.shape) != (ndims + 1, ndims + 1)):
        raise ValueError("Wrong shape of transformation matrix.")

    # Convert to torch if necessary
    if isinstance(vertices, np.ndarray):
        vertices_ = torch.from_numpy(vertices).float()
    else:
        vertices_ = vertices
    if isinstance(faces, np.ndarray):
        faces_ = torch.from_numpy(faces).long()
    else:
        faces_ = faces
    if isinstance(transformation_matrix, np.ndarray):
        transformation_matrix = torch.from_numpy(
            transformation_matrix
        ).float().to(vertices_.device)
    vertices_ = vertices_.view(-1, ndims)
    faces_ = faces_.view(-1, ndims)
    coords = torch.cat(
        (vertices_.T, torch.ones(1, vertices_.shape[0]).to(vertices_.device)),
        dim=0
    )

    # Transform
    new_coords = (transformation_matrix @ coords)

    # Adapt faces s.t. normal convention is still fulfilled
    if torch.sum(torch.sign(torch.diag(transformation_matrix)) == -1) % 2 == 1:
        new_faces = faces_.flip(dims=[1])
    else: # No flip required
        new_faces = faces_

    # Correct shape
    new_coords = new_coords.T[:,:-1].view(vertices.shape)
    new_faces = new_faces.view(faces.shape)

    # Correct data type
    if isinstance(vertices, np.ndarray):
        new_coords = new_coords.numpy()
    if isinstance(faces, np.ndarray):
        new_faces = new_faces.numpy()

    return new_coords, new_faces
