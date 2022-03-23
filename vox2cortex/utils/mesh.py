
""" Mesh representation """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from typing import Union

import numpy as np
import torch
from trimesh import Trimesh
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian
from matplotlib import cm
from matplotlib.colors import Normalize

class Mesh():
    """ Custom meshes
    The idea of this class is to hold vertices and faces of ONE mesh (which may
    consist of multiple structures) together very flexibly.
    For example, vertices may be represented by a 3D tensor (one
    dimenesion per mesh structure) or a 2D tensor of shape (V,3).

    :param vertices: torch.tensor or numpy.ndarray of vertices
    :param faces: torch.tensor or numpy.ndarray of faces
    :param normals: Vertex normals
    :param features: Vertex features
    """
    def __init__(self, vertices, faces, normals=None, features=None):
        self._vertices = vertices
        self._faces = faces
        self._ndims = vertices.shape[-1]
        self.normals = normals
        self.features = features

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, new_vertices):
        assert new_vertices.shape == self.vertices.shape
        self._vertices = new_vertices

    @property
    def ndims(self):
        return self._ndims

    # Faces cannot be changed! --> Preserve topology
    @property
    def faces(self):
        return self._faces

    @property
    def normals(self):
        return self._normals

    @normals.setter
    def normals(self, new_normals):
        if new_normals is not None:
            assert new_normals.shape[-1] == self.ndims
            if len(self.vertices.shape) == 3: # Padded
                assert new_normals.shape[0:2] == self.vertices.shape[0:2]
            else: # Packed
                assert new_normals.shape[0] == self.vertices.shape[0]
        self._normals = new_normals

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, new_features):
        if new_features is not None:
            if len(self.vertices.shape) == 3: # Padded
                assert new_features.shape[0:2] == self.vertices.shape[0:2]
            else: # Packed
                assert new_features.shape[0] == self.vertices.shape[0]
        self._features = new_features

    def to_trimesh(self, process=False):
        assert type(self.vertices) == type(self.faces)
        if isinstance(self.vertices, torch.Tensor):
            if self.vertices.ndim == 3:
                m = Meshes(self.vertices, self.faces)
                faces = m.faces_packed().cpu().numpy()
                vertices = m.verts_packed().cpu().numpy()
            else: # Vx3, Fx3
                vertices = self.vertices.cpu().numpy()
                # Remove padded faces
                faces = self.faces[(self.faces != -1).any(dim=1)].cpu().numpy()
        else:
            # numpy
            vertices = self.vertices
            faces = self.faces

        return Trimesh(vertices=vertices,
                       faces=faces,
                       process=process)

    def to_pytorch3d_Meshes(self):
        assert self.vertices.ndim == self.faces.ndim
        # Note: With current pytorch3d version, vertex normals cannot be
        # handed to Meshes object
        if self.vertices.ndim == 3:
            # Avoid pytorch3d dimensionality check
            return Meshes([v for v in self.vertices],
                          [f for f in self.faces])
        if self.vertices.ndim == 2:
            return Meshes([self.vertices],
                          [self.faces])
        raise ValueError("Invalid dimension of vertices and/or faces.")

    def store(self, path: str):
        """ Store only the mesh itself. For storing with features see
        'store_with_features'.  """
        t_mesh = self.to_trimesh()
        t_mesh.export(path)

    def store_with_features(self, path: str, vmin: float=0.1, vmax: float=3.0):
        """ Store a mesh together with its morphology features. Default vmin
        and vmax correspond to typical values for the visualization of cortical
        thickness."""
        t_mesh = self.to_trimesh(process=False)
        autumn = cm.get_cmap('autumn')
        color_norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        features = self.features.reshape(t_mesh.vertices.shape[0])
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        colors = autumn(color_norm(features))
        t_mesh.visual.vertex_colors = colors
        t_mesh.export(path)

    def get_occupied_voxels(self, shape):
        """Get the occupied voxels of the mesh lying within 'shape'.

        Attention: 'shape' should be defined in the same coordinte system as
        the mesh.
        """
        assert len(shape) == 3, "Shape should represent 3 dimensions."

        voxelized = self.to_trimesh().voxelized(1.0).fill()
        # Coords = trimesh coords + translation
        vox_occupied = np.around(voxelized.sparse_indices +\
            voxelized.translation).astype(int)

        # 0 <= coords < shape
        vox_occupied = np.asarray(vox_occupied)
        mask = np.ones((vox_occupied.shape[0]), dtype=bool)
        for i, s in enumerate(shape):
            in_box = np.logical_and(vox_occupied[:,i] >= 0,
                                    vox_occupied[:,i] < s)
            mask = np.logical_and(mask, in_box)
        vox_occupied = vox_occupied[mask]

        if vox_occupied.size < 1:
            # No occupied voxels in the given shape
            vox_occupied = None

        return vox_occupied

class MeshesOfMeshes():
    """ Extending pytorch3d.structures.Meshes so that each mesh in a batch of
    meshes can consist of several distinguishable meshes (often individual
    structures in a scene). Basically, a new dimension 'M' is introduced
    to tensors of vertices and faces.
    Compared to pytorch3d.structures.Meshes, we do not sort out padded vertices
    in packed representation but assume that every sub-mesh has the same number
    of vertices and faces.
    TODO: Maybe add this functionality in the future.

    Shapes of self.faces (analoguously for vertices and normals):
        - padded (N,M,F,3)
        - packed (N*M*F,3)
    where N is the batch size, M is the number of meshes per sample, and F
    is the number of faces per connected mesh. In general, M and F can be
    different for every mesh and their maximum is used in the padded
    representation.

    """
    def __init__(self, verts, faces, features=None):
        if verts.ndim != 4:
            raise ValueError("Vertices are required to be a 4D tensor.")
        if faces.ndim != 4:
            raise ValueError("Faces are required to be a 4D tensor.")
        if features is not None:
            self.contains_features = True
            if features.ndim != 4:
                raise ValueError("Features are required to be a 4D tensor.")
        else:
            self.contains_features = False

        self._verts_padded = verts
        self._faces_padded = faces
        self._edges_packed = None
        self.ndims = verts.shape[-1]

        if features is not None:
            self.update_features(features)
        else:
            self._features_padded = None

    def update_features(self, features):
        """ Add features to the mesh in padded representation """
        if features.shape[:-1] != self._verts_padded.shape[:-1]:
            raise ValueError("Invalid feature shape.")
        self._features_padded = features

    def verts_padded(self):
        return self._verts_padded

    def features_padded(self):
        return self._features_padded

    def faces_padded(self):
        return self._faces_padded

    def edges_packed(self):
        """ Returns unique edges in packed representation.
        Based on pytorch3d.structures.Meshes.edges_packed()"""
        if self._edges_packed is None:
            if self.ndims == 3:
                # Calculate edges from faces
                faces = self.faces_packed()
                v0, v1, v2 = faces.chunk(3, dim=1)
                e01 = torch.cat([v0, v1], dim=1)  # (N*M*F), 2)
                e12 = torch.cat([v1, v2], dim=1)  # (N*M*F), 2)
                e20 = torch.cat([v2, v0], dim=1)  # (N*M*F), 2)
                # All edges including duplicates.
                edges = torch.cat([e12, e20, e01], dim=0) # (N*M*F)*3, 2)
            else:
                # 2D equality of faces and edges
                edges = self.faces_packed()

            # Sort the edges in increasing vertex order to remove duplicates as
            # the same edge may appear in different orientations in different
            # faces.
            # i.e. rows in edges after sorting will be of the form (v0, v1)
            # where v1 > v0.
            # This sorting does not change the order in dim=0.
            edges, _ = edges.sort(dim=1)

            # Remove duplicate edges: convert each edge (v0, v1) into an
            # integer hash = V * v0 + v1; this allows us to use the
            # scalar version of unique which is much faster than
            # edges.unique(dim=1) which is very slow. After finding
            # the unique elements reconstruct the vertex indices as:
            # (v0, v1) = (hash/V, hash % V) The inverse maps from
            # unique_edges back to edges: unique_edges[inverse_idxs] == edges
            # i.e. inverse_idxs[i] == j means that edges[i] == unique_edges[j]

            V = self.verts_packed().shape[0]
            edges_hash = V * edges[:, 0] + edges[:, 1]
            u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)

            self._edges_packed = torch.stack([u // V, u % V], dim=1)

        return self._edges_packed

    def faces_packed(self):
        """ Packed representation of faces """
        N, M, V, _ = self._verts_padded.shape
        _, _, F, _ = self._faces_padded.shape
        # New face index = local index + Ni * Mi * V
        add_index = torch.cat(
            [torch.ones(F) * i * V for i in range(N*M)]
        ).long().to(self._faces_padded.device)
        return self._faces_padded.view(-1, self.ndims) + add_index.view(-1, 1)

    def verts_packed(self):
        """ Packed representation of vertices """
        return self._verts_padded.view(-1, self.ndims)

    def move_verts(self, offset):
        """ Move the vertex coordinates by offset """
        if offset.shape != self._verts_padded.shape:
            raise ValueError("Invalid offset.")
        self._verts_padded = self._verts_padded + offset

    def features_verts_packed(self):
        """ (features, verts) in packed representation """
        return torch.cat((self.features_packed(), self.verts_packed()), dim=1)

    def features_packed(self):
        """ Packed representation of features """
        if self.contains_features:
            _, _, _, C = self._features_padded.shape
            return self._features_padded.view(-1, C)
        return None

def verts_faces_to_Meshes(verts, faces, ndim):
    """ Convert lists of vertices and faces to lists of
    pytorch3d.structures.Meshes

    :param verts: Lists of vertices.
    :param faces: Lists of faces.
    :param ndim: The list dimensions.
    :returns: A list of Meshes of dimension n_dim.
    """
    meshes = []
    for v, f in zip(verts, faces):
        if ndim > 1:
            meshes.append(verts_faces_to_Meshes(v, f, ndim-1))
        else:
            meshes.append(Meshes(verts=list(v), faces=list(f)))

    return meshes

def curv_from_cotcurv_laplacian(verts_packed, faces_packed):
    """ Construct the cotangent curvature Laplacian as done in
    pytorch3d.loss.mesh_laplacian_smoothing and use it for approximation of the
    mean curvature at each vertex. See also
    - Nealen et al. "Laplacian Mesh Optimization", 2006
    """
    # No backprop through the computation of the Laplacian (taken as a
    # constant), similar to pytorch3d.loss.mesh_laplacian_smoothing
    with torch.no_grad():
        L, inv_areas = cot_laplacian(verts_packed, faces_packed)
        L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1,1)
        norm_w = 0.25 * inv_areas

    return torch.norm(
        (L.mm(verts_packed) - L_sum * verts_packed) * norm_w,
        dim=1
    )
