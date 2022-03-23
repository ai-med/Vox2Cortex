
""" Base class for voxel2mesh models """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from abc import ABC, abstractmethod
from torch.nn import Module

class V2MModel(Module, ABC):
    """ Base class for Voxel2Mesh models """

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    @abstractmethod
    def pred_to_verts_and_faces(pred):
        pass

    @staticmethod
    @abstractmethod
    def pred_to_voxel_pred(pred):
        """ Get the voxel prediction with argmax over classes applied """
        pass

    @staticmethod
    @abstractmethod
    def pred_to_raw_voxel_pred(pred):
        """ Get the voxel prediction per class """
        pass

    @staticmethod
    @abstractmethod
    def pred_to_pred_meshes(pred):
        """ Get the pytorch3d mesh predictions """
        pass

    @staticmethod
    @abstractmethod
    def pred_to_displacements(pred):
        """ Get the displacements of vertices per step and class """
        pass

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
