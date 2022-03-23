
""" Handling different architectures """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from utils.utils import ExtendedEnum

from models.vox2cortex import (
    Vox2Cortex,
)

class ModelHandler(ExtendedEnum):
    vox2cortex = Vox2Cortex
