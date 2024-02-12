
""" Handling different architectures """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from utils.utils import ExtendedEnum

from models.vox2cortex import Vox2Cortex
from models.unet_flow import UNetFlow

class ModelHandler(ExtendedEnum):
    vox2cortex = Vox2Cortex
    unetflow = UNetFlow

    try:
        from models.corticalflow import CorticalFlow
        corticalflow = CorticalFlow
    except ModuleNotFoundError:
        pass
