# tmdc_optics_tools/__init__.py
"""
tmdc_optics_tools
==================
A personal research toolkit for TMD optoelectronics and photonics.

Submodules
----------
constants   Physical constants, material parameters.
loaders     Data loaders (DeviceGeometry, AttoCubePLScan).
plotting    Publication-ready figure style and common plot types.
fitting     Spectral fitting (Lorentzian, Gaussian, multi-peak).
processing  Smoothing, normalisation, spectral conversions.
converters  CSV -> HDF5 / TIFF converters for AttoCube exports.

Quick start
-----------
>>> from tmdc_optics_tools.loaders import DeviceGeometry, AttoCubePLScan
>>> from tmdc_optics_tools import plotting, processing
>>>
>>> plotting.set_style("paper")
>>> geom = DeviceGeometry(t_hbn=53, b_hbn=46, tmdc="WS2")
>>> scan = AttoCubePLScan("myscan.csv", geometry=geom)
>>> print(scan)
>>> fig, ax, mesh = plotting.plot_pl_map(scan, x_axis="energy")
"""

__version__ = "0.1.0"
__author__  = "Brandon Loke"

from . import constants, fitting, plotting

from .loaders import (
    DeviceGeometry,
    AttoCubePLVabScan,
    AttoCubePLScanRealSpace,
    StackLayer,
    SingleSpectrum,
    SingleImage,
)
from . import processing, converters
from .converters import (
    convert_csv,
    convert_path,
    convert_spectral_csv_to_hdf5,
    convert_image_csv_to_tiff,
    convert_image_dir_to_tiff_stack,
    parse_spectral_csv,
    is_image_csv,
)

__all__ = [
    "StackLayer",
    "DeviceGeometry",
    "AttoCubePLVabScan",
    "AttoCubePLScanRealSpace",
    "SingleSpectrum",
    "SingleImage",
    "constants",
    "plotting",
    "fitting",
    "processing",
    "converters",
    "convert_csv",
    "convert_path",
    "convert_spectral_csv_to_hdf5",
    "convert_image_csv_to_tiff",
    "convert_image_dir_to_tiff_stack",
    "parse_spectral_csv",
    "is_image_csv",
]