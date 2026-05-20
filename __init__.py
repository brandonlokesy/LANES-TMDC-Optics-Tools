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

from .loaders import DeviceGeometry, AttoCubePLVabScan, AttoCubePLScanRealSpace, StackLayer
from . import constants, plotting, fitting, processing

__all__ = [
    "StackLayer",
    "DeviceGeometry",
    "AttoCubePLVabScan",
    "constants",
    "plotting",
    "fitting",
    "processing",
]