# tmdc_optics_tools/__init__.py
"""
tmdc_optics_tools
==================
A personal research toolkit for TMD optoelectronics and photonics.

Submodules
----------
constants   Physical constants, material parameters.
loaders     Data loaders (DeviceGeometry, AttoCubeSpectralSweep).
plotting    Publication-ready figure style and common plot types.
fitting     Spectral fitting (Lorentzian, Gaussian, multi-peak).
processing  Smoothing, normalisation, spectral conversions.
hdf5        Self-describing HDF5 storage for spectral sweeps.

Quick start
-----------
>>> from tmdc_optics_tools.loaders import DeviceGeometry, AttoCubeSpectralSweep
>>> from tmdc_optics_tools import plotting
>>>
>>> plotting.set_style("paper")
>>> geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
>>> scan = AttoCubeSpectralSweep(
...     "myscan.csv", spectra_type="PL", sweep="electric_field", geometry=geom,
... )
>>> print(scan)
>>> fig, ax, mesh = plotting.plot_pl_map_Vab_scan(scan, x_axis="energy")
>>> scan.to_hdf5("myscan.h5")          # metadata travels with the data
"""

__version__ = "0.1.0"
__author__  = "Brandon Loke"

from . import constants, fitting, plotting

from .loaders import (
    DeviceGeometry,
    AttoCubeSpectralSweep,
    AttoCubeTRPLSweep,
    AttoCubePLVabScan,
    AttoCubePLScanRealSpace,
    StackLayer,
    SingleSpectrum,
    SingleImage,
)
from . import hdf5, processing

__all__ = [
    "StackLayer",
    "DeviceGeometry",
    "AttoCubeSpectralSweep",
    "AttoCubeTRPLSweep",
    "AttoCubePLVabScan",
    "AttoCubePLScanRealSpace",
    "SingleSpectrum",
    "SingleImage",
    "constants",
    "hdf5",
    "plotting",
    "fitting",
    "processing",
]