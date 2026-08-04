"""
Tests for ``plotting.plot_spectral_map`` and its deprecated alias.

Two things are pinned. First, the y-axis comes from ``sweep_axis`` rather than
the ``gate_axis`` alias, which is what lets the alias be deleted: a test that
only checked the values would pass either way, so the declared sweep is varied
(index, power, piezo) and the axis compared against the property.

Second, ``plot_pl_map_Vab_scan`` warns and then produces a figure identical to
the new name's — a shim that warns but has drifted is worse than no shim.
"""

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tmdc_optics_tools import plotting
from tmdc_optics_tools.loaders import AttoCubeSpectralSweep

from test_loaders import PARAMS, POWER_SCALE, make_spectral_csv


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "scan.csv"
    make_spectral_csv(path)
    return path


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# The y-axis follows the declared sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sweep, expected", [
    (None,      np.arange(len(PARAMS["V_A"]))),
    ("power",   PARAMS["Excitation Power"] * POWER_SCALE),
    ("piezo_y", PARAMS["Scanner Y"]),
])
def test_y_axis_is_the_declared_sweep(csv_path, sweep, expected):
    scan = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL", sweep=sweep)
    fig, ax, mesh = plotting.plot_spectral_map(scan)

    # shading="auto" resolves to "nearest" for equal-shaped X/Y/C, so
    # _coordinates holds cell *edges* — (n_pixels+1, n_sweeps+1, 2), with the
    # outer edges extrapolated half a cell beyond the data. Averaging adjacent
    # edges of row 0 recovers the sweep-axis centres that were passed in.
    y_edges = mesh._coordinates[0, :, 1]
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    assert np.allclose(y_centres, expected)
    assert np.allclose(scan.sweep_axis, expected)
    assert ax.get_ylabel() == scan.sweep_axis_label


def test_x_axis_switches_between_energy_and_wavelength(csv_path):
    scan = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL")

    _, ax_e, _ = plotting.plot_spectral_map(scan, x_axis="energy")
    _, ax_w, _ = plotting.plot_spectral_map(scan, x_axis="wavelength")

    assert ax_e.get_xlabel() == "Energy (eV)"
    assert ax_w.get_xlabel() == "Wavelength (nm)"


# ---------------------------------------------------------------------------
# The deprecated alias
# ---------------------------------------------------------------------------

def test_old_name_warns_and_matches_new_name(csv_path):
    scan = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL", sweep="power")

    _, _, new_mesh = plotting.plot_spectral_map(scan)
    with pytest.warns(FutureWarning, match="plot_spectral_map"):
        _, _, old_mesh = plotting.plot_pl_map_Vab_scan(scan)

    assert np.allclose(new_mesh.get_array(), old_mesh.get_array())
    assert np.allclose(new_mesh._coordinates, old_mesh._coordinates)


def test_old_name_forwards_keyword_arguments(csv_path):
    scan = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL")

    with pytest.warns(FutureWarning):
        _, ax, mesh = plotting.plot_pl_map_Vab_scan(
            scan, x_axis="wavelength", median_kernel=1, clim=(0.0, 0.5),
        )

    assert ax.get_xlabel() == "Wavelength (nm)"
    assert mesh.get_clim() == (0.0, 0.5)
