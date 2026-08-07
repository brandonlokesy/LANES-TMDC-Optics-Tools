"""
Tests for A8: warn when a declared 1-D sweep axis is actually a raster row.

``sweep="Scanner X"`` on a flattened 2-D raster succeeds and returns that row
unchanged -- non-monotonic, repeating every ``n_inner`` points -- because
nothing compared it against what ``sweep_grid()`` already detects. The fix
warns (but does not raise) when the resolved sweep axis is non-monotonic and
``sweep_grid()`` finds a raster shape.

A minimal 2 (inner) x 3 (outer) = 6-point raster CSV is built locally, in the
same AttoCube export header layout used by ``tests/test_loaders.py``, rather
than reusing that module's ``make_spectral_csv`` helper -- its block count is
pinned to a shared ``N_SWEEPS = 3`` module constant, too small to form a
raster (``sweep_grid`` requires at least 4 points).
"""

import warnings

import numpy as np
import pytest

from tmdc_optics_tools.loaders import AttoCubeSpectralSweep

N_INNER, N_OUTER = 2, 3
N_SWEEPS = N_INNER * N_OUTER          # 6
N_PIXELS = 6

# Scanner X cycles through both values every N_INNER points (the fast axis);
# Scanner Y holds each value for a full inner cycle (the slow axis) -- the
# exact shape sweep_grid() looks for.
SCANNER_X = np.array([0.0, 1.0] * N_OUTER)
SCANNER_Y = np.repeat([0.0, 1.0, 2.0], N_INNER)
WAVELENGTH = 800.0 + np.arange(N_PIXELS)

PARAMS = {
    "Scanner X": SCANNER_X,
    "Scanner Y": SCANNER_Y,
}


def _write_raster_csv(path):
    labels = list(PARAMS)
    header = ["Parameters Labels"]
    for i in range(N_SWEEPS):
        header += [f"Par_{i}", f"Wavelength{i}", f"ExpROI1_{i}", f"ExpROI2_{i}"]

    lines = [",".join(header)]
    for r in range(N_PIXELS):
        label = labels[r] if r < len(labels) else ""
        par = PARAMS[label] if label else np.zeros(N_SWEEPS)
        row = [label]
        for i in range(N_SWEEPS):
            row += [f"{par[i]}", f"{WAVELENGTH[r]}", f"{100 + r}", f"{200 + r}"]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def raster_csv(tmp_path):
    path = tmp_path / "raster.csv"
    _write_raster_csv(path)
    return path


def test_sweep_grid_detects_the_raster(raster_csv):
    scan = AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL")
    grid = scan.sweep_grid()
    assert grid is not None
    assert grid.inner_label == "Scanner X" and grid.n_inner == N_INNER
    assert grid.outer_label == "Scanner Y" and grid.n_outer == N_OUTER


def test_declaring_the_raster_row_as_sweep_warns(raster_csv):
    with pytest.warns(UserWarning, match="non-monotonic"):
        scan = AttoCubeSpectralSweep(
            str(raster_csv), spectra_type="PL", sweep="Scanner X",
        )
    # Not an error: the declaration still succeeds and returns the raw row.
    assert np.array_equal(scan.sweep_axis, SCANNER_X)


def test_declaring_the_outer_axis_does_not_warn(raster_csv):
    # Scanner Y is monotonic (non-decreasing) over the full sweep, so it is
    # not the sawtooth case this warning is for.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL", sweep="Scanner Y")
    assert not any("non-monotonic" in str(w.message) for w in caught)


def test_undeclared_sweep_does_not_warn(raster_csv):
    # sweep=None -> sweep index is used, which is always monotonic by
    # construction; the raster shape is irrelevant here.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL")
    assert not any("non-monotonic" in str(w.message) for w in caught)
