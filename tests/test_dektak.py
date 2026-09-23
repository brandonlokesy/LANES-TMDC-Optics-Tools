"""
Tests for DektakScan.

Exercises loading the committed example file
(examples/data/cmi-bruker-dektak/si3n4-c02.csv), metadata parsing, nearest-
neighbour lookup, gradient, height conversion, step-height measurement, repr,
and the plotting function.
"""

import numpy as np
import pytest

from leman.loaders import DektakScan
from leman.plotting import plot_dektak_profile

from _paths import DATA

DEKTAK_FILE = str(DATA / "cmi-bruker-dektak" / "si3n4-c02.csv")


# ---------------------------------------------------------------------------
# Loading and metadata
# ---------------------------------------------------------------------------


def test_load_arrays():
    scan = DektakScan(DEKTAK_FILE)
    assert scan.lateral.shape == (4501,)
    assert scan.height.shape == (4501,)
    assert scan.n_points == 4501


def test_lateral_is_monotonic():
    scan = DektakScan(DEKTAK_FILE)
    assert np.all(np.diff(scan.lateral) > 0)


def test_metadata_keys():
    scan = DektakScan(DEKTAK_FILE)
    for key in ("Date", "ScanLength", "ScanResolution", "StylusForce",
                "ScanType", "StylusType"):
        assert key in scan.metadata, f"Missing metadata key: {key}"


def test_scan_length_and_resolution():
    scan = DektakScan(DEKTAK_FILE)
    assert scan.scan_length_um == pytest.approx(608.31)
    assert scan.scan_resolution_um == pytest.approx(0.13512)


def test_analytical_results():
    scan = DektakScan(DEKTAK_FILE)
    assert "Total_ASH" in scan.analytical_results
    assert "Value" in scan.analytical_results["Total_ASH"]


def test_leveling():
    scan = DektakScan(DEKTAK_FILE)
    assert "R Cursor Position" in scan.leveling
    assert "M Cursor Position" in scan.leveling


# ---------------------------------------------------------------------------
# get_value
# ---------------------------------------------------------------------------


def test_get_value_at_first_point():
    scan = DektakScan(DEKTAK_FILE)
    matched_x, height = scan.get_value(0.0)
    assert matched_x == pytest.approx(0.0)
    assert height == pytest.approx(-66.9724354656773)


def test_get_value_nearest_neighbour():
    scan = DektakScan(DEKTAK_FILE)
    matched_x, height = scan.get_value(300.0)
    assert isinstance(matched_x, float)
    assert isinstance(height, float)
    assert abs(matched_x - 300.0) < scan.scan_resolution_um


# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------


def test_gradient_shape():
    scan = DektakScan(DEKTAK_FILE)
    grad = scan.gradient
    assert grad.shape == scan.lateral.shape


# ---------------------------------------------------------------------------
# height_nm
# ---------------------------------------------------------------------------


def test_height_nm():
    scan = DektakScan(DEKTAK_FILE)
    np.testing.assert_allclose(scan.height_nm, scan.height / 10.0)


# ---------------------------------------------------------------------------
# step_height
# ---------------------------------------------------------------------------


def test_step_height_returns_float():
    scan = DektakScan(DEKTAK_FILE)
    result = scan.step_height((0, 50), (500, 550))
    assert isinstance(result, float)


def test_step_height_empty_region():
    scan = DektakScan(DEKTAK_FILE)
    with pytest.raises(ValueError, match="contains no data points"):
        scan.step_height((9000, 9001), (0, 50))


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr():
    scan = DektakScan(DEKTAK_FILE)
    r = repr(scan)
    assert "DektakScan" in r
    assert "4501 points" in r
    assert "µm" in r


# ---------------------------------------------------------------------------
# plot_dektak_profile
# ---------------------------------------------------------------------------


def test_plot_returns_tuple():
    scan = DektakScan(DEKTAK_FILE)
    fig, ax, line = plot_dektak_profile(scan)
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_height_unit_nm():
    scan = DektakScan(DEKTAK_FILE)
    fig, ax, line = plot_dektak_profile(scan, height_unit="nm")
    assert "nm" in ax.get_ylabel()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_bad_height_unit():
    scan = DektakScan(DEKTAK_FILE)
    with pytest.raises(ValueError, match="height_unit"):
        plot_dektak_profile(scan, height_unit="um")


def test_plot_into_existing_axes():
    import matplotlib.pyplot as plt
    scan = DektakScan(DEKTAK_FILE)
    fig, ax = plt.subplots()
    fig2, ax2, line = plot_dektak_profile(scan, ax=ax)
    assert fig2 is fig
    assert ax2 is ax
    plt.close(fig)
