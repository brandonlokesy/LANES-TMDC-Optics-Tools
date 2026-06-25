"""
Tests for AttoCubePLVabScan parameter loading.

Builds a small synthetic spectral CSV in the AttoCube export layout so the suite
runs without the lab network share.  Focuses on the generalized parameter store
(every labeled row exposed via ``parameters`` / ``get_parameter`` / ``[]``) and
the backward-compatible curated attributes.
"""

import numpy as np
import pytest

from tmdc_optics_tools.loaders import AttoCubePLVabScan

# 3 sweeps x 8 pixels; the first six pixel rows also carry labeled scalars.
N_SWEEPS = 3
N_PIXELS = 8
WAVELENGTH = 800.0 + np.arange(N_PIXELS)          # 800 .. 807 nm

PARAMS = {
    "V_A":              np.array([0.0, 0.5, 1.0]),
    "V_B":              np.array([0.0, -0.5, -1.0]),
    "Excitation Power": np.array([1e-6, 2e-6, 3e-6]),
    "I_A":              np.array([1e-9, 2e-9, 3e-9]),
    "I_B":              np.array([4e-9, 5e-9, 6e-9]),
    "Galvo_X":          np.array([10.0, 11.0, 12.0]),
}
POWER_SCALE = 0.303e6  # AttoCubePLVabScan default


def _roi1(r, i):
    return 100 + r * 10 + i


def _roi2(r, i):
    return 200 + r * 10 + i


def make_spectral_csv(path) -> None:
    """Write a spectral CSV with labeled parameter rows + 2 padding columns."""
    labels = list(PARAMS)                          # first len(PARAMS) rows
    header = ["Parameters Labels"]
    for i in range(N_SWEEPS):
        header += [f"Par_{i}", f"Wavelength{i}", f"ExpROI1_{i}", f"ExpROI2_{i}"]
    header += ["", ""]                             # padding columns

    lines = [",".join(header)]
    for r in range(N_PIXELS):
        label = labels[r] if r < len(labels) else ""
        par = PARAMS[label] if label else np.zeros(N_SWEEPS)
        row = [label]
        for i in range(N_SWEEPS):
            row += [f"{par[i]}", f"{WAVELENGTH[r]}",
                    f"{_roi1(r, i)}", f"{_roi2(r, i)}"]
        row += ["", ""]                            # padding cells
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def scan(tmp_path):
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    return AttoCubePLVabScan(str(csv))


# ---------------------------------------------------------------------------
# Generic parameter store
# ---------------------------------------------------------------------------


def test_all_labeled_parameters_exposed(scan):
    assert set(scan.parameter_labels) == set(PARAMS)
    for label, expected in PARAMS.items():
        assert np.allclose(scan.parameters[label], expected), label


def test_parameter_arrays_have_sweep_length(scan):
    assert scan.n_sweeps == N_SWEEPS
    assert scan.n_pixels == N_PIXELS
    for arr in scan.parameters.values():
        assert arr.shape == (N_SWEEPS,)


def test_unlabeled_pixel_rows_excluded(scan):
    # Only the labeled rows become parameters; the 2 trailing pixel rows do not.
    assert len(scan.parameters) == len(PARAMS)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def test_get_parameter_and_scale(scan):
    assert np.allclose(scan.get_parameter("Galvo_X"), PARAMS["Galvo_X"])
    assert np.allclose(scan.get_parameter("Galvo_X", scale=2.0),
                       PARAMS["Galvo_X"] * 2.0)


def test_getitem_sugar(scan):
    assert np.allclose(scan["Galvo_X"], PARAMS["Galvo_X"])


def test_missing_parameter_raises_with_available(scan):
    with pytest.raises(KeyError) as exc:
        scan.get_parameter("does_not_exist")
    assert "Galvo_X" in str(exc.value)  # error lists available labels


# ---------------------------------------------------------------------------
# Backward-compatible curated attributes
# ---------------------------------------------------------------------------


def test_curated_attributes_unchanged(scan):
    assert np.allclose(scan.v_top, PARAMS["V_A"])           # raw
    assert np.allclose(scan.v_bot, PARAMS["V_B"])           # raw
    assert np.allclose(scan.power, PARAMS["Excitation Power"] * POWER_SCALE)
    assert np.allclose(scan.Ich1, PARAMS["I_A"] * 1e9)      # -> nA
    assert np.allclose(scan.Ich2, PARAMS["I_B"] * 1e9)      # -> nA


def test_curated_attrs_are_raw_views_of_parameters(scan):
    # parameters holds raw units; curated power/current are scaled, not raw.
    assert np.allclose(scan.v_top, scan.parameters["V_A"])
    assert np.allclose(scan.power / POWER_SCALE, scan.parameters["Excitation Power"])


def test_ef_none_without_geometry(scan):
    assert scan.ef is None
    assert np.allclose(scan.gate_axis, scan.v_top)


# ---------------------------------------------------------------------------
# Spectra / axes
# ---------------------------------------------------------------------------


def test_spectra_and_wavelength(scan):
    assert np.allclose(scan.wavelength, WAVELENGTH)
    expected = np.array([[_roi1(r, i) for i in range(N_SWEEPS)]
                         for r in range(N_PIXELS)], dtype=float)
    assert np.allclose(scan.spectra, expected)
    # energy axis is ascending
    assert np.all(np.diff(scan.energy) > 0)


def test_roi2_selection(tmp_path):
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    scan2 = AttoCubePLVabScan(str(csv), roi=2)
    expected = np.array([[_roi2(r, i) for i in range(N_SWEEPS)]
                         for r in range(N_PIXELS)], dtype=float)
    assert np.allclose(scan2.spectra, expected)


def test_padding_columns_stripped(scan):
    # 2 padding cols make raw n_cols = 14 (not divisible by 4); they must be
    # stripped to 12 -> 3 sweeps without raising.
    assert scan.n_sweeps == N_SWEEPS
