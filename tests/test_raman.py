"""
Tests for RamanSpectrum.

Real LabRAM exports (examples/data/Raman/) have a `#`-prefixed header block
whose length varies file to file, and are written in latin-1 (the degree
sign in "#Detector temperature (°C)=" is not valid UTF-8) -- both are
exercised directly against those files. Synthetic files cover the shape
rejection (a spatial-map export has more than 2 columns).
"""

import numpy as np
import pytest

from tmdc_optics_tools.loaders import RamanSpectrum

RAMAN_DIR = "examples/data/Raman"


def _write_spectrum(path, n_header_lines=3, degree_sign=False):
    lines = [f"#Setting {i}=\tvalue" for i in range(n_header_lines)]
    if degree_sign:
        lines.append("#Detector temperature (\xb0C)=\t-60.1")
    shift  = np.linspace(100.0, 200.0, 10)
    counts = np.arange(10, dtype=float)
    for s, c in zip(shift, counts):
        lines.append(f"{s}\t{c}")
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("latin-1"))
    return shift, counts


# ---------------------------------------------------------------------------
# Real files -- varying header length, latin-1 encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fname", [
    "unstrained_bilayer.txt", "strained_bilayer1.txt", "strained_bilayer2.txt",
])
def test_real_files_load_with_the_same_shift_axis(fname):
    s = RamanSpectrum(f"{RAMAN_DIR}/{fname}")
    assert s.shift.size == s.counts.size == 1024
    assert s.shift[0] == pytest.approx(36.5878)
    assert s.shift[-1] == pytest.approx(554.669)
    assert np.all(np.diff(s.shift) > 0)


def test_header_length_difference_does_not_shift_the_data():
    # strained_bilayer1.txt has 2 extra "#Peaks:Edit=" lines vs. the other
    # two -- a fixed-row-count skip would misalign it relative to them.
    a = RamanSpectrum(f"{RAMAN_DIR}/unstrained_bilayer.txt")
    b = RamanSpectrum(f"{RAMAN_DIR}/strained_bilayer1.txt")
    assert np.array_equal(a.shift, b.shift)


def test_map_export_is_rejected_not_misread():
    with pytest.raises(ValueError):
        RamanSpectrum(f"{RAMAN_DIR}/map2.txt")


# ---------------------------------------------------------------------------
# Synthetic files
# ---------------------------------------------------------------------------


def test_variable_header_length_is_skipped_by_content(tmp_path):
    shift, counts = _write_spectrum(tmp_path / "short_header.txt", n_header_lines=1)
    _write_spectrum(tmp_path / "long_header.txt", n_header_lines=20)

    s_short = RamanSpectrum(tmp_path / "short_header.txt")
    s_long  = RamanSpectrum(tmp_path / "long_header.txt")

    assert np.allclose(s_short.shift, shift)
    assert np.allclose(s_short.counts, counts)
    assert np.array_equal(s_short.shift, s_long.shift)


def test_degree_sign_in_header_does_not_break_loading(tmp_path):
    path = tmp_path / "with_degree.txt"
    shift, counts = _write_spectrum(path, degree_sign=True)
    s = RamanSpectrum(path)
    assert np.allclose(s.shift, shift)


def test_no_header_at_all_still_loads(tmp_path):
    path = tmp_path / "no_header.txt"
    shift = np.linspace(50.0, 60.0, 5)
    counts = np.arange(5, dtype=float)
    path.write_text("\n".join(f"{s}\t{c}" for s, c in zip(shift, counts)) + "\n")
    s = RamanSpectrum(path)
    assert np.allclose(s.shift, shift)
    assert np.allclose(s.counts, counts)


def test_extra_columns_raise_a_clear_error(tmp_path):
    # A spatial-map-shaped row: (X, Y, counts_0, counts_1, ...) -- not a
    # single spectrum, must not be silently truncated to the first 2 columns.
    path = tmp_path / "map_like.txt"
    path.write_text("0.0\t0.0\t1\t2\t3\n0.0\t1.0\t4\t5\t6\n")
    with pytest.raises(ValueError, match="spatial-map"):
        RamanSpectrum(path)
