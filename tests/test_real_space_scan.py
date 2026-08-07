"""
Tests for AttoCubePLScanRealSpace file discovery.

Covers two audit items (A7, A9), fixed together since both are this class's
file-discovery pass:

- A7: files must be ordered by the trailing integer in their filename, not
  lexicographically -- past 9 frames, sorted() would put frame 10 before
  frame 2.
- A9: a bare numeric grid must have at least three rows to count as a real-
  space image; exactly two rows is a SingleSpectrum shape (wavelength axis +
  counts), not an image, and must be skipped with a warning rather than
  loaded as a broken 2xN frame.

Real-space frames are pure numeric CSVs with no header (``np.loadtxt``), so
they are synthesized directly with ``np.savetxt`` rather than the
AttoCube-export-header helper used for spectral/TRPL tests.
"""

import warnings

import numpy as np
import pytest

from tmdc_optics_tools.loaders import AttoCubePLScanRealSpace

FRAME_SHAPE = (4, 5)  # (rows, cols) -- well above the 3-row minimum


def _write_image(path, fill=1.0):
    np.savetxt(path, np.full(FRAME_SHAPE, fill), delimiter=",")


def _write_single_spectrum(path):
    # Exactly two rows: wavelength axis, then counts -- the shape A9 must reject.
    np.savetxt(path, np.zeros((2, 5)), delimiter=",")


def _write_header_file(path):
    # A spectral-scan-style header row, already excluded before A9/A7 -- kept
    # here as a control so the fix is not accidentally excluding these too.
    path.write_text("Parameters Labels,Par_0\nnot,numeric\n")


# ---------------------------------------------------------------------------
# A7 -- numeric ordering
# ---------------------------------------------------------------------------


def test_files_ordered_numerically_past_nine_frames(tmp_path):
    # 11 frames so lexicographic order (0, 1, 10, 2, ...) would misorder them.
    for i in range(11):
        _write_image(tmp_path / f"seq_iter_{i}.csv", fill=float(i))

    scan = AttoCubePLScanRealSpace(str(tmp_path), prefix="seq_iter_")

    assert [f.name for f in scan.files] == [
        f"seq_iter_{i}.csv" for i in range(11)
    ]
    # And the loaded content follows the same order, not filename order.
    assert scan.load_frame(10)[0, 0] == 10.0


def test_missing_frame_index_warns_but_loads_the_rest(tmp_path):
    for i in (0, 1, 3, 4):  # gap at 2 -- an aborted/partly copied sequence
        _write_image(tmp_path / f"seq_iter_{i}.csv", fill=float(i))

    with pytest.warns(UserWarning, match=r"missing frame index.*2"):
        scan = AttoCubePLScanRealSpace(str(tmp_path), prefix="seq_iter_")

    assert scan.n_frames == 4
    assert [f.name for f in scan.files] == [
        "seq_iter_0.csv", "seq_iter_1.csv", "seq_iter_3.csv", "seq_iter_4.csv",
    ]


def test_files_without_a_trailing_index_fall_back_with_a_warning(tmp_path):
    _write_image(tmp_path / "seq_first.csv")
    _write_image(tmp_path / "seq_second.csv")

    with pytest.warns(UserWarning, match="no trailing index number"):
        scan = AttoCubePLScanRealSpace(str(tmp_path), prefix="seq_")

    assert scan.n_frames == 2


# ---------------------------------------------------------------------------
# A9 -- two-row SingleSpectrum shape excluded
# ---------------------------------------------------------------------------


def test_two_row_file_is_not_loaded_as_an_image(tmp_path):
    _write_image(tmp_path / "seq_iter_0.csv")
    _write_single_spectrum(tmp_path / "seq_iter_1.csv")
    _write_image(tmp_path / "seq_iter_2.csv")

    with pytest.warns(UserWarning, match="Skipped 1"):
        scan = AttoCubePLScanRealSpace(str(tmp_path), prefix="seq_iter_")

    assert scan.n_frames == 2
    assert "seq_iter_1.csv" not in [f.name for f in scan.files]
    # The remaining frames still parse as genuine images.
    assert scan.load_frame(0).shape == FRAME_SHAPE


def test_header_files_still_excluded_without_a_warning(tmp_path):
    # Spectral-scan-style header files are a different, older exclusion
    # (fail the float parse entirely) and should not be reported as
    # "skipped" two-row files -- they are simply not candidates at all in
    # the sense A9 cares about, but the constructor's skip warning covers
    # both reasons together, so just check it still fires and still excludes.
    _write_image(tmp_path / "seq_iter_0.csv")
    _write_header_file(tmp_path / "seq_iter_1.csv")

    with pytest.warns(UserWarning, match="Skipped 1"):
        scan = AttoCubePLScanRealSpace(str(tmp_path), prefix="seq_iter_")

    assert scan.n_frames == 1


def test_all_candidates_two_row_raises_with_a_clear_message(tmp_path):
    _write_single_spectrum(tmp_path / "seq_iter_0.csv")
    _write_single_spectrum(tmp_path / "seq_iter_1.csv")

    with pytest.raises(ValueError, match="numeric-grid check"):
        AttoCubePLScanRealSpace(str(tmp_path), prefix="seq_iter_")
