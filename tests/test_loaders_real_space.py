"""
Tests for AttoCubePLScanRealSpace frame ordering.

Frame order is checked synthetically, because every committed export is
zero-padded and so cannot exhibit the lexicographic failure.  Each synthetic
frame is filled with its own iteration number, so a test can assert *which* file
landed at an index rather than merely how many there are.  The committed
sequences then pin that the padded case still loads silently.
"""

import warnings

import numpy as np
import pytest

from tmdc_optics_tools.loaders import AttoCubePLScanRealSpace

SHAPE = (4, 5)                      # (ny, nx) — small; no frame content is analysed

# A committed 11-frame sequence, zero-padded iter_0000 … iter_0010, alongside a
# timestamped spectral file whose name also ends "_iter_0".
REAL_DIR     = "examples/data/stark-shift"
REAL_PREFIX  = "PL-dual-gate-sweep_"
REAL_FRAMES  = 11
REAL_SPECTRAL = "PL-dual-gate-sweep_26_05_15_14_03_18_iter_0.csv"


def _frame(tmp_path, name, fill) -> None:
    """Write one numeric-grid frame, filled with *fill* so it is identifiable."""
    np.savetxt(tmp_path / name, np.full(SHAPE, float(fill)), delimiter=",")


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_iter_10_sorts_after_iter_2(tmp_path):
    # Lexicographic order would give 0, 10, 2 and pair every frame with the wrong
    # index. 3..9 are absent, so the gap warning is expected alongside the order.
    for i in (0, 2, 10):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    with pytest.warns(UserWarning, match="missing iteration"):
        scan = AttoCubePLScanRealSpace(tmp_path, prefix="pl_")

    assert [f.stem for f in scan.files] == ["pl_iter_0", "pl_iter_2", "pl_iter_10"]
    # The assertion that actually matters: the frame at index 2 is iter_10's data.
    assert np.allclose(scan.load_frame(2), 10.0)


def test_mixed_padding_widths_order_numerically(tmp_path):
    # Padding width varies between exports, so it cannot be relied on: 4-digit
    # "0010" sorts before 6-digit "000002" as text, and after it as a number.
    _frame(tmp_path, "pl_iter_000002.csv", 2)
    _frame(tmp_path, "pl_iter_0010.csv", 10)
    with pytest.warns(UserWarning, match="missing iteration"):
        scan = AttoCubePLScanRealSpace(tmp_path, prefix="pl_")

    assert np.allclose(scan.load_frame(0), 2.0)
    assert np.allclose(scan.load_frame(1), 10.0)


def test_consecutive_frames_do_not_warn(tmp_path):
    for i in (0, 1, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan = AttoCubePLScanRealSpace(tmp_path, prefix="pl_")

    assert [str(c.message) for c in caught] == []
    assert scan.n_frames == 3


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def test_gap_in_frames_warns_and_does_not_close_up(tmp_path):
    for i in (0, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    with pytest.warns(UserWarning, match=r"missing iteration\(s\) \[1\]"):
        scan = AttoCubePLScanRealSpace(tmp_path, prefix="pl_")

    # Nothing is dropped and nothing is invented; index 1 is iteration 2.
    assert scan.n_frames == 2
    assert np.allclose(scan.load_frame(1), 2.0)


def test_frames_without_iter_suffix_warn(tmp_path):
    for name in ("a", "b"):
        _frame(tmp_path, f"pl_{name}.csv", 0)
    with pytest.warns(UserWarning, match="no '_iter_N' suffix"):
        scan = AttoCubePLScanRealSpace(tmp_path, prefix="pl_")

    assert [f.stem for f in scan.files] == ["pl_a", "pl_b"]


def test_warning_points_at_the_caller(tmp_path):
    # stacklevel=3 at the call site. Without it the warning blames a line inside
    # loaders.py, which tells a researcher nothing about which scan complained.
    for i in (0, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    with pytest.warns(UserWarning) as caught:
        AttoCubePLScanRealSpace(tmp_path, prefix="pl_")

    assert caught[0].filename == __file__


# ---------------------------------------------------------------------------
# The committed zero-padded sequences
# ---------------------------------------------------------------------------


def test_committed_sequence_loads_without_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan = AttoCubePLScanRealSpace(REAL_DIR, prefix=REAL_PREFIX)

    assert [str(c.message) for c in caught] == []
    assert scan.n_frames == REAL_FRAMES
    assert scan.files[0].stem.endswith("iter_0000")
    assert scan.files[-1].stem.endswith("iter_0010")


def test_spectral_companion_excluded_despite_iter_0_collision():
    # The timestamped spectral export in this directory also ends "_iter_0", so it
    # would collide on index 0 with iter_0000 if it ever passed the numeric-grid
    # check. It is excluded by content, which is what keeps the indices distinct.
    scan = AttoCubePLScanRealSpace(REAL_DIR, prefix=REAL_PREFIX)

    assert REAL_SPECTRAL not in [f.name for f in scan.files]
    assert scan.n_frames == REAL_FRAMES
