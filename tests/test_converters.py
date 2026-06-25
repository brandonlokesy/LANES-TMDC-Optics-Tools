"""
Tests for tmdc_optics_tools.converters.

These build small synthetic CSVs that mimic the AttoCube export layouts, so the
suite runs anywhere without the lab network share.  A couple of round-trip
checks against the real files run only when that share is mounted.
"""

from pathlib import Path

import numpy as np
import pytest

from tmdc_optics_tools import converters

h5py = pytest.importorskip("h5py")
tifffile = pytest.importorskip("tifffile")


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

WAVELENGTH = np.array([800.0, 801.0, 802.0, 803.0, 804.0])
V_A = np.array([0.0, 0.5, 1.0])
V_B = np.array([0.0, -0.5, -1.0])
N_SWEEPS = 3
N_PIXELS = WAVELENGTH.size


def _roi1(r, i):
    return 100 + r * 10 + i  # small non-negative integers


def make_spectral_csv(path: Path) -> None:
    """
    Write a spectral CSV with 3 sweeps x 5 pixels, two labeled parameter rows
    (V_A, V_B), an all-zero ExpROI2 channel, and 2 trailing padding columns to
    exercise the all-NaN strip + divisible-by-4 check.
    """
    header = ["Parameters Labels"]
    for i in range(N_SWEEPS):
        header += [f"Par_{i}", f"Wavelength{i}", f"ExpROI1_{i}", f"ExpROI2_{i}"]
    header += ["", ""]  # padding columns

    lines = [",".join(header)]
    for r in range(N_PIXELS):
        if r == 0:
            label, par = "V_A", V_A
        elif r == 1:
            label, par = "V_B", V_B
        else:
            label, par = "", np.zeros(N_SWEEPS)
        row = [label]
        for i in range(N_SWEEPS):
            row += [f"{par[i]}", f"{WAVELENGTH[r]}", f"{_roi1(r, i)}", "0"]
        row += ["", ""]  # padding cells
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


def make_image_csv(path: Path, seed: int = 0, shape=(4, 4)) -> np.ndarray:
    """Write a numeric-grid image CSV (integer counts as floats) and return it."""
    rng = np.random.default_rng(seed)
    img = rng.integers(200, 250, size=shape).astype(float)
    np.savetxt(path, img, delimiter=",", fmt="%.3f")
    return img


# ---------------------------------------------------------------------------
# Detection and path helpers
# ---------------------------------------------------------------------------


def test_is_image_csv(tmp_path):
    spec = tmp_path / "spec.csv"
    img = tmp_path / "img.csv"
    make_spectral_csv(spec)
    make_image_csv(img)
    assert converters.is_image_csv(img) is True
    assert converters.is_image_csv(spec) is False


def test_default_output_raw_to_processed(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    src = raw / "foo.csv"
    src.write_text("0\n")
    out = converters._default_output(src, ".h5")
    assert out.parent.name == "processed"
    assert out.name == "foo.h5"
    assert out.parent.exists()


def test_default_output_explicit_dir_and_file(tmp_path):
    src = tmp_path / "bar.csv"
    src.write_text("0\n")
    as_dir = converters._default_output(src, ".tif", out=tmp_path / "outdir")
    assert as_dir == tmp_path / "outdir" / "bar.tif"
    as_file = converters._default_output(src, ".tif", out=tmp_path / "x" / "name.tif")
    assert as_file == tmp_path / "x" / "name.tif"


# ---------------------------------------------------------------------------
# Spectral -> HDF5
# ---------------------------------------------------------------------------


def test_parse_spectral_csv(tmp_path):
    spec = tmp_path / "spec.csv"
    make_spectral_csv(spec)
    data = converters.parse_spectral_csv(spec)

    assert data["n_pixels"] == N_PIXELS
    assert data["n_sweeps"] == N_SWEEPS
    assert np.allclose(data["wavelength"], WAVELENGTH)
    assert np.allclose(data["parameters"]["V_A"], V_A)
    assert np.allclose(data["parameters"]["V_B"], V_B)
    expected_roi1 = np.array([[_roi1(r, i) for i in range(N_SWEEPS)]
                              for r in range(N_PIXELS)], dtype=float)
    assert np.allclose(data["roi1"], expected_roi1)
    assert np.all(data["roi2"] == 0)
    # Unlabeled pixel rows must not leak into parameters.
    assert set(data["parameters"]) == {"V_A", "V_B"}


def test_convert_spectral_to_hdf5(tmp_path):
    spec = tmp_path / "spec.csv"
    make_spectral_csv(spec)
    h5_path = converters.convert_spectral_csv_to_hdf5(spec, out=tmp_path)

    assert h5_path.suffix == ".h5"
    with h5py.File(h5_path, "r") as f:
        assert np.allclose(f["wavelength_nm"][:], WAVELENGTH)
        roi1 = f["spectra/ExpROI1"][:]
        assert roi1.dtype == np.int32          # integer counts -> compact dtype
        assert np.all(f["spectra/ExpROI2"][:] == 0)
        assert np.allclose(f["parameters/V_A"][:], V_A)
        assert f["parameters/V_A"].attrs["label"] == "V_A"
        assert f.attrs["source_filename"] == "spec.csv"
        assert f.attrs["n_sweeps"] == N_SWEEPS
        assert f.attrs["converter"] == "tmdc_optics_tools"


def test_spectral_padding_not_divisible_by_four_is_stripped(tmp_path):
    # make_spectral_csv writes 2 padding columns; raw n_cols = 14 (not /4),
    # which must be stripped to 12 without raising.
    spec = tmp_path / "spec.csv"
    make_spectral_csv(spec)
    data = converters.parse_spectral_csv(spec)
    assert data["n_sweeps"] == N_SWEEPS  # 12 / 4


# ---------------------------------------------------------------------------
# Image -> TIFF
# ---------------------------------------------------------------------------


def test_convert_image_to_tiff_lossless(tmp_path):
    img_csv = tmp_path / "img.csv"
    img = make_image_csv(img_csv)
    tif = converters.convert_image_csv_to_tiff(img_csv, out=tmp_path)

    assert tif.suffix == ".tif"
    back = tifffile.imread(tif)
    assert back.dtype == np.uint16            # integer raster -> uint16
    assert np.array_equal(back.astype(float), img)


def test_convert_image_forced_float32(tmp_path):
    img_csv = tmp_path / "img.csv"
    make_image_csv(img_csv)
    tif = converters.convert_image_csv_to_tiff(img_csv, out=tmp_path, dtype="float32")
    assert tifffile.imread(tif).dtype == np.float32


def test_convert_image_dir_to_stack(tmp_path):
    frames = []
    for k in range(3):
        f = tmp_path / f"frame_iter_{k:04d}.csv"
        frames.append(make_image_csv(f, seed=k))
    stack_path = converters.convert_image_dir_to_tiff_stack(
        tmp_path, prefix="frame_iter_", out=tmp_path)

    stack = tifffile.imread(stack_path)
    assert stack.shape == (3, 4, 4)
    for k in range(3):
        assert np.array_equal(stack[k].astype(float), frames[k])


# ---------------------------------------------------------------------------
# Routing and batch conversion
# ---------------------------------------------------------------------------


def test_convert_csv_auto_routes(tmp_path):
    spec = tmp_path / "spec.csv"
    img = tmp_path / "img.csv"
    make_spectral_csv(spec)
    make_image_csv(img)
    assert converters.convert_csv(spec, out=tmp_path).suffix == ".h5"
    assert converters.convert_csv(img, out=tmp_path).suffix == ".tif"


def test_convert_path_directory_per_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    make_spectral_csv(src / "spec.csv")
    make_image_csv(src / "a_iter_0000.csv", seed=1)
    make_image_csv(src / "a_iter_0001.csv", seed=2)

    outputs, errors = converters.convert_path(src, out=out, stack_images=False)
    assert errors == []
    suffixes = sorted(p.suffix for p in outputs)
    assert suffixes == [".h5", ".tif", ".tif"]


def test_convert_path_directory_stacked(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    make_spectral_csv(src / "spec.csv")
    make_image_csv(src / "a_iter_0000.csv", seed=1)
    make_image_csv(src / "a_iter_0001.csv", seed=2)

    outputs, errors = converters.convert_path(src, out=out, stack_images=True)
    assert errors == []
    suffixes = sorted(p.suffix for p in outputs)
    assert suffixes == [".h5", ".tif"]  # one spectral h5 + one image stack


# ---------------------------------------------------------------------------
# Optional: round-trip against the real lab files when the share is mounted
# ---------------------------------------------------------------------------

_BASE = Path(r"\\lanesnas.epfl.ch\lanes\Brandon\01_Projects\training-stark-shift"
             r"\samples\C22_HS1\measurements")
_REAL_SPEC = (_BASE / "EXP-2026-05-15-PL-dual-gate-sweep-pos2" / "raw"
              / "PL_dual_gate_sweep_26_05_15_14_47_34_iter_0.csv")


@pytest.mark.skipif(not _REAL_SPEC.exists(), reason="lab network share not mounted")
def test_real_spectral_roundtrip_matches_loader(tmp_path):
    from tmdc_optics_tools.loaders import AttoCubePLVabScan

    h5_path = converters.convert_spectral_csv_to_hdf5(_REAL_SPEC, out=tmp_path)
    scan = AttoCubePLVabScan(str(_REAL_SPEC))
    with h5py.File(h5_path, "r") as f:
        assert np.allclose(f["wavelength_nm"][:], scan.wavelength)
        assert np.allclose(f["spectra/ExpROI1"][:], scan.spectra)
        assert np.allclose(f["parameters/V_A"][:], scan.v_top)
        assert np.allclose(f["parameters/V_B"][:], scan.v_bot)
    assert h5_path.stat().st_size < _REAL_SPEC.stat().st_size  # smaller
