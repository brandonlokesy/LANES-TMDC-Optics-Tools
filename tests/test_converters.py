"""
Tests for tmdc_optics_tools.converters.

Frames are written synthetically into ``tmp_path``, so nothing here needs the
lab share.  Two cases carry the weight and neither exists on the branch this
module was ported from:

* a folder holding frames *and* a two-row spectrum must convert only the frames
  (**A9** — a first-line float test takes the spectrum for an image);
* an **unpadded** ``_iter_N`` sequence must stack in numeric order (**A7** —
  every committed export is zero-padded, which hides the lexicographic failure,
  and so was the branch's own fixture).

Each frame is filled with its own iteration number, so a test can assert *which*
file landed on a page rather than merely how many pages there are.
"""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tmdc_optics_tools import converters

SHAPE = (4, 5)                      # (ny, nx) — small; no frame content is analysed


def _frame(directory, name, fill) -> np.ndarray:
    """Write one numeric-grid frame filled with *fill*, and return it."""
    img = np.full(SHAPE, float(fill))
    np.savetxt(Path(directory) / name, img, delimiter=",")
    return img


def _spectrum(directory, name) -> None:
    """Write a two-row single spectrum: row 0 a wavelength axis, row 1 counts."""
    np.savetxt(Path(directory) / name,
               np.vstack([np.linspace(650, 700, 8), np.arange(8.0)]),
               delimiter=",")


def _export(directory, name) -> None:
    """Write a one-block AttoCube spectral export header."""
    header = "Parameters Labels,Par_0,Wavelength0,ExpROI1_0,ExpROI2_0"
    (Path(directory) / name).write_text(f"{header}\nTemperature,4.2,0.0,0.0\n")


# ---------------------------------------------------------------------------
# A9 — a spectrum is not a frame
# ---------------------------------------------------------------------------

def test_spectrum_beside_frames_is_not_converted(tmp_path):
    # The ported branch selected files by parsing the first field of line 1 as a
    # float, which a wavelength axis passes.  The spectrum would have become a
    # 2 x 8 "image".
    for i in (0, 1, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    _spectrum(tmp_path, "pl_iter_3.csv")

    report = converters.convert_path(tmp_path)

    assert len(report.outputs) == 3
    assert [p.stem for p in report.outputs] == ["pl_iter_0", "pl_iter_1", "pl_iter_2"]
    assert {f.name: kind for f, kind in report.skipped.items()} == {
        "pl_iter_3.csv": "spectrum"
    }


def test_converting_a_spectrum_directly_is_refused(tmp_path):
    _spectrum(tmp_path, "spec.csv")

    with pytest.raises(ValueError) as excinfo:
        converters.convert_image_csv_to_tiff(tmp_path / "spec.csv")

    # The refusal says what the file is and where it should go instead, in the
    # same words the loaders use.
    assert "two-row single spectrum" in str(excinfo.value)
    assert "SingleSpectrum" in str(excinfo.value)


def test_an_export_beside_frames_is_skipped_not_stacked(tmp_path):
    for i in (0, 1):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    _export(tmp_path, "pl_parameters.csv")

    stack_path = converters.convert_image_dir_to_tiff_stack(tmp_path, out=tmp_path)

    assert tifffile.imread(stack_path).shape == (2, *SHAPE)


# ---------------------------------------------------------------------------
# A7 — acquisition order, not filename order
# ---------------------------------------------------------------------------

def test_unpadded_stack_is_in_acquisition_order(tmp_path):
    # iter_2 … iter_10 unpadded: lexicographically "10" sorts before "2", so a
    # filename-ordered stack would open with the last frame.
    fills = list(range(2, 11))
    for i in fills:
        _frame(tmp_path, f"pl_iter_{i}.csv", i)

    stack_path = converters.convert_image_dir_to_tiff_stack(
        tmp_path, prefix="pl_iter_", out=tmp_path)
    stack = tifffile.imread(stack_path)

    assert stack.shape == (len(fills), *SHAPE)
    # Every page identifies its own frame, so this pins the order and not just
    # the count.
    assert [int(page[0, 0]) for page in stack] == fills


def test_a_three_frame_stack_is_three_pages(tmp_path):
    # Left to guess, tifffile stores a (3, ny, nx) array as one RGB image with
    # separate colour planes, so a three-point sweep opens as a single colour
    # frame.  Three is the only count that triggers it, hence three here.
    for i in (0, 1, 2):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)

    stack_path = converters.convert_image_dir_to_tiff_stack(tmp_path, out=tmp_path)

    with tifffile.TiffFile(stack_path) as tif:
        assert len(tif.pages) == 3
        assert tif.pages[0].photometric == tifffile.PHOTOMETRIC.MINISBLACK
    assert [int(page[0, 0]) for page in tifffile.imread(stack_path)] == [0, 1, 2]


def test_stack_is_named_after_the_prefix(tmp_path):
    for i in (0, 1):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)

    stack_path = converters.convert_image_dir_to_tiff_stack(
        tmp_path, prefix="pl_iter_", out=tmp_path)

    assert stack_path.name == "pl_iter_stack.tif"


def test_a_gap_warns_at_the_callers_line(tmp_path):
    # Measured, not read off a def line: the warning must be attributed to this
    # test file rather than to anything inside the package.
    for i in (0, 1, 3):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)

    with pytest.warns(UserWarning, match="missing iteration") as caught:
        converters.convert_image_dir_to_tiff_stack(tmp_path, out=tmp_path)

    assert Path(caught[0].filename).name == "test_converters.py"


def test_empty_directory_names_what_it_found(tmp_path):
    _spectrum(tmp_path, "spec.csv")

    with pytest.raises(ValueError) as excinfo:
        converters.convert_image_dir_to_tiff_stack(tmp_path)

    assert "No real-space image CSV" in str(excinfo.value)
    assert "spec.csv" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Pixel type
# ---------------------------------------------------------------------------

def test_integer_counts_become_uint16(tmp_path):
    img = _frame(tmp_path, "img.csv", 300)
    tif = converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)

    back = tifffile.imread(tif)
    assert back.dtype == np.uint16
    assert np.array_equal(back.astype(float), img)


def test_counts_above_uint16_become_uint32(tmp_path):
    # float32 cannot hold every integer past 2**24, so the fallback the branch
    # used here was not lossless.
    img = _frame(tmp_path, "img.csv", 70000)
    tif = converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)

    back = tifffile.imread(tif)
    assert back.dtype == np.uint32
    assert np.array_equal(back.astype(float), img)


def test_non_integer_counts_become_float32(tmp_path):
    _frame(tmp_path, "img.csv", 1.5)
    tif = converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)

    assert tifffile.imread(tif).dtype == np.float32


def test_dtype_can_be_forced(tmp_path):
    _frame(tmp_path, "img.csv", 300)
    tif = converters.convert_image_csv_to_tiff(
        tmp_path / "img.csv", out=tmp_path, dtype="float32")

    assert tifffile.imread(tif).dtype == np.float32


def test_unknown_dtype_is_refused(tmp_path):
    _frame(tmp_path, "img.csv", 1)

    with pytest.raises(ValueError, match="Unsupported image dtype"):
        converters.convert_image_csv_to_tiff(
            tmp_path / "img.csv", out=tmp_path, dtype="int8")


# ---------------------------------------------------------------------------
# Where output lands
# ---------------------------------------------------------------------------

def test_a_raw_folder_writes_to_its_sibling_processed(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _frame(raw, "img.csv", 1)

    tif = converters.convert_image_csv_to_tiff(raw / "img.csv")

    assert tif == tmp_path / "processed" / "img.tif"
    assert tif.is_file()


def test_any_other_folder_gets_a_processed_subfolder(tmp_path):
    _frame(tmp_path, "img.csv", 1)

    tif = converters.convert_image_csv_to_tiff(tmp_path / "img.csv")

    assert tif == tmp_path / "processed" / "img.tif"


def test_out_names_a_file_verbatim(tmp_path):
    _frame(tmp_path, "img.csv", 1)

    tif = converters.convert_image_csv_to_tiff(
        tmp_path / "img.csv", out=tmp_path / "elsewhere" / "named.tif")

    assert tif == tmp_path / "elsewhere" / "named.tif"
    assert tif.is_file()


def test_a_stack_lands_beside_the_frames_own_output(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in (0, 1):
        _frame(raw, f"pl_iter_{i}.csv", i)

    stack_path = converters.convert_image_dir_to_tiff_stack(raw)

    assert stack_path == tmp_path / "processed" / "raw_stack.tif"


# ---------------------------------------------------------------------------
# Overwriting
# ---------------------------------------------------------------------------

def test_an_existing_output_is_refused(tmp_path):
    _frame(tmp_path, "img.csv", 1)
    converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)


def test_overwrite_replaces_the_file(tmp_path):
    _frame(tmp_path, "img.csv", 1)
    converters.convert_image_csv_to_tiff(tmp_path / "img.csv", out=tmp_path)

    _frame(tmp_path, "img.csv", 7)
    tif = converters.convert_image_csv_to_tiff(
        tmp_path / "img.csv", out=tmp_path, overwrite=True)

    assert tifffile.imread(tif)[0, 0] == 7


def test_a_refused_conversion_creates_no_folder(tmp_path):
    _spectrum(tmp_path, "spec.csv")

    with pytest.raises(ValueError):
        converters.convert_image_csv_to_tiff(tmp_path / "spec.csv")

    assert not (tmp_path / "processed").exists()


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def test_recursive_run_keeps_folders_apart(tmp_path):
    # The same frame name in two sweeps.  Resolved once for the whole run, the
    # second would overwrite the first with nothing said.
    for name, fill in (("scan_a", 1), ("scan_b", 2)):
        raw = tmp_path / name / "raw"
        raw.mkdir(parents=True)
        _frame(raw, "pl_iter_0.csv", fill)

    report = converters.convert_path(tmp_path, recursive=True)

    assert len(report.outputs) == 2
    assert not report.errors
    assert tifffile.imread(tmp_path / "scan_a" / "processed" / "pl_iter_0.tif")[0, 0] == 1
    assert tifffile.imread(tmp_path / "scan_b" / "processed" / "pl_iter_0.tif")[0, 0] == 2


def test_recursive_stack_is_one_per_folder(tmp_path):
    for name in ("scan_a", "scan_b"):
        raw = tmp_path / name / "raw"
        raw.mkdir(parents=True)
        for i in (0, 1):
            _frame(raw, f"pl_iter_{i}.csv", i)

    report = converters.convert_path(tmp_path, recursive=True, stack=True)

    assert sorted(p.name for p in report.outputs) == ["raw_stack.tif", "raw_stack.tif"]
    assert {p.parent.parent.name for p in report.outputs} == {"scan_a", "scan_b"}


def test_a_batch_continues_past_a_failure(tmp_path):
    for i in (0, 1):
        _frame(tmp_path, f"pl_iter_{i}.csv", i)
    # Pre-place one output so its conversion is refused while the other succeeds.
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "pl_iter_0.tif").write_bytes(b"")

    report = converters.convert_path(tmp_path)

    assert [p.stem for p in report.outputs] == ["pl_iter_1"]
    assert len(report.errors) == 1
    assert report.errors[0][0].name == "pl_iter_0.csv"
    assert "already exists" in report.errors[0][1]


def test_a_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        converters.convert_path(tmp_path / "nope")


def test_report_unpacks_as_a_tuple(tmp_path):
    _frame(tmp_path, "pl_iter_0.csv", 1)

    outputs, skipped, errors = converters.convert_path(tmp_path)

    assert len(outputs) == 1 and not skipped and not errors


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_cli_writes_frames_and_returns_zero(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in (0, 1):
        _frame(raw, f"pl_iter_{i}.csv", i)

    assert converters.main([str(raw)]) == 0
    assert sorted(p.name for p in (tmp_path / "processed").iterdir()) == [
        "pl_iter_0.tif", "pl_iter_1.tif"
    ]


def test_cli_stack_writes_one_file(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in (0, 1):
        _frame(raw, f"pl_iter_{i}.csv", i)

    assert converters.main([str(raw), "--stack"]) == 0
    assert [p.name for p in (tmp_path / "processed").iterdir()] == ["raw_stack.tif"]


def test_cli_returns_one_on_failure(tmp_path):
    _frame(tmp_path, "pl_iter_0.csv", 1)
    converters.main([str(tmp_path)])

    # Second run has nothing to overwrite with, so every conversion is refused.
    assert converters.main([str(tmp_path)]) == 1


def test_cli_returns_one_on_a_missing_path(tmp_path):
    assert converters.main([str(tmp_path / "nope")]) == 1
