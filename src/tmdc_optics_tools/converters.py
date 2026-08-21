# tmdc_optics_tools/converters.py
"""
Convert AttoCube real-space image CSV exports to TIFF.

The AttoCube software writes a real-space frame as a bare ``H x W`` grid of
comma-separated numbers with no header — a text encoding of what is a binary
raster.  A gate or position sweep arrives as a directory of such frames, one per
acquisition point, indexed by an ``_iter_N`` filename suffix.  Text costs roughly
six bytes per count, so a sweep is an order of magnitude larger than the same
data as TIFF, and no image viewer will open it.

Converting produces either one ``.tif`` per frame, preserving the input
one-to-one, or a single multi-page ``.tif`` per directory.  A stack is the more
convenient object to hand to ImageJ, but combining frames asserts that they
belong together, so it is asked for rather than assumed.

Where output lands
------------------
A converted file goes into a ``processed/`` folder: the sibling of ``raw/`` when
the source sits in one, and a folder created beside the source otherwise.  Pass
*out* to override.  The two folder names are fixed and take no parameter.

Public functions
----------------
convert_image_csv_to_tiff
    Single image CSV -> ``.tif``.
convert_image_dir_to_tiff_stack
    A directory of image CSVs -> one multi-page ``.tif``, in acquisition order.
convert_path
    Convert a file, directory, or tree in one call, continuing past failures.
main
    ``tmdc-convert`` command-line entry point.

Notes
-----
Spectral and temporal exports are a different shape entirely and are not handled
here; they are read by the loaders and archived with
:func:`tmdc_optics_tools.hdf5.write_sweep`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tifffile

from .loaders import _CSV_KIND_REASON, _classify_csv, _order_by_iter

# The folder a source has to sit in for its output to become a sibling, and the
# folder output always lands in.  Fixed names rather than parameters: a setting
# would be a second place the answer lives, and the first time it disagreed with
# the folder on screen it would cost more than it saved.
_RAW_DIR = "raw"
_OUT_DIR = "processed"

_DTYPES = ("auto", "uint16", "uint32", "float32")

# Largest value uint16 holds.  Integer counts above it need uint32, because
# float32 cannot represent every integer past 2**24.
_U16_MAX = 65535


class ConversionReport(NamedTuple):
    """
    What a :func:`convert_path` run did.

    A tuple, so it unpacks as ``outputs, skipped, errors``.

    Attributes
    ----------
    outputs : list of Path
        Files written, in the order they were written.
    skipped : dict
        ``{Path: kind}`` for every CSV that was not a real-space image, named as
        the loaders name it.  Reported rather than warned about, because a
        directory of spectra is the likeliest way to get an empty run and
        "nothing written" is not a diagnosis.
    errors : list of tuple
        ``(source, message)`` for each conversion that raised.  A batch continues
        past a failure, so this can be non-empty alongside outputs.
    """

    outputs : list
    skipped : dict
    errors  : list


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def _default_output(src, new_suffix: str, out=None) -> Path:
    """
    Resolve where converting *src* to *new_suffix* should write.

    Resolves only — nothing is created and nothing is checked for existence, so
    this is safe to call before deciding whether to convert at all.

    Parameters
    ----------
    src : str or Path
        Source file.
    new_suffix : str
        Suffix of the output, leading dot included, e.g. ``".tif"``.
    out : str or Path, optional
        Explicit destination.  A path carrying a suffix is used verbatim; one
        without (or an existing directory) is treated as a directory and the
        output is named after *src*.

    Returns
    -------
    pathlib.Path
        With *out* omitted: ``<parent>/processed/<stem><suffix>``, where
        ``<parent>`` is the grandparent when *src* sits in a ``raw/`` folder and
        the source's own folder otherwise.
    """
    src  = Path(src)
    name = src.stem + new_suffix

    if out is not None:
        out = Path(out)
        return out / name if (out.suffix == "" or out.is_dir()) else out

    # A raw/ folder already names its counterpart, so the output belongs beside
    # it rather than nested inside it.
    root = src.parent.parent if src.parent.name.lower() == _RAW_DIR else src.parent
    return root / _OUT_DIR / name


def _claim_target(target: Path, overwrite: bool) -> Path:
    """
    Refuse *target* if it already exists, then create its parent directory.

    Overwriting is opt-in for the same reason it is in
    :func:`tmdc_optics_tools.hdf5.write_sweep`: the file being replaced may be
    the only copy.  The existence check runs first, so a refused conversion
    leaves no empty ``processed/`` folder behind.

    Raises
    ------
    FileExistsError
        If *target* exists and *overwrite* is False.
    """
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"'{target}' already exists. Pass overwrite=True (or --overwrite) to "
            f"replace it, or give a different out= path."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

def _image_frames(directory, prefix=None, *, stacklevel: int) -> tuple:
    """
    Every real-space image CSV in *directory*, in acquisition order.

    Selection is by content, not by name: a parameter export written beside the
    frames, and a two-row single spectrum, are both excluded and returned as
    *skipped* instead.  Ordering is by the ``_iter_N`` suffix, so ``iter_10``
    follows ``iter_2``; export padding widths vary and cannot be relied on.

    Parameters
    ----------
    directory : str or Path
    prefix : str, optional
        Filename prefix to select, e.g. ``"PL-dual-gate-sweep_iter_"``.  Omitted,
        every ``*.csv`` in the folder is considered.
    stacklevel : int
        Passed straight to the ordering helper, whose warnings about a missing or
        repeated ``_iter_N`` should point at the caller's own call.

    Returns
    -------
    (frames, skipped) : (list of Path, dict)
        Ordered image paths, and ``{Path: kind}`` for everything excluded.
    """
    pattern = f"{prefix}*.csv" if prefix else "*.csv"
    kinds   = {f: _classify_csv(f) for f in sorted(Path(directory).glob(pattern))}

    images  = [f for f, kind in kinds.items() if kind == "image"]
    skipped = {f: kind for f, kind in kinds.items() if kind != "image"}
    if not images:
        return [], skipped
    return _order_by_iter(images, directory, stacklevel=stacklevel), skipped


def _describe(skipped: dict) -> str:
    """One indented ``name — why`` line per entry of a ``{Path: kind}`` map."""
    return "\n".join(
        f"  {f.name} — {_CSV_KIND_REASON.get(kind, kind)}"
        for f, kind in skipped.items()
    )


def _require_image(path: Path) -> None:
    """
    Raise unless *path* is a real-space image CSV.

    Raises
    ------
    ValueError
        Naming what the file is instead, in the same words the loaders use, so a
        spectrum is never silently written out as a two-pixel-tall image.
    """
    kind = _classify_csv(path)
    if kind != "image":
        raise ValueError(
            f"'{path}' is not a real-space image CSV: "
            f"{_CSV_KIND_REASON.get(kind, kind)}."
        )


# ---------------------------------------------------------------------------
# Pixel type
# ---------------------------------------------------------------------------

def _as_image_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    """
    Cast an image array for TIFF output.

    ``dtype="auto"`` keeps integer counts exactly — ``uint16`` up to 65535 and
    ``uint32`` above it — and falls back to ``float32`` for anything else.  That
    fallback is a **narrowing**: the CSV is read as ``float64``, so a non-integer
    or negative image loses precision.  Force ``"float32"`` knowingly, or keep the
    CSV.

    Parameters
    ----------
    arr : np.ndarray
    dtype : {"auto", "uint16", "uint32", "float32"}

    Returns
    -------
    np.ndarray
        A new array of the chosen type.

    Raises
    ------
    ValueError
        If *dtype* is not one of the four.
    """
    if dtype == "auto":
        finite = arr[np.isfinite(arr)]
        # An all-NaN frame has no integers to preserve, so it takes the float
        # branch rather than casting NaN to an unsigned type.
        is_count = bool(finite.size) and bool(
            np.all(finite >= 0) and np.all(finite == np.round(finite))
        )
        if is_count:
            return arr.astype(np.uint16 if finite.max() <= _U16_MAX else np.uint32)
        return arr.astype(np.float32)
    if dtype in _DTYPES[1:]:
        return arr.astype(dtype)
    raise ValueError(
        f"Unsupported image dtype '{dtype}'. Choose from: {list(_DTYPES)}."
    )


# ---------------------------------------------------------------------------
# Image CSV -> TIFF
# ---------------------------------------------------------------------------

def convert_image_csv_to_tiff(
    path,
    out       = None,
    dtype     : str  = "auto",
    overwrite : bool = False,
) -> Path:
    """
    Convert a single real-space image CSV to a TIFF file.

    Parameters
    ----------
    path : str or Path
        Source image CSV.  A file that is not a numeric grid of at least three
        rows is refused rather than converted.
    out : str or Path, optional
        Destination file or directory.  Omitted, the file lands in a
        ``processed/`` folder — see the module docstring.
    dtype : {"auto", "uint16", "uint32", "float32"}
        Pixel type.  ``"auto"`` keeps integer counts exactly.
    overwrite : bool
        Replace an existing output file.  Default False, which raises.

    Returns
    -------
    pathlib.Path
        The ``.tif`` written.

    Raises
    ------
    ValueError
        If *path* is not a real-space image CSV, or *dtype* is unknown.
    FileExistsError
        If the output exists and *overwrite* is False.
    """
    path = Path(path)
    _require_image(path)
    target = _claim_target(_default_output(path, ".tif", out), overwrite)
    tifffile.imwrite(target, _as_image_dtype(np.loadtxt(path, delimiter=","), dtype))
    return target


def convert_image_dir_to_tiff_stack(
    directory,
    prefix    = None,
    out       = None,
    dtype     : str  = "auto",
    overwrite : bool = False,
) -> Path:
    """
    Combine a directory of real-space image CSVs into one multi-page TIFF.

    Frames are ordered by their ``_iter_N`` suffix, so page *i* of the stack is
    acquisition *i*.  A gap in that sequence is warned about and never closed up,
    so a stack built from an incomplete export has fewer pages than iterations.
    One pixel type is chosen for the whole stack.

    Parameters
    ----------
    directory : str or Path
        Folder holding the frames.
    prefix : str, optional
        Filename prefix to select, e.g. ``"PL-dual-gate-sweep_iter_"``.  Omitted,
        every image CSV in the folder is used.
    out : str or Path, optional
        Destination file or directory.  Omitted, the stack lands in the same
        ``processed/`` folder its frames would have, named after *prefix* with
        trailing separators stripped, or after the folder.
    dtype : {"auto", "uint16", "uint32", "float32"}
        Pixel type for every page.
    overwrite : bool
        Replace an existing output file.  Default False, which raises.

    Returns
    -------
    pathlib.Path
        The multi-page ``.tif`` written.

    Raises
    ------
    ValueError
        If the folder holds no image CSV matching *prefix*.  The message names
        what was found instead.
    FileExistsError
        If the output exists and *overwrite* is False.
    """
    directory = Path(directory)
    # 4 frames out to the caller's line: the ordering helper, _image_frames, this
    # function, the call.  A run arriving through convert_path lands one frame
    # short, on convert_path's own line.
    frames, skipped = _image_frames(directory, prefix, stacklevel=4)
    if not frames:
        pattern = f"{prefix}*.csv" if prefix else "*.csv"
        if skipped:
            raise ValueError(
                f"No real-space image CSV matching '{pattern}' in '{directory}'. "
                f"Found {len(skipped)} other file(s):\n{_describe(skipped)}"
            )
        raise ValueError(f"No CSV files matching '{pattern}' in '{directory}'.")

    stem = (prefix.rstrip("_- ") if prefix else directory.name) + "_stack"
    if out is None:
        # The frames' own processed/ folder, so a stack sits beside the per-frame
        # output rather than in a second place.
        target = _default_output(frames[0], ".tif").parent / (stem + ".tif")
    elif Path(out).suffix == "" or Path(out).is_dir():
        target = Path(out) / (stem + ".tif")
    else:
        target = Path(out)

    target = _claim_target(target, overwrite)
    # (n_frames, ny, nx) — every frame read once, cast as one block so the whole
    # stack shares a pixel type.
    stack = np.stack([np.loadtxt(f, delimiter=",") for f in frames])
    # photometric is not decoration: left to guess, tifffile reads a 3-frame
    # stack as one RGB image with separate colour planes, and a 3- or 4-point
    # sweep would open as a single colour frame instead of its pages.
    tifffile.imwrite(target, _as_image_dtype(stack, dtype), photometric="minisblack")
    return target


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def convert_path(
    path,
    out       = None,
    recursive : bool = False,
    stack     : bool = False,
    dtype     : str  = "auto",
    overwrite : bool = False,
) -> ConversionReport:
    """
    Convert a file, a directory, or a tree, continuing past failures.

    Every failure is collected rather than raised, so one unreadable frame does
    not abandon a sweep.  Call :func:`convert_image_csv_to_tiff` directly to have
    the exception instead.

    Parameters
    ----------
    path : str or Path
        A single CSV, or a directory of them.
    out : str or Path, optional
        Destination.  Omitted, each source folder gets its own ``processed/``
        folder, which is what keeps two sweeps holding an identically named frame
        from overwriting one another.
    recursive : bool
        Descend into subdirectories.  Ignored when *path* is a file.
    stack : bool
        Write one multi-page TIFF per folder instead of one file per frame.
    dtype : {"auto", "uint16", "uint32", "float32"}
        Pixel type.
    overwrite : bool
        Replace existing output files.

    Returns
    -------
    ConversionReport
        ``(outputs, skipped, errors)``.

    Raises
    ------
    FileNotFoundError
        If *path* is neither a file nor a directory.
    """
    path = Path(path)
    outputs, skipped, errors = [], {}, []

    if path.is_file():
        try:
            outputs.append(
                convert_image_csv_to_tiff(
                    path, out=out, dtype=dtype, overwrite=overwrite
                )
            )
        except Exception as exc:              # a batch reports and carries on
            errors.append((path, str(exc)))
        return ConversionReport(outputs, skipped, errors)

    if not path.is_dir():
        raise FileNotFoundError(f"No such file or directory: '{path}'")

    # Every folder holding a CSV, deduplicated: a stack is built per folder, and
    # an omitted out= resolves per folder too.  Which of those CSVs are frames is
    # settled inside the loop, by content.
    csvs = path.rglob("*.csv") if recursive else path.glob("*.csv")

    for folder in sorted({csv.parent for csv in csvs}):
        # 3 frames out from the ordering helper: itself, _image_frames, this loop.
        frames, folder_skipped = _image_frames(folder, stacklevel=3)
        skipped.update(folder_skipped)
        if not frames:
            continue

        if stack:
            try:
                outputs.append(
                    convert_image_dir_to_tiff_stack(
                        folder, out=out, dtype=dtype, overwrite=overwrite
                    )
                )
            except Exception as exc:
                errors.append((folder, str(exc)))
            continue

        for frame in frames:
            try:
                outputs.append(
                    convert_image_csv_to_tiff(
                        frame, out=out, dtype=dtype, overwrite=overwrite
                    )
                )
            except Exception as exc:
                errors.append((frame, str(exc)))

    return ConversionReport(outputs, skipped, errors)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """
    Entry point for the ``tmdc-convert`` command.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse.  ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit status: 1 if any conversion failed, 0 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="tmdc-convert",
        description=(
            "Convert AttoCube real-space image CSV exports to TIFF. Output lands "
            "in a processed/ folder: the sibling of raw/ when the source is in "
            "one, and a folder beside the source otherwise."
        ),
    )
    parser.add_argument("path", help="Image CSV, or a directory of them.")
    parser.add_argument(
        "--out", default=None,
        help="Destination directory (or file, for a single input). "
             "Default: a processed/ folder per source folder.",
    )
    parser.add_argument(
        "--stack", action="store_true",
        help="Write one multi-page TIFF per folder, in _iter_N order, instead of "
             "one file per frame.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Descend into subdirectories when PATH is a directory.",
    )
    parser.add_argument(
        "--dtype", default="auto", choices=_DTYPES,
        help="Pixel type. Default 'auto': integer counts kept exactly, "
             "anything else narrowed to float32.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing output files. Default: refuse.",
    )
    args = parser.parse_args(argv)

    try:
        report = convert_path(
            args.path,
            out       = args.out,
            recursive = args.recursive,
            stack     = args.stack,
            dtype     = args.dtype,
            overwrite = args.overwrite,
        )
    except FileNotFoundError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1

    for target in report.outputs:
        print(f"wrote {target}")
    for src, message in report.errors:
        print(f"ERROR {src}: {message}", file=sys.stderr)

    summary = f"\n{len(report.outputs)} file(s) written"
    if report.skipped:
        summary += f", {len(report.skipped)} not image CSV(s)"
    summary += f", {len(report.errors)} error(s)."
    print(summary)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
