# tmdc_optics_tools/converters.py
"""
Convert AttoCube confocal CSV exports to compact, self-describing formats.

The AttoCube software exports two kinds of CSV, both inefficient text encodings
of what is fundamentally binary data:

* **Spectral gate sweeps** — a wide grid where every sweep point occupies a
  ``[Par, Wavelength, ExpROI1, ExpROI2]`` block.  The wavelength axis is repeated
  once per sweep, ``ExpROI2`` is often all zeros, and ~58 labeled scalar
  parameters (``V_A``, ``V_B``, ``Excitation Power``, ``T`` …) are overlaid on the
  first rows via the ``Par`` column.  These are converted to **HDF5**.
* **Real-space images** — a clean ``H x W`` numeric raster with no header.  A
  directory typically holds a gate sweep of these frames.  These are converted to
  **TIFF** (one file each, or a single multi-page stack per directory).

The two CSV types are told apart by :func:`is_image_csv` (the first line of an
image CSV parses as floats; a spectral file begins with a text header).

Public functions
----------------
is_image_csv
    Detect whether a CSV holds a numeric image (vs a spectral scan).
parse_spectral_csv
    Decode a spectral CSV into wavelength / ROI spectra / parameter arrays.
convert_spectral_csv_to_hdf5
    Spectral CSV -> ``.h5`` (lossless, compressed).
convert_image_csv_to_tiff
    Single image CSV -> ``.tif``.
convert_image_dir_to_tiff_stack
    A directory of image CSVs -> one multi-page ``.tif``.
convert_csv
    Auto-route a single CSV to HDF5 or TIFF.
convert_path
    Convert a file, directory, or tree in one call.
main
    ``tmdc-convert`` command-line entry point.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__

# ---------------------------------------------------------------------------
# Detection and output-path helpers
# ---------------------------------------------------------------------------


def is_image_csv(path) -> bool:
    """
    Return ``True`` if *path* is a numeric-grid image CSV.

    The test parses the first comma-separated value of the first line as a
    float.  Spectral scan files begin with a text header (``"Parameters
    Labels"``) and therefore return ``False``.
    """
    try:
        with open(path, "r") as fh:
            first_line = fh.readline()
        float(first_line.strip().split(",")[0])
        return True
    except (ValueError, OSError, IndexError):
        return False


def _default_output(src: Path, new_suffix: str, out=None) -> Path:
    """
    Resolve the output path for converting *src* to *new_suffix*.

    * If *out* is a directory (existing, or with no suffix), the file is placed
      inside it with ``src.stem + new_suffix``.
    * If *out* is a file path (has a suffix), it is used verbatim.
    * If *out* is ``None`` and *src* lives in a ``raw/`` folder, the file is
      written to the sibling ``processed/`` folder; otherwise it is written
      alongside *src*.

    The parent directory is created if necessary.
    """
    src = Path(src)
    if out is not None:
        out = Path(out)
        if out.suffix == "" or out.is_dir():
            target = out / (src.stem + new_suffix)
        else:
            target = out
    elif src.parent.name.lower() == "raw":
        target = src.parent.parent / "processed" / (src.stem + new_suffix)
    else:
        target = src.with_suffix(new_suffix)

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# Spectral CSV -> HDF5
# ---------------------------------------------------------------------------

# Column offsets within each 4-column sweep block, matching the layout decoded
# by loaders.AttoCubePLVabScan.
_BLOCK = 4
_OFF_PAR, _OFF_WL, _OFF_ROI1, _OFF_ROI2 = 0, 1, 2, 3


def parse_spectral_csv(path) -> dict:
    """
    Decode a spectral gate-sweep CSV into plain NumPy arrays.

    Reuses the block-stride layout used by
    :class:`tmdc_optics_tools.loaders.AttoCubePLVabScan`: padding columns
    (all NaN) are stripped, the remaining columns must be a multiple of four,
    and each block is ``[Par, Wavelength, ExpROI1, ExpROI2]``.

    Returns
    -------
    dict with keys:
        ``wavelength`` : (n_pixels,) float64 -- wavelength axis in nm.
        ``roi1``, ``roi2`` : (n_pixels, n_sweeps) float64 -- raw ROI counts.
        ``parameters`` : dict[str, np.ndarray] -- one (n_sweeps,) array per
            labeled row (e.g. ``"V_A"``), taken from the ``Par`` column.
        ``n_pixels``, ``n_sweeps`` : int.
    """
    raw = pd.read_csv(path, header=0, index_col=0, low_memory=False)
    row_labels = [("" if pd.isna(lbl) else str(lbl)) for lbl in raw.index]

    d = raw.to_numpy(dtype=float)
    d = d[:, ~np.all(np.isnan(d), axis=0)]          # strip all-NaN padding cols

    n_cols = d.shape[1]
    if n_cols % _BLOCK != 0:
        raise ValueError(
            f"After stripping padding, got {n_cols} columns which is not "
            f"divisible by {_BLOCK}. Unexpected spectral CSV layout in '{path}'."
        )

    par_cols  = np.arange(_OFF_PAR,  n_cols, _BLOCK)
    wl_cols   = np.arange(_OFF_WL,   n_cols, _BLOCK)
    roi1_cols = np.arange(_OFF_ROI1, n_cols, _BLOCK)
    roi2_cols = np.arange(_OFF_ROI2, n_cols, _BLOCK)

    wl_raw   = d[:, wl_cols[0]]
    valid_px = np.isfinite(wl_raw)

    wavelength = wl_raw[valid_px]
    roi1 = d[valid_px][:, roi1_cols]
    roi2 = d[valid_px][:, roi2_cols]

    parameters = {
        label: d[i, par_cols]
        for i, label in enumerate(row_labels)
        if label != ""
    }

    return {
        "wavelength": wavelength,
        "roi1": roi1,
        "roi2": roi2,
        "parameters": parameters,
        "n_pixels": int(wavelength.size),
        "n_sweeps": int(par_cols.size),
    }


def _hdf5_key(label: str) -> str:
    """Sanitise a parameter label into an HDF5-safe dataset name."""
    return label.replace("/", "_").strip()


def _counts_dtype(arr: np.ndarray):
    """int32 if the array holds non-negative integers, else the input float."""
    finite = arr[np.isfinite(arr)]
    if finite.size and np.all(finite >= 0) and np.all(finite == np.round(finite)):
        return np.int32
    return arr.dtype


def convert_spectral_csv_to_hdf5(path, out=None, compression="gzip") -> Path:
    """
    Convert a spectral gate-sweep CSV to a compact, self-describing HDF5 file.

    Layout
    ------
    ``/wavelength_nm``        (n_pixels,)           float64
    ``/spectra/ExpROI1``      (n_pixels, n_sweeps)  int32, compressed
    ``/spectra/ExpROI2``      (n_pixels, n_sweeps)  int32, compressed
    ``/parameters/<key>``     (n_sweeps,)           float64 (one per labeled row,
                              with the original label kept as a ``label`` attr)

    Root attributes record the source filename, array shapes, the conversion
    timestamp, and the toolkit version for provenance.

    Parameters
    ----------
    path : str or Path
        Source spectral CSV.
    out : str or Path, optional
        Output file or directory.  Defaults to the sibling ``processed/`` folder
        when *path* is in a ``raw/`` folder (see :func:`_default_output`).
    compression : str or None
        h5py compression filter for the spectra (default ``"gzip"``).
        ``None`` disables compression.

    Returns
    -------
    Path to the written ``.h5`` file.
    """
    import h5py

    path = Path(path)
    data = parse_spectral_csv(path)
    target = _default_output(path, ".h5", out)

    comp = None if compression in (None, "none", "None") else compression
    ckw = {"compression": comp} if comp is not None else {}

    with h5py.File(target, "w") as f:
        f.create_dataset("wavelength_nm", data=data["wavelength"].astype(np.float64))

        spectra = f.create_group("spectra")
        for name, arr in (("ExpROI1", data["roi1"]), ("ExpROI2", data["roi2"])):
            spectra.create_dataset(
                name, data=arr.astype(_counts_dtype(arr)), **ckw
            )

        params = f.create_group("parameters")
        for label, arr in data["parameters"].items():
            ds = params.create_dataset(_hdf5_key(label), data=arr.astype(np.float64))
            ds.attrs["label"] = label

        f.attrs["source_filename"] = path.name
        f.attrs["n_pixels"] = data["n_pixels"]
        f.attrs["n_sweeps"] = data["n_sweeps"]
        f.attrs["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        f.attrs["tool_version"] = __version__
        f.attrs["converter"] = "tmdc_optics_tools"

    return target


# ---------------------------------------------------------------------------
# Image CSV -> TIFF
# ---------------------------------------------------------------------------


def _as_image_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    """
    Cast an image array losslessly for TIFF output.

    ``dtype="auto"`` chooses ``uint16`` when every value is an integer within
    ``[0, 65535]`` (the natural raster encoding for these counts) and
    ``float32`` otherwise.  ``"uint16"`` / ``"float32"`` force a specific dtype.
    """
    if dtype == "auto":
        finite = arr[np.isfinite(arr)]
        is_int = finite.size and np.all(finite == np.round(finite))
        in_u16 = is_int and finite.min() >= 0 and finite.max() <= 65535
        return arr.astype(np.uint16 if in_u16 else np.float32)
    if dtype in ("uint16", "float32"):
        return arr.astype(dtype)
    raise ValueError(f"Unsupported image dtype '{dtype}'. Use auto/uint16/float32.")


def convert_image_csv_to_tiff(path, out=None, dtype="auto") -> Path:
    """
    Convert a single numeric-grid image CSV to a TIFF file.

    Parameters
    ----------
    path : str or Path
        Source image CSV.
    out : str or Path, optional
        Output file or directory (see :func:`_default_output`).
    dtype : {"auto", "uint16", "float32"}
        Pixel type for the TIFF.  ``"auto"`` is lossless (see
        :func:`_as_image_dtype`).

    Returns
    -------
    Path to the written ``.tif`` file.
    """
    import tifffile

    path = Path(path)
    img = np.loadtxt(path, delimiter=",")
    target = _default_output(path, ".tif", out)
    tifffile.imwrite(target, _as_image_dtype(img, dtype))
    return target


def convert_image_dir_to_tiff_stack(
    directory, prefix=None, out=None, dtype="auto"
) -> Path:
    """
    Combine a directory of image CSVs into one multi-page (3-D) TIFF.

    Parameters
    ----------
    directory : str or Path
        Folder containing the image CSV frames.
    prefix : str, optional
        Filename prefix to select (e.g. ``"PL-dual-gate-sweep_iter_"``).  When
        omitted, every ``*.csv`` in the folder that passes :func:`is_image_csv`
        is used.  Frames are sorted by filename.
    out : str or Path, optional
        Output file or directory.  When a directory (or ``None``), the stack is
        named after *prefix* (trailing separators stripped) or the folder.
    dtype : {"auto", "uint16", "float32"}
        Pixel type for the stack.

    Returns
    -------
    Path to the written multi-page ``.tif`` file.
    """
    import tifffile

    directory = Path(directory)
    pattern = f"{prefix}*.csv" if prefix else "*.csv"
    files = sorted(p for p in directory.glob(pattern) if is_image_csv(p))
    if not files:
        raise ValueError(
            f"No image CSV frames matching '{pattern}' found in '{directory}'."
        )

    stack = np.stack([np.loadtxt(f, delimiter=",") for f in files])

    stem = (prefix.rstrip("_- ") if prefix else directory.name) + "_stack"
    if out is None:
        # Mirror a raw/ frame's processed/ destination, named after the stack.
        sample = _default_output(files[0], ".tif")
        target = sample.parent / (stem + ".tif")
    elif Path(out).suffix == "" or Path(out).is_dir():
        target = Path(out) / (stem + ".tif")
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(target, _as_image_dtype(stack, dtype))
    return target


# ---------------------------------------------------------------------------
# Routing and batch conversion
# ---------------------------------------------------------------------------


def convert_csv(path, out=None, dtype="auto", compression="gzip") -> Path:
    """
    Convert a single CSV, auto-routing by content.

    Image CSVs become TIFF; spectral CSVs become HDF5.
    """
    path = Path(path)
    if is_image_csv(path):
        return convert_image_csv_to_tiff(path, out=out, dtype=dtype)
    return convert_spectral_csv_to_hdf5(path, out=out, compression=compression)


def convert_path(
    path,
    out=None,
    recursive=False,
    stack_images=False,
    dtype="auto",
    compression="gzip",
):
    """
    Convert a file, directory, or tree.

    Parameters
    ----------
    path : str or Path
        A single CSV, or a directory of CSVs.
    out : str or Path, optional
        Output directory (or file, only meaningful for a single-file *path*).
    recursive : bool
        When *path* is a directory, also descend into subdirectories.
    stack_images : bool
        When *path* is a directory, combine its image CSVs into one multi-page
        TIFF per directory instead of one TIFF per frame.  Spectral CSVs are
        still converted individually.
    dtype, compression
        Forwarded to the image / spectral converters.

    Returns
    -------
    (outputs, errors) : (list[Path], list[tuple[Path, str]])
        Successful output paths and ``(source, message)`` pairs for any files
        that failed (the batch continues past errors).
    """
    path = Path(path)
    outputs: list[Path] = []
    errors: list[tuple[Path, str]] = []

    if path.is_file():
        outputs.append(convert_csv(path, out=out, dtype=dtype, compression=compression))
        return outputs, errors

    if not path.is_dir():
        raise FileNotFoundError(f"No such file or directory: '{path}'")

    # Group CSVs by directory so image stacks are built per folder.
    csvs = sorted(path.rglob("*.csv") if recursive else path.glob("*.csv"))
    by_dir: dict[Path, list[Path]] = {}
    for c in csvs:
        by_dir.setdefault(c.parent, []).append(c)

    for folder, items in by_dir.items():
        images = [c for c in items if is_image_csv(c)]
        spectra = [c for c in items if c not in images]

        if stack_images and images:
            try:
                outputs.append(
                    convert_image_dir_to_tiff_stack(folder, out=out, dtype=dtype)
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                errors.append((folder, str(exc)))
        else:
            spectra = items if not stack_images else spectra
            for c in images if not stack_images else []:
                try:
                    outputs.append(convert_image_csv_to_tiff(c, out=out, dtype=dtype))
                except Exception as exc:  # noqa: BLE001
                    errors.append((c, str(exc)))

        for c in spectra:
            if is_image_csv(c):
                continue  # already handled above when not stacking
            try:
                outputs.append(
                    convert_spectral_csv_to_hdf5(c, out=out, compression=compression)
                )
            except Exception as exc:  # noqa: BLE001
                errors.append((c, str(exc)))

    return outputs, errors


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """Entry point for the ``tmdc-convert`` command."""
    parser = argparse.ArgumentParser(
        prog="tmdc-convert",
        description=(
            "Convert AttoCube CSV exports to compact formats: spectral scans -> "
            "HDF5, images -> TIFF.  Converted files default to the measurement's "
            "processed/ folder."
        ),
    )
    parser.add_argument("path", help="CSV file, or directory of CSV files.")
    parser.add_argument(
        "--out", default=None,
        help="Output directory (or file for a single input). "
             "Default: sibling processed/ folder.",
    )
    parser.add_argument(
        "--stack", action="store_true",
        help="Combine image CSVs in each directory into one multi-page TIFF.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recurse into subdirectories when PATH is a directory.",
    )
    parser.add_argument(
        "--dtype", default="auto", choices=("auto", "uint16", "float32"),
        help="Pixel type for TIFF output (default: auto, lossless).",
    )
    parser.add_argument(
        "--compression", default="gzip",
        help="HDF5 spectra compression filter, or 'none' (default: gzip).",
    )
    args = parser.parse_args(argv)

    outputs, errors = convert_path(
        args.path,
        out=args.out,
        recursive=args.recursive,
        stack_images=args.stack,
        dtype=args.dtype,
        compression=args.compression,
    )

    for o in outputs:
        print(f"wrote {o}")
    for src, msg in errors:
        print(f"ERROR {src}: {msg}", file=sys.stderr)

    print(f"\n{len(outputs)} file(s) written, {len(errors)} error(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
