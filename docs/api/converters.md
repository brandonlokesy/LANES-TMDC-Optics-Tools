# Converters

AttoCube real-space image CSV exports to TIFF.

A frame leaves the AttoCube software as a bare grid of comma-separated numbers
with no header, and a sweep leaves as a directory of them indexed by an
`_iter_N` suffix. Text costs roughly six bytes per count, so the export is an
order of magnitude larger than the same data as TIFF, and no image viewer will
open it.

Two shapes of output. One `.tif` per frame keeps the input one-to-one; a single
multi-page `.tif` is the more convenient object to hand to ImageJ. Combining
frames asserts that they belong together, so the stack is asked for rather than
assumed.

## Where output lands

A converted file goes into a `processed/` folder:

| Source | Output |
|---|---|
| `measurement/raw/frame_iter_0.csv` | `measurement/processed/frame_iter_0.tif` |
| `scratch/frame_iter_0.csv` | `scratch/processed/frame_iter_0.tif` |

The folder names are fixed and take no parameter. `out=` is the single override,
and names either a directory or a file.

A directory run resolves this **per source folder**, so two sweeps holding an
identically named frame keep their own output rather than one overwriting the
other.

!!! note "An existing output is refused"
    Both writers take `overwrite=False` by default and raise `FileExistsError`,
    matching [`write_sweep`](hdf5.md). The file being replaced may be the only
    copy, so replacing it is asked for.

## From the command line

Installing the package provides `tmdc-convert`:

```
tmdc-convert scan/raw/
  -> scan/processed/frame_iter_0.tif, frame_iter_1.tif, ...

tmdc-convert scan/raw/ --stack
  -> scan/processed/raw_stack.tif        (multi-page, in _iter_N order)
```

`--recursive` descends into subdirectories, `--dtype` fixes the pixel type, and
`--overwrite` replaces existing files. A batch continues past a failure and
reports what it could not convert, returning a non-zero exit status.

## Which files are converted

Selection is by content, not by filename. A parameter export written beside the
frames, and a two-row single spectrum, are both excluded — a spectrum is numeric
on its first line exactly like an image, so the row count is what separates them.
Anything excluded is named in the report rather than passed through.

Frames are ordered by the integer in their `_iter_N` suffix, so `iter_10` follows
`iter_2`. Export padding widths vary between acquisitions and cannot be relied
on. A gap in the sequence is warned about and never closed up.

::: tmdc_optics_tools.converters
