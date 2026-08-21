# Converters

AttoCube CSV exports to compact formats: real-space images to TIFF, spectral and
TRPL sweeps to HDF5.

An export leaves the acquisition software as text. A frame is a bare grid of
comma-separated numbers with no header; a spectral sweep is a wide grid of
`[Par, Wavelength, ExpROI1, ExpROI2]` blocks; a TRPL sweep is a whole *directory*,
one decay per file. Text costs roughly six bytes per count, so every one of them is
far larger than the same data in a binary format, and no viewer will open a frame.

Measured on the committed example data:

| Source | As exported | Converted |
|---|---|---|
| `position-scan/PL`, 46 frames | 96.49 MB | 24.12 MB (one stack) |
| `stark-shift` spectral sweep | 4.59 MB | 0.142 MB |
| `TRPL`, 3 decays + companion | 11.57 MB | 0.070 MB |

Nothing here interprets a measurement. No correction is applied, no axis is
derived, and every question a loader already answers is left to it.

## Where output lands

A converted file goes into a `processed/` folder:

| Source | Output |
|---|---|
| `measurement/raw/frame_iter_0.csv` | `measurement/processed/frame_iter_0.tif` |
| `measurement/raw/sweep.csv` | `measurement/processed/sweep.h5` |
| `scratch/frame_iter_0.csv` | `scratch/processed/frame_iter_0.tif` |

The folder names are fixed and take no parameter. `out=` is the single override,
and names either a directory or a file.

A directory run resolves this **per source folder**, so two sweeps holding an
identically named frame keep their own output rather than one overwriting the
other.

!!! note "An existing output is refused"
    Every writer here takes `overwrite=False` and raises `FileExistsError`,
    matching [`write_sweep`](hdf5.md). The file being replaced may be the only
    copy, so replacing it is asked for. For an HDF5 output the check runs *before*
    the decode, so a re-run over a folder of 12 MB exports refuses immediately
    rather than parsing each one first.

## HDF5: one archive format

The spectral and TRPL converters own no format. They load with
`AttoCubeSpectralSweep` or `AttoCubeTRPLSweep` and write with
[`write_sweep`](hdf5.md), so a converted sweep reopens by handing the `.h5` straight
back to its loader.

### Only the measurement *type* is declared at conversion time

A raw spectral export records no measurement type and none can be inferred from the
data, so that one fact has to be supplied. Nothing else does, because the loader
takes `sweep`, `gates` and `geometry` from its arguments and only falls back to
what the file stored:

```python
tmdc-convert scan/raw --spectra-type PL
```

```python
# The archive holds every array and parameter row. The physics is declared when
# you open it, exactly as it is for the CSV.
scan = AttoCubeSpectralSweep(
    "scan/processed/sweep.h5", spectra_type="PL",
    sweep="electric_field",
    gates={"top": "V_A", "bottom": "V_B"},
    geometry=geom,
)
```

`AttoCubeTRPLSweep` needs no type at all — it defaults to `"TRPL"`, the class name
having already declared the modality.

### A TRPL directory converts only when you name it

A TRPL sweep is a directory, so converting one means deciding which files belong to
one measurement. The loader settles most of that on its own: an IRF reference is
excluded by name, and a spectral-header file in a TRPL folder is read as the
sweep's parameter-table companion rather than as a sweep.

What it cannot settle is how many measurements are in the folder. Two — say
`right1_*` and `right2_*` — merge into one archive, and the warning about their
colliding iteration indices would be buried in a long run.

So: the directory you **name** converts; a directory reached by `--recursive` is
reported as deferred and named on stdout. Pass `--prefix` to convert one
measurement at a time.

```
tmdc-convert data/trpl-sweep                    # converts
tmdc-convert data --recursive                   # defers, and says which folder
tmdc-convert data/right_spots --prefix right1_  # converts just right1
```

## From the command line

Installing the package provides `tmdc-convert`:

```
tmdc-convert PATH [--out DIR] [--spectra-type {A,PL,R,RC,T,TRPL}] [--prefix P]
             [--stack] [--recursive] [--dtype auto|uint16|uint32|float32]
             [--compression gzip|none] [--overwrite]
```

A batch continues past a failure and reports what it could not convert, returning a
non-zero exit status if anything failed. One folder can produce both kinds of
output at once — `examples/data/stark-shift` holds 14 frames beside a spectral
export, and a single command writes 14 TIFFs and one `.h5`.

Asking for no `--spectra-type` in a folder that holds a spectral export is not
fatal: the frames still convert, and the export is reported as an error carrying the
loader's own message, which names the flag and lists the valid types.

## Which files are converted

Selection is by content, not by filename. A two-row single spectrum is excluded — it
is numeric on its first line exactly like an image, so the row count is what
separates them. Anything excluded is named in the report rather than passed through.

Frames destined for a **stack** are ordered by the integer in their `_iter_N`
suffix, so `iter_10` follows `iter_2`; export padding widths vary between
acquisitions and cannot be relied on. A gap in that sequence is warned about and
never closed up. Per-frame conversion needs no order, each output being named after
its own input.

::: tmdc_optics_tools.converters
