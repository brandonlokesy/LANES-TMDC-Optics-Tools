# Converters

Turn AttoCube CSV exports into formats that are smaller and that other tools can
open. Images become TIFF; spectral and TRPL sweeps become HDF5.

An export leaves the acquisition software as text. A frame is a bare grid of
comma-separated numbers with no header; a spectral sweep is a wide grid of
`[Par, Wavelength, ExpROI1, ExpROI2]` blocks; a TRPL sweep is a whole *directory*,
one decay per file. Text costs roughly six bytes per count, so every one of them is
far larger than the same data in a binary format, and no viewer will open a frame.

Nothing here interprets a measurement. No correction is applied, no axis is
derived, and every question a loader already answers is left to it. The output
holds the same numbers the export held.

Measured on the committed example data:

| Source | As exported | Converted |
|---|---|---|
| `position-scan/PL`, 46 frames | 96.49 MB | 24.12 MB (one stack) |
| `stark-shift` spectral sweep | 4.59 MB | 0.142 MB |
| `TRPL`, 3 decays + companion | 11.57 MB | 0.070 MB |

## Quick start

### One image frame

```python
from tmdc_optics_tools import converters

converters.convert_image_csv_to_tiff("scan/raw/frame_iter_0.csv")
# -> scan/converted/frame_iter_0.tif
```

### A folder of frames, as one stack

A gate or position sweep of frames is a sequence, so it can go into a single
multi-page TIFF — the object ImageJ opens most naturally.

```python
converters.convert_image_dir_to_tiff_stack("scan/raw", prefix="PL-dual-gate-sweep_")
# -> scan/converted/PL-dual-gate-sweep_stack.tif
```

Pages are ordered by the `_iter_N` suffix, so `iter_10` follows `iter_2`. Without
`--stack` (or this function), each frame becomes its own `.tif` instead.

### A spectral sweep

```python
converters.convert_spectral_csv_to_hdf5("scan/raw/sweep.csv", "PL")
# -> scan/converted/sweep.h5
```

The measurement type is required, and is the only measurement fact asked for. A raw
export records none, and it cannot be inferred from the numbers.

### A TRPL sweep

A TRPL sweep is a whole **directory** — one decay per file — so converting it
collapses many files into one archive.

```python
converters.convert_trpl_dir_to_hdf5("trpl-sweep", prefix="TRPL_")
# -> converted/trpl.h5
```

No type argument: `AttoCubeTRPLSweep` defaults it to `"TRPL"`, the class name
having already declared the modality.

### A file, a folder, or a whole tree

```python
report = converters.convert_path("data", recursive=True, spectra_type="PL")
outputs, skipped, errors = report
```

Failures are collected rather than raised, so one unreadable frame does not abandon
a sweep. Call a single-purpose converter directly to get the exception instead.

## Reading an archive back

The HDF5 side owns no format. It loads with `AttoCubeSpectralSweep` or
`AttoCubeTRPLSweep` and writes with [`write_sweep`](hdf5.md), so an archive reopens
by handing the `.h5` straight back to its loader.

Only the measurement *type* is stored at conversion time. What was swept, which
channel reached which gate, and the device stack are declared when you **open** the
archive — the loader takes each from its argument and only falls back to what the
file recorded:

```python
scan = AttoCubeSpectralSweep(
    "scan/converted/sweep.h5", spectra_type="PL",
    sweep="electric_field",
    gates={"top": "V_A", "bottom": "V_B"},
    geometry=geom,
)
```

So an archive is declared against exactly as the CSV was, and nothing is lost by
converting before you have decided how to analyse.

## Where output lands

Output goes into a **`converted/`** folder, created if it is not already there.
Which `converted/` depends on **the folder you name in the command** — nothing is
searched for, so you can work the answer out from what you typed.

!!! note "`converted/`, not `processed/`"
    A conversion is not an analysis. `processed/` is for what an analysis pulls
    *out* of the raw data — fitted positions, integrated intensities, diffusion
    lengths — which took a decision to produce and may not be reproducible.
    Everything in `converted/` is the raw data in a better container, so it can be
    deleted and regenerated at any time. Which folder a file is in tells you which
    kind it is.

### If you keep a `raw/` folder

Point at `raw/` and its **sibling** `converted/` is used, with your folder
structure copied underneath:

```
EXP/
├── converted/          ← created for you
└── raw/
    ├── spot01/{01-PL-Vbot/, ref/}
    └── spot02/{01-PL-Vbot/, ref/}
```

```
tmdc-convert EXP/raw --recursive --spectra-type PL

EXP/raw/spot01/01-PL-Vbot/sweep.csv  ->  EXP/converted/spot01/01-PL-Vbot/sweep.h5
EXP/raw/spot01/ref/laser_ref.csv     ->  EXP/converted/spot01/ref/laser_ref.tif
EXP/raw/spot02/ref/laser_ref.csv     ->  EXP/converted/spot02/ref/laser_ref.tif
```

Your structure is copied, not flattened, so two spot folders can hold the same
`laser_ref_*.csv` without one overwriting the other. `RAW/` and `Raw/` count too.

### If you do not

Nothing changes for you. A `converted/` folder appears next to your files:

```
tmdc-convert Downloads/mydata --spectra-type PL

Downloads/mydata/sweep.csv  ->  Downloads/mydata/converted/sweep.h5
```

### If you point *inside* `raw/`

The command only looks at the folder you actually named. Name a single measurement
folder and that folder is not called `raw`, so output lands beside your files —
which means inside `raw/`:

```
tmdc-convert EXP/raw/spot01/01-PL-Vbot --spectra-type PL

->  EXP/raw/spot01/01-PL-Vbot/converted/sweep.h5
```

The same applies to converting one file. Two ways out:

**`--from-raw`** searches *upward* for the nearest folder called `raw` and places
output relative to that:

```
tmdc-convert EXP/raw/spot01/01-PL-Vbot --spectra-type PL --from-raw

->  EXP/converted/spot01/01-PL-Vbot/sweep.h5
```

It is not the default because it reads folders you did not name. If a folder called
`raw` sits high up an unrelated path — `X:/Brandon/raw/01_Projects/…` — then
everything beneath it anchors there, and you could not tell from the command.
Check your path, then use it. If no `raw` is found it warns and falls back.

**`--out`** says where output goes outright, and mirrors the tree beneath the
folder you named:

```
tmdc-convert EXP/raw --recursive --spectra-type PL --out D:/archive
->  D:/archive/spot01/01-PL-Vbot/sweep.h5
```

`--out` and `--from-raw` cannot be combined — both answer the same question, so
giving both is refused. An `--out` path carrying a suffix is taken as one filename
instead of a root, which is meaningful with `--stack`.

### All of it at a glance

| You run | You get |
|---|---|
| `tmdc-convert EXP/raw --recursive` | `EXP/converted/spot01/…` |
| `tmdc-convert EXP/raw --recursive --out D:/a` | `D:/a/spot01/…` |
| `tmdc-convert EXP/raw/spot01/01-PL` | `EXP/raw/spot01/01-PL/converted/…` |
| `tmdc-convert EXP/raw/spot01/01-PL --from-raw` | `EXP/converted/spot01/01-PL/…` |
| `tmdc-convert Downloads/mydata` (no `raw/`) | `Downloads/mydata/converted/…` |

!!! note "An existing output is refused"
    Every writer takes `overwrite=False` and raises `FileExistsError`, matching
    [`write_sweep`](hdf5.md). The file being replaced may be the only copy. For an
    HDF5 output the check runs *before* the decode, so a re-run over a folder of
    12 MB exports refuses immediately rather than parsing each one first.

## Walking a measurement tree

`--recursive` descends the tree and picks the converter per file by content, so a
benchmarking run converts in one command:

```
EXP-2026-08-05-PL-benchmarking/raw/
├── spot01/
│   ├── 01-PL-Vbot-sweep/   PL_*.csv        -> .h5
│   ├── 03-R-Vbot-sweep/    R_*.csv         -> .h5
│   └── ref/                wl_*, laser_*   -> .tif
└── spot02/ …
```

Empty folders and non-CSV files (a stray `.stackdump`) are ignored.

**Two things to watch.**

`--spectra-type` is **one value for the whole run**. A tree holding both PL and
reflectance sweeps needs one pass per type, or the reflectance sweeps are archived
as PL:

```powershell
Get-ChildItem raw -Recurse -Directory -Filter "*PL*" | ForEach-Object { tmdc-convert $_.FullName --spectra-type PL }
Get-ChildItem raw -Recurse -Directory -Filter "*R-*" | ForEach-Object { tmdc-convert $_.FullName --spectra-type R }
Get-ChildItem raw -Recurse -Directory -Filter "ref"  | ForEach-Object { tmdc-convert $_.FullName }
```

Asking for no `--spectra-type` at all is not fatal: the frames still convert, and
each spectral export is reported as an error carrying the loader's own message,
which names the flag and lists the valid types.

**A TRPL directory converts only when you name it.** Reached by `--recursive` it is
reported as deferred and printed, because two measurements in one folder would
merge into a single archive with nothing in the file saying so. Take them one at a
time:

```
tmdc-convert data/trpl-sweep                    # converts
tmdc-convert data --recursive                   # defers, and names the folder
tmdc-convert data/right_spots --prefix right1_  # converts just right1
```

## From the command line

Installing the package provides `tmdc-convert`. An editable install needs
re-installing once for the command to appear.

```
tmdc-convert PATH [--out ROOT] [--from-raw] [--spectra-type {A,PL,R,RC,T,TRPL}]
             [--prefix P] [--stack] [--recursive]
             [--dtype auto|uint16|uint32|float32] [--compression gzip|none]
             [--overwrite]
```

It returns a non-zero exit status if anything failed. One folder can produce both
kinds of output at once — `examples/data/stark-shift` holds 14 frames beside a
spectral export, and a single command writes 14 TIFFs and one `.h5`.

## Which files are converted

Selection is by content, not by filename. A two-row single spectrum is excluded: it
is numeric on its first line exactly like an image, so the row count is what
separates them. In a TRPL folder, a file with a spectral header is that sweep's
parameter-table companion and is read as part of the sweep rather than converted on
its own. Anything excluded is reported rather than passed through.

Pixel type for images follows the data. `"auto"` keeps integer counts exactly —
`uint16` up to 65535, `uint32` above — and falls back to `float32` for anything
else, which is a narrowing from the `float64` the CSV is read as.

::: tmdc_optics_tools.converters
