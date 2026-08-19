# LabRAM export — format and hardware facts

One file per acquisition system. This is the **record** for the LabRAM Raman export:
established from the files committed under `examples/data/Raman/`, and not to be
re-derived by inference from the parser.

What belongs here: how the exporter lays out a file, and the hardware facts a column
encodes. What does not: why the loader responds to any of it the way it does — that is
a decision record in `dev/decisions/`. This file says *what the instrument and its
exporter did*; a decision record says *what we chose to do about it*.

Where a number appears below, it is evidence from one export, not a value a caller
should expect. The *mechanisms* generalise; the arithmetic does not.

## Provenance and unknowns

Established from seven committed files: six single spectra
(`unstrained_bilayer.txt`, `strained_bilayer1.txt`, `strained_bilayer2.txt`,
`unstrained_monolayer.txt`, `strained_monolayer1.txt`, `strained_monolayer2.txt`) and
one spatial map (`map2.txt`).

Still unknown, and no committed file can answer it:

- Which LabRAM model and which vendor software version emits this layout, and whether
  the layout or the header key set is version-stable.
- Whether Y-fast ordering (below) is fixed by the exporter or is a per-session scan
  setting.
- What material and how many layers a file holds. **The header carries neither**, so
  both come from the filename and from session identification, and neither is
  independently checkable from the data. See `dev/physics-conventions.md` §10.

## Export format

**Encoding is latin-1, not UTF-8.** `#AxisUnit[2]=µm` puts a non-ASCII byte in the
header, so a UTF-8 read fails on it.

**The header is a block of `#`-prefixed lines of no fixed length** — 50 of them in
`map2.txt`. Its length varies between files: `#Peaks:Edit=` lines appear only on
acquisitions someone has annotated in the vendor software. So the header is skipped by
content, never by a row count, which would silently misalign the data on a file whose
header is a different size.

Header fields seen include acquisition settings (`#Acq. time (s)`, `#Accumulations`,
`#Binning`, `#Spike filter`, dark/ICS correction flags) and an axis declaration block,
which is the part that carries units:

| Key | Value in `map2.txt` | Meaning |
|---|---|---|
| `#AxisType[0]` / `#AxisUnit[0]` | `Intens` / `Cnt` | the intensity axis, in counts |
| `#AxisType[1]` / `#AxisUnit[1]` | `Spectr` / `1/cm` | the Raman shift axis, in cm⁻¹ |
| `#AxisType[2]` / `#AxisUnit[2]` | `XA` / `µm` | stage X, in micrometres |
| `#AxisType[3]` / `#AxisUnit[3]` | `YA` / `µm` | stage Y, in micrometres |

Axes 2 and 3 appear on a map export. No field in the header is parsed by the loaders;
they are recorded here because they are what establishes the units.

**Two body shapes follow the header**, and they are not distinguishable from the
header:

*Single spectrum* — two whitespace-separated columns, shift in cm⁻¹ (ascending in
every file seen) and counts.

*Spatial map* — tab-separated, every row carrying the same **1026** fields in
`map2.txt`:

| Row | Fields |
|---|---|
| 0 | two **empty** leading fields, then the 1024-point shift axis, shared by every spectrum in the map |
| 1, 2, … | `X`, `Y`, then that position's 1024 counts |

**The map's first row is ragged only under a whitespace delimiter.** All rows hold 1026
tab-separated fields, but a whitespace delimiter collapses runs of whitespace and drops
the two empty leading fields, so row 0 reads as 1024 columns against row 1's 1026. This
is why `numpy.loadtxt` at its default delimiter fails on a map export with
`ValueError: the number of columns changed from 1024 to 1026 at row 2`, and why an
explicit tab delimiter sees 1026 in both.

## Hardware facts the format encodes

**Y is the fast (inner) axis and X the slow (outer) one** in `map2.txt`: the first four
position rows share `X = 0.112262` and step Y, so a full Y sweep runs at each X before
X moves.

**File order is not ascending in either stage axis.** X runs from `+0.1123` µm down to
`−10.6649` µm across the file while Y runs upward, `0.0898` to `14.7288` µm. So a grid
must be built from each row's own `(X, Y)` rather than from row order — which is also
what lets a differently-ordered export of the same shape load correctly.

**The reference map is 10 X × 8 Y = 80 positions**, at 1024 shift points each. The row
count is therefore 81: one shift-axis row plus 80 position rows.

## Measurements not yet represented here

Depth series (a `Z` axis), and any map whose position rows do not fill a complete
rectangular grid. Nothing committed exercises either.
