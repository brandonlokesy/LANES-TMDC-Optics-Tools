# AttoCube export — format and hardware facts

One file per acquisition system. This is the **record** for the AttoCube export:
established from real files on 2026-07-30, and not to be re-derived by inference from
the parser.

What belongs here: how the exporter lays out a file, and the hardware facts a column
encodes. What does not: why the loader responds to any of it the way it does — that is
a decision record in `dev/decisions/`. The distinction in one line: this file says
*what the instrument and its exporter did*; a decision record says *what we chose to do
about it*.

**Every figure below came from one particular export.** The next one will differ —
different raster dimensions, different sweep length, possibly a different acquisition
version. The *mechanisms* generalise and are what the parser keys on; the arithmetic
does not. Where a number appears, it is here because it is evidence, not because a
caller should expect it.

Adding a second instrument: copy the three top-level sections below.

---

## Provenance and unknowns

Read this first, because it qualifies everything after it.

- **Which acquisition software and version emits this format is unknown**, and no file
  answers it. Whether the layout is version-stable is therefore also unknown.
- The **57 labelled parameter rows are format-fixed** — verified identical across PL,
  R, TRPL and the TRPL companion. A missing row therefore means a different
  acquisition version, not routine variation. That is why the loader's permissive
  path makes its error *diagnostic* rather than merely tolerant.

---

## Export format

### Block structure

One column **block** per sweep point, after a `"Parameters Labels"` label column. A
block is a fixed group of consecutive columns carrying everything about that one
point:

```
Parameters Labels │ Par_0 Wavelength0 ExpROI1_0 ExpROI2_0 │ Par_1 Wavelength1 … │ …
   ↑ label column │ ←──────── block 0 (one sweep point) ─→ │ ←── block 1 ───────→ │
```

The block's field names, with the trailing index stripped, identify the **layout**.

### Two layouts, told apart by field names

| Layout | Block | Written by |
|---|---|---|
| spectral | `[Par_i, Wavelength{i}, ExpROI1_{i}, ExpROI2_{i}]` | PL, R, RC |
| temporal | `[Par_i, Wavelength{i}, Exp_{i}]` | TRPL |

**In the temporal layout, the column named `Wavelength` holds time.** An
acquisition-software misnomer. Read it as time; do not "fix" the name in the file.

`R`/`RC` need no parser work — reflectance uses the identical spectral layout as PL.
What reflectance needs is a reference spectrum, which arrives as a 2-row
`SingleSpectrum` CSV of the bare substrate and carries **no parameter rows**.

### Two separate over-allocations

Easy to conflate, and they behave differently — one holds zeros, the other holds
nothing.

**Surplus blocks are numeric zero.** The exporter declares more blocks in the header
than it fills, and the surplus carry literal `0.0` in every field — not blank, not
NaN, so no NaN-strip removes them. Evidence: a 2091-point reflectance raster was
exported with 4182 declared blocks, the surplus half zero-filled throughout.

**Trailing row padding is empty.** Beyond the *named* blocks, every row carries a
further `n_declared × block_width` unnamed, **empty** fields — in the same file, 16728
of them past a named width of `1 + 4182×4 = 16729`, so 33457 fields in total, exactly
twice the named width minus the label column. Nothing needs to strip this: the block
count comes from columns matching `^Par_?\d+$`, and an empty field cannot match, so
the count is correct regardless of how wide the row is.

### A sweep can be a directory, and a raster arrives flattened

- **A TRPL sweep is a directory**, one file per point, each carrying its own full
  57-row parameter snapshot.
- It also has a **metadata companion**: a *spectral*-layout file whose `Par_i` columns
  hold one snapshot per point and whose Wavelength/ROI columns are identically zero.
  The companion collides on `iter_0` with the first data file and is written last, so
  classification must be by **content, not filename**.
- Data-file order comes from the integer in `_iter_N`. Exports are zero-padded but the
  *width* varies between them, so lexicographic order is right only by luck — it puts
  `iter_10` before `iter_2`.
- **A 2-D spatial raster is one flattened file** — 41 X inside 51 Y in the reflectance
  example. Nothing in the rows states the nest; it has to be declared.

### An image sequence can hold one more frame than its paired sweep

A real-space image sequence and the spectral sweep taken alongside it do not always
carry the same number of points. In the committed position scan:

| Directory | Image frames | Paired spectral export |
|---|---|---|
| `examples/data/position-xy-scan/wl/` | **58** (`wl_iter_000000`–`wl_iter_000057`) | declares **57** blocks |
| `examples/data/position-xy-scan/pl/` | **57** (`pl_iter_0000`–`pl_iter_0056`) | declares **114** blocks, i.e. 57 after the zero-filled half is dropped |

So the white light carries one frame more than its sweep has points, and the PL matches
exactly. The extra frame sits at the **end** of the sequence. This is what the exporter
does, not a truncated or corrupted acquisition.

Note also that the two directories pad `_iter_` to different widths — six digits for the
white light, four for the PL — so the integer, not the string, is what orders them.

### The committed fixtures

`examples/data/reflectance-contrast/sample_truncated_…csv` holds the first **50**
points of that 41 × 51 scan — one complete X row plus 9 of the next, which the
filename says and the numbers do not. It is therefore a real fixture for the
aborted-scan refusal and for nothing else: **no complete raster is committed**, and a
test needing one must synthesise it (`tests/test_loaders_nesting.py`).

---

## Hardware facts the format encodes

These are properties of the instrument, not of the file layout, but a column cannot be
read correctly without them.

**`ExpROI1` / `ExpROI2` are two spatial ROIs on the CCD** — the excitation spot and a
remote, spatially-filtered spot, for two-spot galvo scans. `ExpROI2` is identically
zero in every other measurement.

**`Scanner X` / `Scanner Y` are in volts.** Settled 2026-08-04. The scanners are
piezos and the rows carry their *drive voltage*, scale 1.0. A distance requires a
per-stage µm/V calibration that no file contains.

**The TRPL time axis is in ns with 4 ps bins**, giving a ~12.8 ns range. Consistent
with the Picoharp rows and a ~78 MHz repetition rate, but **not independently
confirmed**. The per-file time axes are *not* bit-identical — bin width varies in its
seventh figure — so assembling a directory compares axes with a tolerance, never
equality. The consequence for fitted lifetimes is in `dev/physics-conventions.md`.

---

## Measurements not yet represented here

Absorption, cavity, and BFP/k-space data are measured in the lab and have no loader,
so nothing is recorded about how they export. If they come off this same system they
will most likely reuse the spectral layout above, but that is an expectation, not a
finding.
