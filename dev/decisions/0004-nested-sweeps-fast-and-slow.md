# 0004 — Nested sweeps are declared with `fast_sweep=` / `slow_sweep=`, and flat stays canonical

| | |
|---|---|
| **Status** | Accepted · verification mechanism amended by A13 · §3 superseded by 0016 |
| **Date** | 2026-08-06 |
| **Audit** | E14 · closes A8 |

## Context

A 2-D spatial raster arrives as **one flattened file**, and nothing in the rows states
the nest. Without a declaration the sweep axis is a sawtooth, and a map built from it is
silently wrong.

The motivating measurement is harder than a raster, and it is what settled the design:
both gates moved together to sweep the displacement field at fixed density. Each gate
row then takes a different value at every one of the `n_fast × n_slow` points, so **no
row-level reading can express the nest**, while the field they encode takes exactly
`n_fast` values.

A declaration had been settled on 2026-07-31 as `grid=("Scanner X", "Scanner Y")`,
inner-first.

## Decision

### 1. Two keywords, and they are not aliases of `sweep=`

| Declaration | Question it answers | Cardinality |
|---|---|---|
| `sweep=` | which length-`n_sweeps` array labels each flat point? | one, always |
| `fast_sweep=` / `slow_sweep=` | those points are a nest, `n_fast` inside `n_slow` | zero or two |

Everywhere in this package **"sweep" means the flattened measurement point** —
`n_sweeps`, `sweep_mask`, `sweep_axis` (an array *of length `n_sweeps`*), and the
`index=` the accessors take. The nest declaration is a separate statement. On a nested
scan `sweep=` is normally omitted, so `sweep_axis` is the flat index — the honest
labelling of a raster.

### 2. Both axes resolve through the one sweep vocabulary

`fast_sweep="electric_field", slow_sweep="power"` needed no new vocabulary: the
resolver already accepted a registry key or a raw row. This is the point of the whole
design — a **derived** quantity can be a nest axis, which is the only way the
field × power measurement above is expressible. A keyword-only `param=` was threaded
through the resolver so a bad `fast_sweep=` reports itself rather than naming an
argument the caller never passed.

`sweep_grid()` still detects on **raw rows only**, so on that measurement it reports the
channels rather than the field. It is a diagnostic that says what to declare — the same
relation `gate_mode` has to `gates=`. It also returns the *first* pair that verifies, and
on a raster taken during an anti-symmetric gate sweep two pairs verify equally well, so
it may name the gates rather than the scanners. Nothing in the rows says which pair the
experiment was about, so this is not a defect to chase.

### 3. Verification, not detection

Count distinct values with a span-scaled tolerance — a derived axis is float-valued and
may cross zero — then reshape both axes to `(n_slow, n_fast)` and check the structure
directly: every row of the fast array equals row 0, every row of the slow array is
constant, and the slow values are distinct. One broadcast comparison each.

On failure the message retries the **transposed** reading and, when that works, says so
outright. The reversed declaration is the one mistake worth naming, since detection
cannot. Coordinates come out of the same verified reshape in acquisition order, so a
descending sweep stays descending.

### 4. Flat stays canonical; the grid is a view

`spectra` keeps shape `(n_points, n_sweeps)` **whether or not a nest is declared**.
`as_grid(array)` is one method serving all spectra arrays and every parameter row.

### 5. Accessors, and where an ambiguous coordinate is refused

`get_spectrum_at` (by value) and `get_spectrum_by_index` (by position), each taking
`fast=` / `slow=`, with `nearest_index` underneath for composing. Naming both axes gives
`(n_points,)`; naming one gives `(n_points, n)` with the swept dimension last, so a
slice drops into a spectral map or a peak fit without a transpose.

`axis=` locates a point by a quantity other than the declared sweep axis —
`get_spectrum_at(15.0, axis="top_voltage")` on a field sweep driven by both gates. It is
the fourth entry point onto the same vocabulary, and it is **flat-sweep only**.

An ambiguous coordinate is **warned by `nearest_index` and refused by the accessors**.
Not an inconsistency: a single `int` is all `nearest_index` can return, whereas
`get_spectrum_at` returns data and the API already has the complete answer — a declared
nest addresses every match at once through `fast=` / `slow=`, so handing back one of four
would be a silent partial answer. Matches are compared on the **coordinate** rather than
on the distance, so a request landing midway between two distinct points is not a tie.

### 6. The sweep axis must label each point individually

The loader warns when it does not, asked as *"how many different values does this axis
take?"* rather than *"is it monotonic?"*.

## Rejected

**`grid=(inner, outer)`, the spelling settled on 2026-07-31.** Two objections. The
weaker: the order carries the meaning and nothing at the call site states it, so
reversing it transposes every map silently — the exact failure the declaration exists to
prevent. The stronger: bare `sweep=` would have had to mean "the fast axis", making it
the one place in the package the word meant an axis, two lines from `n_sweeps` still
meaning the flattened point.

**Making `fast_sweep=` an alias of `sweep=`.** They look equivalent in a 1-D scan only
because there is one loop. Keeping them separate cost nothing: no alias, no conflict
branch, no fallout onto `sweep_label` / `sweep_unit`, and an additive HDF5 version bump
rather than a major one.

**Reshaping `spectra` when a nest is declared.** A declaration must not change the rank
of an attribute. `spectra.ndim` would depend on a constructor argument, every consumer
would branch on `is_nested`, `n_sweeps` (defined as `spectra.shape[1]`) would silently
start meaning `n_slow`, and both the axis/signal validation and the HDF5
`axes = "wavelength, sweep"` attribute would be describing something else.

**A `*_grid` attribute per array.** One `as_grid(array)` method serves all of them.

**Fancy indexing in the accessors.** The selectors are `int`s and `slice`s so every
result is a **view**. None is C-contiguous — no column selection out of a row-major array
can be — which the docstrings state rather than claiming otherwise.

**Extending `axis=` to nested sweeps.** There an arbitrary quantity matches `n_slow`
points or one, depending on how the scan was driven, so the return rank would follow the
data rather than the call. Combining `axis=` with `fast=`/`slow=` raises and points at
`fast_sweep=` instead.

**A `get_spectrum_from_parameter` sibling** instead of `axis=`. It is the same selection
read against a different coordinate, so a third accessor would have duplicated the
docstring, the validation and the refusals.

**Guarding the axis-labelling warning on `sweep_grid()`.** That returns `None` for a
field × power nest, so a guarded check would be silent on exactly the case this record
exists for. Monotonicity was rejected for the same class of reason: it catches a nest's
inner quantity (sawtooth) and misses its outer one (staircase), which is the worse
failure of the two. Deliberate repeat measurements warn too — their map collapses the
same way.

**A `constants.py` home for the spectra-source vocabulary.** `source=` takes the
plotting resolver's vocabulary rather than forking it, which meant moving that resolver,
since a loader cannot import `plotting`. It went to `loaders.py`: the table maps names to
*attribute names on the loader classes*, whereas `constants.py` holds physics and
controlled vocabularies. It stays duck-typed, since `SingleSpectrum` mirrors a subset of
those attributes.

## Consequences

- A raster or a nested parameter scan is expressible without a second loader or a
  reshaped attribute.
- **The committed reflectance export cannot test this.** It is the first 50 points of a
  41 × 51 scan — one complete X row plus 9 — so it exercises the aborted-raster refusal
  and nothing else. Raster fixtures are synthetic; the real file has a test of its own
  for the refusal.
- **Not built:** map plotting on top of `as_grid`, a TRPL decay accessor (the base
  machinery is inherited, but the `source=` vocabulary is spectral-specific and no
  nested TRPL sweep has been seen), and snake/bidirectional rasters — a reversing row
  fails verification loudly, which is the right failure until the lab confirms whether
  the instrument writes them.
