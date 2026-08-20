# 0028 — The spectral map pins a nest, through the resolver the series already used

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-20 |
| **Extends** | [0004](0004-nested-sweeps-fast-and-slow.md), [0012](0012-plot-spectrum-selects-like-the-accessors.md) |

## Context

[0004](0004-nested-sweeps-fast-and-slow.md) opens on the failure a nest declaration
exists to prevent: *"Without a declaration the sweep axis is a sawtooth, and a map built
from it is silently wrong."* It closes by listing **map plotting on top of `as_grid`**
as not built.

`plot_spectral_map` was the function that gap named. It read `scan.sweep_axis` and the
whole flat `(n_pixels, n_sweeps)` array, unconditionally. On a declared nest `sweep=` is
normally omitted, so `sweep_axis` is the flat index (0004 §1) — the map drew every point
of the raster, with consecutive rows at unrelated settings and a y-axis that named
nothing. It was the one plot in the module that could not say what its rows meant.

`plot_spectral_series` had already solved this. It holds one nest axis with `fast=` /
`index_fast=` / `slow=` / `index_slow=`, calls `get_spectrum_at` / `get_spectrum_by_index`
with the other left free, and gets back the `(n_pixels, n)` block 0004 §5 describes as
dropping "into a spectral map or a peak fit without a transpose". The block was already
the right shape for a map; nothing consumed it as one.

The measurement that forced it is a gate voltage nested inside excitation power. Its
useful figures are *one map per power* and *one map per gate setting* — a line of the
grid, not the grid.

## Decision

### 1. One resolver, two plots

`plotting._resolve_sweep_block` returns `(data, coord, coord_label)`: the `(n_pixels, n)`
spectra a plot draws and the `(n,)` coordinate labelling their columns. A flat sweep and
a pinned nest both come back in that shape, so neither caller branches on `is_nested`.

It is the block lifted out of `plot_spectral_series` unchanged, not a reimplementation.
Selection still goes through the loader's accessors, so 0004 §5's policies — an ambiguous
coordinate refused, a distant one warned — reach the map without being restated.

The caller's own name for the flat-sweep axis argument is passed as `axis_param=`, the
convention `_lookup_axis` already uses, so a bad value reports itself as `y_axis=` in a
map and `series_axis=` in a series rather than as the accessors' `axis=`.

### 2. An unpinned nest is refused

`plot_spectral_map(raster)` raises and names the four keywords, exactly as the series
does. The flat-index map is not offered as a default.

### 3. `spectra_source=` and `y_axis=` land with it

The map took neither. `spectra_source=` was hardwired to `"best"`, and no argument read a
flat sweep against another quantity. Both exist on the series with settled meanings, so
the map takes the same names, the same vocabulary, and the same flat-sweep-only rule for
the axis argument. The colour-bar default follows the source, so a `"contrast"` map is
labelled ΔR/R₀ rather than as PL counts.

## Rejected

**Keeping the flat-index map as the nest default, with pinning optional.** It reads as
the safe choice and is the opposite. The two functions would then disagree about what an
unpinned nest means — the series refuses it, the map draws it — and the thing the map
drew is the picture 0004's first paragraph exists to prevent. A researcher who forgot to
pin would get a figure rather than an error, which is how a sawtooth reaches a slide.
The cost is real and was accepted: a call that worked before now raises. Alpha, and the
break is the feature.

**Duplicating the resolution block into the map.** Sixty-five lines, three error
messages and every refusal 0004 settled, in two places. The second copy is where the
first one's next fix does not land.

**Naming the axis argument `series_axis` on the map.** A map draws no series. Naming a
parameter after a concept its own function does not have makes the shared helper legible
at the cost of every caller.

**`y_axis` taking `x_axis`'s vocabulary.** The two sit next to each other and look like a
pair, but `x_axis` orders detector pixels (`"energy"` / `"wavelength"`) and `y_axis` names
a sweep quantity. Merging them was never possible; the docstring says so outright instead,
and `_lookup_axis` lists what it does accept when a caller guesses wrong.

**Extending `axis=` to nests, in either function.** Still rejected, for 0004's reason
unchanged: an arbitrary quantity matches `n_slow` points or one depending on how the scan
was driven, so the return rank would follow the data rather than the call.

**Reshaping the map onto the whole grid.** `as_grid` gives `(n_pixels, n_slow, n_fast)`,
which is three dimensions and a mesh takes two. Choosing which to collapse is a
measurement decision, and pinning is the caller stating it.

## Consequences

- A raster or a nested parameter scan has a map, along whichever axis was left free. The
  y-axis carries that axis's own label and unit rather than an index.
- **`plot_spectral_map` on a nested scan raises where it used to draw.** No committed
  notebook, test or README snippet does this; the untracked gate × power notebook that
  motivated the change is the only caller, and it had an empty cell waiting.
- `plot_pl_map_Vab_scan` forwards `*args, **kwargs`, so the deprecated alias inherits all
  six new keywords and its promise to accept exactly what the new name does stays true.
- **`plot_spectral_series`' nest path acquired its first tests.** It had none: no test in
  the tree passed `fast=`, `slow=`, `index_fast=` or `index_slow=` to it. Rewording its
  errors with nothing pinning them is what surfaced this.
- `dev/TODO.md`'s "still open: plot it as a map" entry is **not** closed. That asks for a
  `plot_spatial_map` drawing a fitted quantity over real-space *x*/*y*; this draws spectra
  along one line of a nest. Different figure, different inputs.

## Load-bearing choices

- **The refusal in §2 is the part to revisit if the lab pushes back.** It is one branch in
  the shared helper and could become a warning plus the old behaviour without touching
  anything else. Nothing else in this record depends on it.
- **`data` comes back as a view** on the pinned-nest path, since the accessors return one
  by design (0004, *Rejected*: fancy indexing). Both callers copy before filtering. A
  third caller must too.
