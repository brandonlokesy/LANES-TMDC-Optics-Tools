# 0025 — `plot_image` carries its colorbar, and a new member is appended

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-19 |

## Context

[0024](0024-long-plotting-returns-are-named.md) named the four long plotting returns and
recorded, in its load-bearing section, that `plot_image` builds a colorbar and does not
return it — a gap in the very contract that record is about. This closes it.

The gap is not that the object is unreachable. Matplotlib sets a back-reference on the
mappable, so `plot_image(img).im.colorbar` already yields the `Colorbar`, and `None` when
`colorbar=False`; `tests/test_plotting_labels.py` had been reading
`im.colorbar.ax.get_ylabel()` all along. The gap is that the package's own return contract
did not carry it, so `SpectralSeriesPlot` had a `cb` member and `ImagePlot` did not, for no
reason a caller could see.

The question this forced is where a new member goes. `ImagePlot` was `(fig, ax, im, circle)`
and a colorbar sits at position two in `SpectralSeriesPlot` — `(fig, ax, cb, lines,
ax_twin)` — so matching that order would have put `cb` third.

## Decision

1. **`plot_image` returns its colorbar** as `ImagePlot.cb`, `None` when `colorbar=False`.
   The mappable's back-reference still exists and still agrees; the member is what the
   package promises, and `tests/test_plotting_return_shapes.py` pins
   `res.im.colorbar is res.cb` so the two cannot drift.
2. **A new member is appended, never inserted.** `ImagePlot` is
   `(fig, ax, im, circle, cb)`. Every position that existed keeps its meaning, so a caller
   indexing `[3]` still gets the circle. Ordering consistency across classes is worth less
   than positions that never change meaning.

## Rejected

**Leaving it to `im.colorbar`.** The colorbar is genuinely reachable that way, it is
matplotlib's own documented attribute, and it costs nothing. Rejected because it makes the
two classes disagree for no visible reason: a reader who has seen
`SpectralSeriesPlot.cb` looks for `ImagePlot.cb`, does not find it, and has no way to tell
whether the colorbar is absent, unreachable, or merely undocumented here. A back-reference
is also the wrong shape for the rule 0024 restates — the return contract *is* the styling
API, so an artist the function drew belongs in the return.

**Inserting `cb` third, to match `SpectralSeriesPlot`'s order.** Tidier to read across the
two classes. Rejected because it silently changes what position three holds: every existing
`fig, ax, im, circle = plot_image(...)` would keep working and hand back the colorbar as
`circle`, with no error anywhere. That is exactly the failure 0024's field-order rule and
the CLAUDE.md entry forbid. Cross-class ordering is a reading convenience; a silent
value swap is a wrong plot.

**Not adding it because appending lengthens the tuple.** Appending does break any caller
that unpacks exactly four values, which a `NamedTuple` cannot avoid — 0024 already recorded
that as the one thing a dataclass would fix. Rejected as a reason to stop: the package is
pre-adoption, the seven affected sites are all in the suite, and refusing to complete a
contract because completing it has a cost is how the gap became worth a record in the first
place.

## Consequences

- `ImagePlot` has five members. Callers unpacking exactly four must add a slot or, better,
  read the member by name — which is what the seven updated sites in
  `tests/test_plotting_laser_circle.py`, `tests/test_plotting_cmap.py` and
  `tests/test_plotting_return_shapes.py` now do, so a future appended member will not touch
  them again.
- One positional index disappeared: the multi-panel test in
  `tests/test_plotting_laser_circle.py` read `plot_image(...)[3]` and now reads
  `.circle`.
- `cb` is initialised to `None` before the `if colorbar:` block, so the name exists on
  every path.
- 0024's load-bearing note about this gap is now closed. That record is not edited beyond
  its status row.
