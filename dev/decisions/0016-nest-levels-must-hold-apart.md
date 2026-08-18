# 0016 — A nest is verified by levels holding apart, not by a tolerance

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-17 |
| **Audit** | A19 · supersedes `0004` §3 |

## Context

`0004` §3 settled verification as *"count distinct values with a span-scaled tolerance,
then reshape both axes and check the structure directly"*. `A13` then made the counting
and the checking agree on what "the same grid point" means, and recorded what it had not
fixed: *"an axis whose scatter approaches its level spacing will chain levels together."*

That is not a tuning problem. One tolerance per axis has to sit **above** the scatter
within a level and **below** the step between levels, and for a measured axis there need
be no such value. Two properties of a power sweep close the window together:

- Meter scatter is roughly a fixed fraction of the reading, so the scatter to be
  tolerated is set by the **largest** value on the axis.
- A power series is conventionally log-spaced, so the smallest step to be resolved is a
  **tiny slice** of the span the tolerance is scaled to.

Measured on the file that forced this, an 11 × 6 gate-inside-power nest: worst
within-level scatter 3.06 µW against a tolerance of 0.42 µW. A clean synthetic sweep with
levels 100 µW apart needs only 0.6 µW of meter noise to fail the same way.

The orientation mattered too, and worse. With power as the *slow* axis the structural
checks at least ran and failed. With power as the *fast* axis the miscount became
`n_fast`, `n_sweeps % n_fast` was non-zero, and nothing structural was examined at all.

## Decision

### 1. Separation replaces tolerance

A shape is verified by asking whether the readings of different grid levels **interleave**.
Each level's lowest-to-highest range must begin above where the level below it ended.

The property that matters is that every comparison is **local**: one level's own scatter
against the step to the level immediately beside it, at the same place on the axis. The
scatter at the top of a log-spaced axis is never weighed against the step at its bottom,
which is the comparison a single tolerance is forced to make.

The same test serves both axes, which is what makes the orientations symmetric: a slow
axis holds still while the fast axis runs, so its levels are the grid's **rows**; a fast
axis repeats its run in every row, so its levels are the grid's **columns**.

### 2. The shape is enumerated, not counted

`n_sweeps` is exact, and the exporter runs the fast axis fastest, so every slow step
carries one complete run of the fast axis and `n_fast × n_slow == n_sweeps` holds exactly.
Only the divisors of `n_sweeps` can be shapes. Each is tested; `n_fast` is therefore
**never derived from a level count**, which is what removed the tolerance from the fast
axis as well as the slow one.

Exactly one shape may survive. Several is refused by naming them, on the same reasoning
`0004` §5 refuses an ambiguous coordinate: the file does not say which was measured, and
choosing silently would be a wrong answer rather than an error.

### 3. Separable, not tight

The test proves levels can be **told apart**, not that each is narrow. A level whose
readings are spread far wider than expected still passes when that spread happens to fall
in a gap. This is a known limit, not an oversight — see *Rejected*.

### 4. A refusal is anchored, and quotes the readings

When no shape fits, the message anchors on whichever axis *does* separate, because that
axis pins the shape the file actually has, and quotes the overlapping ranges of the axis
that does not.

## Rejected

**Raising `_NEST_RTOL` from `1e-3` to `1e-2`.** Fixes the linear case and the mildest
log case, and still fails a log-spaced sweep at 3% meter noise. Its real cost is at the
other end: the tolerance must also stay below the step between levels, roughly
span ÷ levels, so at `1e-2` a **101-step gate sweep collapses to one level** and is
refused. It trades a broken measurement for a different broken measurement, and buys one
arbitrary constant with another.

**Choosing the tolerance from the data** — sorting the gaps between readings and cutting
at their largest multiplicative jump. Removes the constant, and measured, it still fails
a log sweep at 3% noise and reports a constant axis as 66 distinct values. It is still
one global threshold, so it inherits the whole problem; it only moves where the threshold
comes from.

**Trimming outliers before comparing**, to make the range less sensitive to one bad
reading. Measured over 400 trials, trimming one reading per level accepted **92%** of
trials in which spectra really were mixed up, and accepted a reading three full steps out
of place. It buys robustness with false acceptance, and the two failures are not
symmetric: a refusal is visible and the caller can declare a commanded row instead,
whereas an acceptance files spectra under the wrong setting and says nothing.

**Requiring the neighbour gap to exceed a level's own spread**, which would have given
§1 tightness as well as separability. It is scale-free and it was measured: it refuses
genuine log-spaced power sweeps, because at the top of such an axis the spread is
comparable to the step even when the levels are plainly distinct. Separability is the
weaker claim and the correct one.

**Grouping the finite readings of an axis that also holds a non-finite one.** A level
containing a non-finite reading has no range to compare, and grouping around it would put
a spectrum on a level nothing measured. The axis is refused instead, which preserves
`0004`'s refusal of a non-finite slow coordinate.

**Keeping the old failure message.** It reported `n_fast × n_slow ≠ n_sweeps` from counts
the algorithm no longer uses for anything — on the reported file, *"11 × 9 = 99, not
66"*. Neither number decided the refusal, so the message described an arithmetic mismatch
where the failure was structural, and sent the reader to `sweep_grid()`, which returns
`None` for a measured axis by construction.

**Anchoring that message on an arbitrary divisor.** Written first and caught in review.
Walking divisors and reporting the first where exactly one axis separates blamed a
perfectly healthy gate axis at a 6 × 2 shape nobody had declared.

## Consequences

`_axis_atol`, `_level_labels` and `_count_distinct` all remain, and still describe and
match the **sweep axis** — `_warn_if_sweep_axis_repeats` and the accessor coordinate
lookup. They no longer decide a nest. `_NEST_RTOL` is renamed `_AXIS_RTOL` to stop the
name claiming otherwise.

A measured axis is now usable as either half of a nest, in either order, which is what
makes a power series run at each gate voltage expressible at all.

**Resolving the structure does not make a coordinate usable.** The file behind `A19`
groups correctly by its commanded row and then measures 416.9, 824.7, 835.6, 836.4, 836.1
and 722.2 µW across its six levels — not monotonic, three within 1 µW. `0004` §5 still
refuses a lookup against it, correctly. Separating *what groups the sweep points* from
*what labels them* is owed work, not settled here.

An axis that only ramps monotonically is now refused as a nest axis, where before it
could pass: its columns overlap under every shape. A ramp is not a grid, so this is the
intended reading rather than a regression.

## Load-bearing choices

**Separable rather than tight (§3).** If a future file resolves to a shape that is
structurally legal but physically wrong, this is the assumption to revisit — and the
measurement above is the reason the obvious strengthening was not taken.

**Exactly one surviving shape (§2).** No real file was found that reshapes two ways, so
the refusal path for ambiguity is reasoned rather than observed. If it starts firing, the
question to ask is whether a third signal should break the tie, not whether to pick one.
