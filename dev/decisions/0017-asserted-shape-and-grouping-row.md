# 0017 — A nest's shape and its grouping row are declared with named keywords

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-17 |
| **Audit** | A19 · builds on `0016`, extends `0004` §1 |

## Context

`0016` made a nest resolve by whether each axis's levels hold apart. That is the right
test, and it leaves two things a file cannot always supply.

**A file whose readings establish no shape at all.** The measurement behind `A19` is
11 gate steps inside 6 commanded powers, and the laser plateaued: three setpoints
produced 834.3–836.9, 835.5–837.2 and 835.5–836.8 µW. Those ranges sit on top of one
another, so the power column cannot say which spectrum belongs to which setting, and
`0016` correctly refuses. The researcher, however, knows the scan they programmed.

**A file where two different rows are needed for two different jobs.** In that same
file `Fianium_Select_A4` is the commanded setpoint and steps exactly — 160000, 156000,
152000, 148000, 144000, 140000 — so it says which spectra share a power.
`Excitation Power` is the meter reading and carries the µW the axis should be labelled
in. One row cannot do both: the setpoint has no physical units, and the reading cannot
resolve its own levels.

The power is set through a filter wheel whose position is not linear in output power, so
converting the setpoint to µW is not available either.

## Decision

### 1. The shape is asserted with `n_fast=` / `n_slow=`, named separately

Either count alone is enough, since `n_sweeps` is exact; giving both cross-checks them.
A count that does not divide `n_sweeps`, or that leaves fewer than 2 points on the other
axis, raises — an aborted scan is still refused rather than truncated.

`n_fast=` / `n_slow=` require `fast_sweep=` / `slow_sweep=`. A shape says how the points
divide up, not what was scanned along each axis.

### 2. An asserted shape suppresses the refusal, not the checks

The overlap test still runs and **warns**, naming each axis that cannot tell its own
settings apart and quoting the overlapping ranges. On the file this exists for, that
names the power axis, which the caller already knows about. On a shape asserted the wrong
way round it names the *gate* axis — an axis known to be clean, reporting that each of
its levels spans the full 10 V — which is what surfaces a transposition that no check can
detect.

`SweepNesting.asserted` records that the shape was named rather than established, and
`__repr__` says `shape asserted`.

### 3. Grouping is separated from labelling by `fast_group_by=` / `slow_group_by=`

The grouping row establishes the levels and is what must hold apart; the declared axis
supplies the coordinates and need not. Where the labelled row's own levels overlap, the
load warns that the grid is sound while the axis is not usable for plotting or for
`get_spectrum_at`.

```python
slow_sweep    = "power",                 # coordinate, in µW
slow_group_by = "Fianium_Select_A4",     # the exact setpoint
```

### 4. A level's coordinate is the median of its readings, and the spread comes with it

`fast_spread` / `slow_spread` carry each level's peak-to-peak range beside the
coordinate. On the file behind `A19` two levels drift by 0.31 and 0.23 µW per point while
sitting 0.25 µW apart, so a single number per level hides that they are not flat
settings.

### 5. Both declarations round-trip through HDF5, conditionally

`n_fast` / `n_slow` are written **only when the shape was asserted**, and the grouping
rows **only where they differ from the axis they group**. A nest the readings established
re-establishes itself on read, which is what keeps the overlap checks doing their job.
`FORMAT_VERSION` goes to `2.2`; the reader only refuses on a major mismatch, so this is
additive.

## Rejected

**`sweep_shape=(N_fast, N_slow)` as a single tuple**, which is what was first proposed.
`0004` rejected `grid=(inner, outer)` because *"the order carries the meaning and nothing
at the call site states it, so reversing it transposes every map silently"*, and a
product check cannot recover the order: `66 = 11 × 6 = 6 × 11` both pass. Named counts
make the reversal unspellable, which is the same move `0004` made when it replaced
`grid=` with two keywords.

**Requiring both counts.** Redundant against an exact `n_sweeps`, and redundancy is what
creates the possibility of an inconsistent declaration. Both are *accepted* and
cross-checked, because a caller who states both is asking to be told when the scan they
remember is not the scan in the file.

**Letting an asserted shape skip the overlap checks.** They cost nothing, they are the
only thing that can surface a transposition, and going quiet would make an assertion look
like a verification.

**`force_power_by_fianium=True`**, a boolean that would have passed
`slow_group_by="Fianium_Select_A4"` internally. Three objections. The file carries eight
`Fianium_Select_A0..A7` channels among its 57 rows, and which one drives the power is
per-session configuration — the same category as `gates=`, which `0002` requires be
declared and nowhere else. It would hard-code one channel of one laser into a signature
the group is meant to share. And its entire body would be passing a string the caller can
already pass, which is the shape of parameter *parameters earn their place* excludes.

**Doing nothing, on the grounds that `as_grid` already reduces the measured row.** It
does — `scan.as_grid(scan.power)` needs no new API — but it cannot put µW on a plot axis
without the caller reducing and relabelling by hand at every call site.

**The mean of a level's readings.** Differs from the median by at most 0.47 µW on the
real file, so it buys nothing in accuracy, and one stray reading moves it by up to
34 µW against the median's 0.00 µW. A stray that small still passes the `0016` overlap
test, so the reduction is the last thing standing between it and the axis.

**The first reading of a level**, which is what `slow_axis` was before this. Measured to
be biased low by up to 1.3 µW on exactly the two levels that drift, because the first
spectrum of a row is taken before the source settles.

**Warning at load when the reduced coordinates contain near-duplicates.** `0004` §5
already refuses an ambiguous coordinate at the accessor, which is the point of use and
the place a caller can act on it. The load-time warning is reserved for levels that
*overlap*, which is a statement about the axis rather than about one lookup.

**Storing `n_fast` / `n_slow` unconditionally in HDF5.** Every round trip would then be
an assertion, silently downgrading the `0016` checks to warnings for files that never
needed it.

## Consequences

A measured axis can now be the coordinate of a nest whose structure comes from a
commanded one, which is what makes a power series on this rig expressible with the axis
labelled in µW.

`SweepNesting` gains four fields — `asserted`, `fast_spread` / `slow_spread`,
`fast_group` / `slow_group` — all defaulted, so constructing one positionally is
unchanged. `__repr__` gains `via <row>` where a grouping row differs.

**The refusal message names the argument that was actually tested.** Grouping by a row is
a claim that it steps cleanly, so when it does not the message says `slow_group_by=`
rather than blaming `slow_sweep=`.

An asserted shape still cannot rescue a bidirectional raster: it asserts the shape, not
the acquisition order, and would reverse alternate rows in silence. `0004` leaves snake
rasters unbuilt and that is unchanged.

Nothing about this makes the plateau in the reported file acceptable — it makes it
*visible*, through the spread beside each coordinate and the warning about the axis. Why
the laser did it is a lab question, recorded in `A19`.

## Load-bearing choices

**A transposition is caught by a warning, not by a check (§2).** It cannot be caught by a
check — that is what asserting a shape gives up. If transposed shapes start reaching real
analysis, the question is whether the warning should become a refusal when the *fast*
axis fails, not whether to detect it some other way.

**The median (§4).** It is a judgement about what represents a level, and the file that
settled it had only 11 readings per level with two levels drifting. A level sampled
differently — many more points, or a bimodal source — is the case that would reopen it.
