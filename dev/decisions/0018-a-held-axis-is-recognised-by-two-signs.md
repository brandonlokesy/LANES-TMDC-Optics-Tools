# 0018 — An axis that never moved is recognised by two signs, either sufficient

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-18 |
| **Audit** | A20 · completes what `0016` left at the flat-sweep call sites |

## Context

`_axis_atol` answers *how far apart may two readings be and still be the same instrument
setting*, and two things turn on the answer: the warning that a declared sweep axis does
not label its points individually, and the refusal of a coordinate that does not identify
one spectrum (`0004` §5).

Answering it with `1e-3 ×` the axis's own span is circular when the axis did not move.
The span is then the instrument's own scatter, so the tolerance comes out a thousandth of
the scatter it exists to absorb — and a quieter instrument does not help, because the
tolerance shrinks in the same proportion. Twelve spectra taken at one power setting were
read as twelve settings, and both safeguards above fell silent rather than firing.

`0016` solved the same underlying problem for a *nest* by comparing each level with its
neighbour, which needs the grid to lean on. A flat sweep has no grid, so it needs a
different answer.

The hard part is that the two situations are numerically identical. Twelve readings spread
over 0.33 µW around 500 µW are either one setting plus scatter or a very fine sweep; a
real 12-step sweep over the same span produces the same numbers. Nothing in the values
alone separates them, so the decision must draw on something else.

## Decision

### 1. Two independent signs that a row was driven, either sufficient

| | sign 1 — travel against size | sign 2 — how it steps |
|---|---|---|
| test | `span > rtol × rms` | `median\|Δ in acquisition order\| < 0.1 × span` |
| draws on | how far it went for its own magnitude | that a sweep *progresses* where scatter *jumps about* |
| blind to | a fine sweep on a large offset — 300.0→300.2 K travels 0.07% of its readings | a coarse sweep, whose few points step as far as scatter would |

A row is called **held** only when both fail. The two mistakes are not symmetric:
collapsing a real axis loses it, and the caller gets spurious warnings and refused
lookups, whereas leaving scatter uncollapsed is what already happened. Requiring agreement
means a false collapse needs both signs wrong at once.

Sign 2 reads consecutive differences in **acquisition order**, not sorted order. That is
the whole of its value: sorted gaps are about `span/n` for scatter *and* for a sweep, so
they cannot tell them apart, whereas acquisition order is the structure a sweep preserves
and scatter destroys. The median rather than the mean, so the jump back at the end of each
row of a flattened nest is outvoted instead of reading as scatter.

### 2. A held axis is judged on its whole span

The same number, used honestly in each case. For a driven axis the span measures how far it
travels, so a fraction of it is a fair proxy for how finely it steps. For a held one the
span measures the instrument's scatter, so everything inside it is the one setting.

### 3. `varying_parameters` shares the helper

It asked the same question with sign 1 alone. Both now call `_axis_driven`, so the report
of what varied and the tolerance behind `get_spectrum_at` cannot contradict each other —
which its own comment already asked for. Its ranking by span relative to RMS is unchanged.

### 4. The reach is documented, not implied

Because sign 1 fires as soon as scatter exceeds `rtol` of the reading, a held setting is
recognised only while the read-back is stable to better than that: 100% up to 1e-4
relative scatter, 0% from 1e-3. A source-meter holding a gate is recognised; a power meter
holding a power is not, and reads as many settings exactly as before. A test pins the
boundary so a reader does not have to rediscover it.

## Rejected

**Flooring the tolerance at a fraction of the row's magnitude** (`max(rtol*span,
rtol*rms)`) — the one-line fix that `A19` originally prescribed, and the reason that sketch
was withdrawn rather than followed. It fixes every held row and destroys a real
measurement: a 201-step sweep from 4.99 V to 5.01 V has a threshold set by the 5 V offset
at fifty times its own 0.1 mV step, so all 201 settings collapse into one. An offset has
nothing to do with how finely a sweep steps, so it must not set the tolerance.

**Sign 1 alone.** Calls a genuine 300.0–300.2 K temperature sweep noise. Kelvin is the
trap: an absolute scale puts a large offset under a narrow sweep.

**Sign 2 alone, at any fixed threshold.** A coarse sweep steps 20% of its range per
reading at 6 points and 50% at 3, against scatter's ~40%. The populations overlap, and no
threshold separates them.

**Sign 2 as `acquisition gap / sorted gap`**, which is dimensionless, needs no threshold
that depends on the point count, and is sharper for monotonic sweeps — a real improvement
on two counts. Fatal for any axis that revisits values: a flattened nest sawtooth scores
3.9e+05 and reads as pure scatter, because its sorted gaps are within-level scatter while
its consecutive gaps are a full step.

**Sign 2 with a threshold scaling as `k/(n-1)`**, which removes the dependence on
magnitude entirely and would have caught a held power meter. Same failure: for a repeating
axis the point count is not the level count, so a sawtooth of 66 points over 11 levels is
judged against 65 and collapses. Fixing the reach for a noisy meter and keeping flattened
nests intact is not something a single scalar rule appears to do.

**Requiring both signs to agree before calling a row driven** (rather than before calling
it held). Catches the noisy power meter, and collapses every coarse sweep — the dangerous
direction.

**A `sweep_atol=` constructor argument** to state the instrument's resolution outright. It
is the thing the researcher actually knows, and it would close every corner above. Left
unbuilt: `get_spectrum_by_index()` is already the escape from a wrongly collapsed axis and
the refusal message names it, and no committed file needs the parameter. To be revisited if
one does.

## Consequences

Files with a genuinely held declared sweep axis now **warn at load** and **refuse
coordinate lookups** where before they did neither. That is the repair, but it appears on
data that loads quietly today. Verified against every tracked export and the TRPL
directory: exactly one row reclassifies, `V_A` held at −5 V in
`PL_Vbot_power_sweep_26_08_10_…csv`, which should be one setting and is not that file's
sweep axis. Zero test fixtures reclassify.

Three advice paths were corrected because the fix made them reachable: the repeat warning
read *"1 different values"* and offered nest advice for a row that never moved, the
accessor refusal did the same, and `_nearest`'s distance warning reported a value as
absent from the only axis holding it. That last one is now floored at `_axis_atol`, which
on a driven axis is far the smaller of the two thresholds and changes nothing.

`_AXIS_RTOL` now does double duty — the fraction two readings of a driven axis may differ
by, and the fraction of its own magnitude a row must travel to count as driven. One
constant, so the two cannot drift apart.

## Load-bearing choices

**"Either sign is sufficient" (§1).** It is what makes the rule safe and what caps its
reach at §4. If a held noisy axis ever needs recognising, this is the line to revisit —
and the two rejected magnitude-free rules above are the evidence that the obvious ways to
do it break flattened nests.

**`_AXIS_STEP_FRAC = 0.1` (§1).** Chosen because a driven axis of n points steps about
`1/(n-1)` of its travel — 10% at 11 points — while scatter sits near 40% regardless of n.
It has the least margin at small point counts, where sign 1 is carrying the decision
anyway.
