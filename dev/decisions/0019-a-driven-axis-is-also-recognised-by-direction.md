# 0019 — A driven axis is also recognised by which way it steps

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-18 |
| **Audit** | A21 · amends `0018` §1 |

## Context

`0018` recognises a driven axis by two signs, either sufficient, and calls a row **held**
only when both fail. Its argument for safety is that "a false collapse needs both signs
wrong at once", and it presents the two blind spots as complementary:

| sign | blind to |
|---|---|
| travel against size | a fine sweep on a large offset — 300.0 → 300.2 K travels 0.07% of its readings |
| how far it steps | a coarse sweep, whose few points step as far as scatter would |

Those are two descriptions of one region, not two disjoint ones. A sweep that is **both**
narrow for its offset and coarse defeats them together, and nothing else was left to see
it by. Sign 2 fires when `median|Δ| < 0.1 × travel`, which for an even sweep of *n* points
means `1/(n-1) < 0.1`, so it needs more than eleven points — while the case sign 2 exists
to catch is exactly the one sign 1 cannot help with. Measured with the shipped helpers:

```
np.linspace(300.0, 300.2, n)  n = 3..10 → held, _axis_atol 0.2 (the full travel), 1 level
                              n = 11    → driven, but only by float noise in the diffs
np.linspace(5.000, 5.004, 5)            → held, _axis_atol 0.004, 1 level
```

So a real five-point gate step of 1 mV at 5 V, or a six-point 300.00 → 300.20 K sweep,
warned that the row never moved and had every coordinate lookup on it refused. `0018` names
that the dangerous direction — "collapsing a real axis loses it… whereas leaving scatter
uncollapsed is what already happened" — and it is the direction the gap runs in.

`0018`'s *Load-bearing choices* is where the reasoning slipped: `_AXIS_STEP_FRAC = 0.1`
"has the least margin at small point counts, **where sign 1 is carrying the decision
anyway**". At small point counts *on a large offset*, sign 1 is precisely the sign that is
blind.

## Decision

### 1. A third sign: the direction the row steps in

A row whose non-zero steps in acquisition order all share a sign is being driven, however
few readings it has. Scatter reverses; a sweep does not turn round.

The sign carries no magnitude and no point count, which is what lets it reach where the
other two cannot. Its own blind spot is an axis that **revisits values** — the fast axis of
a flattened nest restarts every row — and that is the case sign 1 is strongest on, because
such an axis travels its full range.

Exact repeats are dropped before the comparison. A slow axis sits still between its steps,
and standing still is not turning round.

### 2. Sign 3 is *added*, never substituted

This is what separates it from the three rules `0018` rejected, and it is the whole
argument for its safety. Because the signs are OR-ed, a further sufficient sign can only
move rows from held to driven — never the reverse. No sawtooth, serpentine, or log-spaced
sweep can be lost to it whatever the rule says, because whichever sign already recognised
them still does. Verified rather than assumed: sign 1 alone recognises the sawtooth, the
serpentine, the log-spaced power sweep and the slow axis's plateaus, so sign 3 is never
what rescues a nest.

The cost is therefore one-sided and bounded: held rows escaping into "driven", which is the
error `0018` names as safe.

### 3. `_AXIS_MIN_MOVES = 4`, below which the sign abstains

Direction is a coin toss on a row with barely any moves. Measured on a held 5 V gate with
gaussian read-back scatter, the fraction of rows wrongly called driven:

| n | two signs | three signs |
|---|---|---|
| 4 | 1.5% | 1.5% — abstains |
| 5 | 0.5% | 2.2% |
| 6 | 1.1% | 1.2% |
| 8 | 1.0% | 1.0% |
| 12, 20, 66 | — | unchanged |

Four moves is where the reach stops being bought at a price worth noticing. It leaves a
four-point narrow sweep collapsing, which is the residue of this decision.

### 4. The documented reach gains a second edge

`0018` §4 measured the rule against **i.i.d. scatter**, and concluded a held row is
recognised while its read-back is stable to better than *rtol*. Read-back that **drifts**
rather than jitters is seen by signs 2 and 3 at any amplitude, so the honest statement is
that a held row is recognised only while its scatter is directionless. The docstring says
so; the consequence for `gate_mode` is recorded in the audit rather than fixed here.

## Rejected

**Loosening sign 2's `<` to `<=`.** Fixes exactly `n = 11` — the point at which the
constant's own comment says a driven axis steps 10% of its travel — and nothing else. It
treats a boundary as the defect when the defect is a region.

**The three rules `0018` already rejected**, re-examined and still rejected, for a reason
worth restating: each of them *replaces* sign 2. Flooring `_axis_atol` at the row's
magnitude collapses a 201-step sweep at 5 V; sign 2 as a gap **ratio** reads a flattened
sawtooth as pure scatter; scaling sign 2's threshold by `k/(n-1)` judges a 66-point,
11-level sawtooth against 65. All three lose a real axis, which §2 above shows an added
sign cannot do. That structural difference is the reason this record exists rather than a
fourth attempt at replacing sign 2.

**A reversal *fraction* rather than strict monotonicity** — `reversals / (n_moves - 1) ≤
0.25`, so a nearly-monotone sweep with one glitch still counts. It works, and it recognises
a flattened sawtooth directly (16% reversals) instead of leaning on sign 1. Rejected on
two counts: it needs a threshold constant that nothing measures, and its false-driven rate
is erratic in *n* — 10% at six readings against 2% at five — because the count of
comparisons is small and the threshold lands between integers differently for odd and even
lengths. Strict monotonicity has no constant to justify and no such artefact.

**Gating sign 3 behind a minimum relative travel**, `travel > 1e-4 × rms`, so a µV-scale
monotone drift stays held. It does separate the cases that matter — 300.0–300.2 K is
6.7e-4 of its RMS, a 20 µV drift on 5 V is 4e-6 — but it costs a second constant to defend
and the margin is thinner than it looks: a 1 mV drift on 5 V is 2e-4 and would still read
as driven. No committed file shows the case. Left out; §4 documents the exposure instead.

**Fixing it in `_axis_atol` rather than `_axis_driven`** — keeping `rtol × travel` as a
floor under the held branch. That is `0018`'s rejected magnitude floor wearing different
clothes, and it makes the tolerance disagree with the classification that produced it.

## Consequences

A sweep of five readings or more is recognised however narrow it is for its offset, which
is what `main` did before `0018` and what `0018` did not restore. Four-point narrow sweeps
still collapse; `get_spectrum_by_index()` remains the escape, as `0018` established.

`_AXIS_MIN_MOVES` counts **moves**, not readings, so a slow axis of five levels over twenty
readings clears it on four steps rather than nineteen. That is the same arithmetic that
makes the plateaus safe.

Verified to the standard `0018` set for itself: every loadable spectral export and the TRPL
directory reclassifies **zero** rows, and the suite passes with the rule in place.

A held gate whose read-back drifts monotonically now enters `varying_parameters()`, and
through it `gate_mode`. Sign 2 already did this; sign 3 makes it likelier. Recorded as open
in the audit, not fixed — the honest fix is a declared instrument resolution, which is the
`sweep_atol=` argument `0018` left unbuilt for want of a file that needs it.

## Load-bearing choices

**"Added, never substituted" (§2).** It is the whole safety argument, and it is what a
fourth sign must also satisfy. Anything that *replaces* an existing sign has to re-clear
every counterexample in `0018`'s Rejected section; anything OR-ed in only has to justify
its false-driven rate.

**`_AXIS_MIN_MOVES = 4` (§3).** The one number here that trades reach against the A20 fix.
Lowering it to 3 reaches four-point sweeps at 9% false-driven; raising it to 5 gives back
five-point sweeps for 0.2%.
