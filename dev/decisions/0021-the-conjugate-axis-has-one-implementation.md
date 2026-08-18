# 0021 — The conjugate spectral axis has one implementation, and a plot returns the one it drew

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-18 |
| **Audit** | E16 · E17 |

## Context

A spectrum is read in two units at once: the physics is in eV, the instrument and most of
the literature are in nm. So a plot carries the other unit along the top edge. The package
had **three** implementations of that one axis.

Two were the same 21 lines, comments included: `plot_spectrum` and `plot_power_series` each
called `ax.twiny()`, filtered the primary axis' ticks, and rewrote their *text*. The third,
`_conjugate_x_axis`, was added with `SpectrumLinePanel` and uses
`ax.secondary_xaxis("top", functions=(f, f))`.

The duplication is not the whole problem — the two copies are also wrong in a way that is
visible on the page:

- Tick *positions* stay in the primary unit. On a 500-800 nm sweep plotted in energy, the
  relabelled top axis reads 495.9, 550.4, … where the transformed one reads 500, 550 …
  800. A wavelength axis a reader cannot read round numbers off is not doing its job.
- The labels are frozen at build time. `ax.set_xlim(1.60, 1.80)` after plotting — the
  ordinary act of zooming onto a peak — leaves the top axis showing the wavelengths of the
  old view. It is then not merely ugly but wrong.
- The unit strings are hardcoded, bypassing `constants.X_AXES` and `_x_axis_name_unit`,
  which every other spectral label in the package composes from.

What forced the decision was a third copy that nearly landed. Work on an unrelated
change — annotating the laser spot — gave `plot_spectrum` a `twin_axis` by pasting the
`plot_power_series` block into it, comments and all, months after `_conjugate_x_axis`
already existed. It was taken back out of that change before it merged, so the third copy
is not in the history; it is in this record because the near miss is the argument. A
duplicated block attracts a copy faster than it attracts a fix, because copying is the
cheapest way to answer "how does this function do it?".

`HC_EV_NM / x` is its own inverse, so the forward and reverse transforms
`secondary_xaxis` wants are one function passed twice — which is why one helper covers
both directions rather than one per direction.

## Decision

1. **There is one implementation of the conjugate spectral axis**, `_conjugate_x_axis`,
   and it is built with `secondary_xaxis` on a live transform. Anything drawing that axis
   calls it. A second hand-rolled `twiny` is a defect, not a variant.
2. Its label is composed from `_x_axis_name_unit`, so the top axis carries the same string
   the bottom axis would if the two were swapped — [0011](0011-label-contract-derive-or-verbatim.md)'s
   derive-or-verbatim contract, applied to a second axis.
3. **A function that draws the conjugate axis returns it.** `plot_spectrum` returns
   `fig, ax, line, ax_twin`, with `ax_twin` `None` when `twin_axis=False` — the fixed
   arity and the `None` mirroring how `plot_power_series` already returns its `cb`.

`plot_power_series` is not migrated here. Its return is a released 4-tuple, so the same
change is a real break for its callers; E16 and E17 record that it should land as one
deliberate change rather than two. This record is the pattern it should follow.

## Rejected

**Keeping `twiny` with relabelled ticks.** It is what two of the three sites did, so it is
what someone reading those sites will write again. It is rejected on output, not on taste:
it puts the nm labels at 495.9, 550.4 … and it desynchronises from the data on any later
change of limits. There is no configuration of it that fixes either, because both follow
from the mechanism — a twin axes shares nothing with its host but the figure, so its ticks
have to be *told* the mapping once instead of *holding* it.

**Drawing the axis and not returning it.** What all three sites originally did, and the
cheaper change: nothing breaks, the axis still renders. Rejected because the returned
artist *is* this package's styling API. Without it, a caller who wants the top axis in a
smaller font recovers it by walking `fig.axes` and guessing which entry is theirs — and the
pressure to add `twin_labelsize=`, `twin_color=`, `twin_labelpad=` arrives immediately.
That is the failure *parameters earn their place* names: an enumerated style argument is a
symptom of a broken return contract, so the return is fixed first. `plot_current` already
returns its second axes; this makes the two agree.

**Widening the return only when the axis exists.** A 3-tuple with `twin_axis=False` and a
4-tuple with it `True` keeps existing unpackings working. Rejected because the arity would
then depend on an argument's value, so no caller could unpack the result without knowing
what it had asked for, and every downstream helper would need the same branch. Fixed arity
with `None` is what the package already does for `cb` and for
[0020](0020-plot-image-annotates-and-returns-the-circle.md)'s `circle`.

**Deferring the whole thing to E17.** The register already held this defect, so letting
the third copy land and filing it against E17 was available. Rejected because
`plot_spectrum`'s `twin_axis` was unreleased and untested: its tick appearance and its
return arity were both free at that moment and will not be again once anyone depends on
them. The breaking-change
argument that justifies deferring `plot_power_series` did not apply to it.

## Consequences

- `plot_spectrum` returns four elements. Callers unpacking three must add a slot; the
  28 sites in the suite and the one in the README were updated in the same change.
- `twin_axis=True` on `plot_spectrum` produces different ticks than before — round in the
  displayed unit — and the top axis now moves with `set_xlim`.
- Two implementations remain, not three. `plot_power_series` is the last one, and closing
  it is E16 + E17 as a single change.
- `tests/test_x_axis_vocabulary.py` pins the conjugate axis against the vocabulary table
  rather than against literal strings, so a new row in `X_AXES` would be covered by the
  same assertions.

## Load-bearing choices

The claim that one function serves both directions rests on `HC_EV_NM / x` being an
involution. A third spectral axis that is *not* self-inverse — wavenumber, say, for the
Raman work — breaks that, and `_conjugate_x_axis` would then need a real forward/inverse
pair and a table rather than the two-entry `_CONJUGATE_AXIS` flip.
