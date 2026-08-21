# 0030 — `clim` and `rescale_img` are refused together, in both functions

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-21 |
| **Audit** | E3 |

## Context

`plot_spectral_map` and `plot_image` each take both `clim=(vmin, vmax)` and
`rescale_img`. `clim` is handed to matplotlib as `vmin`/`vmax`, so it is read in the
data's own units. `rescale_img` calls
`skimage.exposure.rescale_intensity(in_range="image", out_range=(0, 1))` **before** the
colour scale is applied, so by the time `clim` is used the data no longer has those
units. The two contradict each other.

Measured on a map whose counts run 503–1497, with `clim=(500, 1500)`:

| | fraction of the colour scale the data spans |
|---|---|
| counts, no rescale | 0.994 |
| rescaled to [0, 1] | 0.001 |

The whole panel draws as one flat colour. Nothing raises, nothing warns, and the figure
does not look like a bad argument — it looks like a measurement that failed.

The same measurement settles what `rescale_img` does *on its own*. With the colour
limits auto-scaled, `pcolormesh` and `imshow` already stretch their colours to the
data's own minimum and maximum, so remapping the data to [0, 1] first and letting it
stretch again is the same stretch twice: the colours drawn are identical, and only the
numbers on the colour bar change, from counts to 0–1. That is a real change — a colour
bar is a claim about what the numbers are — but it is a change to the label and the
scale, not to the picture. `clim` is therefore the only argument through which
`rescale_img` can alter what is drawn, and it alters it destructively.

## Decision

1. Giving both raises `ValueError`, before the figure is made.
2. The message names the function, echoes the `clim` that was given, states the
   mechanism, and names the two ways out: keep `clim` and drop `rescale_img` to set the
   limits in the data's units, or keep `rescale_img` and drop `clim`.
3. One private guard, `_refuse_rescaled_clim(clim, rescale_img, what)`, serves both
   functions.
4. Both docstrings say the pair cannot be given together, on both parameter entries, and
   both carry a `Raises` clause.

## Rejected

**Warn and draw it anyway.** *Corrections are opt-in* says that where a permitted
default can still destroy a feature, the package warns and names what was affected, so
this was the closest alternative and the one that matches existing practice elsewhere in
the package. Rejected because a warning names a *survivable* loss — a clipped peak, a
smoothed shoulder — and there is nothing left here to survive. Every cell is the same
colour: the figure carries no information about the measurement at all, and the two
arguments cannot both be honoured in any reading. A warning would also arrive on stderr
in a notebook, where the flat panel is the thing being looked at.

**Reinterpret `clim` in [0, 1] terms**, rescaling the limits alongside the data. Draws
something sensible and needs no error. Rejected because it silently changes numbers the
caller typed. `clim=(500, 1500)` would become `clim=(0.0, 1.0)` — or, worse, some
partial window — and the colour bar would then be labelled with limits nobody asked for.
It also has no correct answer when the limits lie outside the data's range, which is
exactly when a caller reaches for `clim`: to hold two figures on one scale. Rescaling
already destroys that use, so accommodating it would only hide the fact.

**Delete `rescale_img` instead.** Tempting, since with auto-scaled limits it changes
nothing but the colour-bar numbers, and *parameters earn their place* asks whether an
argument changes the numbers or only the pixels. Rejected because it does change the
numbers — `dev/design-principles.md` lists `rescale_img` among the arguments governed by
*corrections are opt-in*, and the colour bar reads `norm.` rather than `counts` as a
result. A researcher normalising a map for a figure panel is doing something the package
should support.

**A separate guard in each function.** Two call sites, two short `if` blocks, no helper.
Rejected because `colorbar_label` is the recorded example of what happens next: it grew
three incompatible contracts across three functions — appended a unit in
`plot_spectral_map`, was discarded whenever `rescale_img` was set in `plot_image`, used
verbatim in `plot_diffusion_cloud` (`dev/plan-E12.md`, "Why `colorbar_label` stays"). One
guard cannot drift.

## Consequences

- `plot_spectral_map(scan, clim=..., rescale_img=True)` and the same pair on
  `plot_image` now raise. Neither combination could previously produce a usable figure,
  so no working call is broken.
- The guard runs before `plt.subplots()`, so a refused call leaves no orphan figure to
  close. `tests/test_rescale_clim_refusal.py` pins that, not only the message.
- `rescale_img` with no `clim` is documented as changing the colour-bar numbers rather
  than the colours drawn. That is now written where a caller reads it, so the argument no
  longer looks like it does more than it does.
- Anything else in `plotting.py` that gains a rescale and a limit pair goes through the
  same guard rather than growing its own.

## Load-bearing choices

Refusing rather than warning. The argument rests on the loss being *total* — one flat
colour — not merely large. If a future rescale mode were per-sweep rather than global,
so that a `clim` in the original units still left some structure visible, the trade would
be worth re-examining for that mode.
