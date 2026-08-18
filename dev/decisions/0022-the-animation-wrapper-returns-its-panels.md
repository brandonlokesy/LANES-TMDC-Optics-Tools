# 0022 — The animation wrapper forwards one dict to its spectrum panel, and returns its panels

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-18 |

## Context

`animate_wl_pl_spectra` is the convenience path into the animation engine: hand it file
paths, get an animation. It builds up to three panels itself and passes each one a fixed
set of arguments. For the spectrum panel that set was three — `x_axis`, `sweep_attr`,
`sweep_unit` — out of eleven the panel accepts.

`twin_axis` was one of the eight left behind. So the wavelength scale on top of a
spectrum, which [0021](0021-the-conjugate-axis-has-one-implementation.md) had just settled
one implementation for, could not be switched on through the wrapper at all. The only route
was to abandon it, build a `SpectrumLinePanel` by hand and call `animate_panels` directly —
which is the whole thing the wrapper exists to save.

The wrapper also returned only `fig, anim`. The panels were local variables. So even had
`twin_axis` been reachable, the axis it draws would not have been: `SpectrumLinePanel`
exposes it as `panel.ax_twin`, and the panel was unreachable.

The same is true of every other artist a panel owns — `line`, `mappable`, `colorbar`, and
`ImageSequencePanel.laser_circle`. The convenience path could create them and not hand any
of them over.

## Decision

1. **A `spectrum_style` dict is forwarded to the spectrum panel**, normalised from `None`
   to `{}` and splatted into the constructor. It is documented naming the target class,
   with an example dict, exactly as its sibling `laser_style` already is for the two image
   panels.
2. **The wrapper returns `fig, anim, panels`.** `panels` is the list it built, in
   white-light / real-space-PL / spectrum order, with omitted panels absent rather than
   `None` — so its length follows which arguments were given, and the spectrum panel is
   `panels[-1]` whenever `spectra` was passed. The docstring says so, because a
   variable-length list is only usable if the order is stated.

## Rejected

**An explicit `twin_axis=` parameter on the wrapper.** The narrowest change, and the one
the problem seems to ask for. Rejected because it answers one of eight identical questions.
The next person wanting `cmap` on the spectrum panel, or `ylabel`, or `show_sweep_title`,
has the same argument available and no reason to hear no — and the wrapper ends up
re-declaring the panel's constructor one parameter at a time, each needing its own
docstring entry, each able to drift from the panel's own default. *Parameters earn their
place* asks what the caller cannot readily supply; the answer here is *the whole panel*,
not this one flag.

**A second `**kwargs` splat, forwarding to the panel.** `dev/design-principles.md` allows
one `**kwargs` passthrough per function, and `**engine_kwargs` already holds that slot
here — it forwards to `animate_panels`. Two splats in one signature cannot be told apart at
the call site: a misspelt key would land in whichever one caught it and die deep inside the
other function's frame. Named dicts avoid that, which is why this package already has
three of them (`laser_style`, `laser_ref_kwargs`, `cosmic_rays`).

**Validating the dict's keys against the panel's signature.** `loaders`' `cosmic_rays` does
this, with `inspect.signature`, and raises naming the target function. Rejected as
unnecessary here: `SpectrumLinePanel(**{"twin_axes": True})` already raises a `TypeError`
that names the bad keyword and the constructor that rejected it, which is the same
information one step later. The `cosmic_rays` check exists because that dict is stored on a
loader and consumed much later, so the error would otherwise surface far from the
declaration. This one is consumed on the next line.

**Keeping the two-element return.** No caller would have broken; three tests and two
notebook cells needed a slot added. Rejected on the argument [0021](0021-the-conjugate-axis-has-one-implementation.md)
already made and this change would otherwise contradict: a function that draws an artist
returns it, because the returned artist *is* the styling API. Opening `twin_axis` while
leaving `ax_twin` unreachable would deliver the switch and withhold the reason for having
it, and would bring back the pressure for `twin_labelsize=` and its relatives — as
wrapper-level parameters this time, which is worse.

## Consequences

- `animate_wl_pl_spectra` returns three elements. Callers unpacking two must add a slot.
- Every `SpectrumLinePanel` option is reachable from the convenience path, and every panel
  artist with it.
- `laser_style` gained its first test. It had none, so the pattern being copied was
  unpinned.
- `dev/plan-E12.md`'s proposal to delete `sweep_attr` / `sweep_label` / `sweep_unit` from
  the panel gets simpler rather than harder: those three keys move into `spectrum_style`
  instead of needing three parameters deleted from two places.
- Still not reachable: `bg_region`. It belongs to the loader, not the panel, so
  `spectrum_style` does not open it and `dev/defects.md` keeps that entry open.

## Load-bearing choices

Returning a list whose length varies with the arguments is the part most worth revisiting.
It is honest about what was built, and `panels[-1]` is a stable handle on the spectrum
panel, but a caller wanting the PL panel specifically has to know whether white light was
passed. A named tuple or a small object with `wl` / `pl` / `spectrum` slots would answer
that; it was not chosen because it introduces a return type for one function's convenience,
and because `animate_panels` takes a plain list, so a plain list keeps the two agreeing.
