# Design principles — the reasoning

`.claude/CLAUDE.md` carries these as **rules**, in the shortest form that can be
followed. This file carries the **argument**: why each rule is the rule, the worked
examples, the stated exceptions, and the standing counter-examples in the codebase.

Read it when a rule looks wrong for the case in front of you, or when you are about to
add a parameter, write a docstring paragraph, or make a second copy of something.

Four principles. They overlap deliberately — most bad changes violate two of them.

---

## 1. Corrections are opt-in

> **The researcher looks at the result and decides whether a further correction is
> warranted. The package never makes that decision on their behalf.**

A default must not apply a step that distorts the data or removes a feature from it.
Anything destructive is off until it is switched on deliberately.

The reason is not caution for its own sake. This is analysis software for measurements
whose interesting features are often small, narrow, or unexpected — a weak trion
shoulder, a single hot pixel, a step in a charging curve. A default that smooths,
subtracts or replaces cannot know which of those it just removed, and the researcher
cannot see what is no longer there. So the burden falls where the knowledge is.

### What this governs

**A processing step that alters the data is a parameter set to off/none, not a
behaviour that happens unless suppressed.** `apply_jacobian=False`, background
subtraction only when a background region is supplied, and `median_kernel=1` are all
this one rule, not three separate decisions.

**Off means the least-assuming option, not merely the least code.** For a function
whose whole purpose *is* a destructive correction, the opt-in already happened at the
call site; its parameters then follow the same rule one level down. Default to the
parameterisation that assumes least about the data — the one whose results do not depend
on how the data happened to be batched, and that presumes nothing about what the sweep
axis means.

**Where a permitted default can still destroy a feature, it must say so.** Silent
damage is what this principle actually forbids. `remove_cosmic_rays` keeps
`cross_sweep_veto=False` — conservative, assumption-free, shape-invariant — and **warns**
when a pixel is flagged in most sweeps, because that is precisely the case where the
conservative default is the damaging one. Prefer a warning that names what was affected
over a safer-looking default that hides it.

**Return the evidence.** Masks, flags and fit diagnostics come back to the caller so the
decision can actually be made. See also *raw arrays are never mutated after load*: a
correction produces a new array, and the file's own numbers stay reachable.

**Never move a correction into a loader's default path**, however obviously right it
looks. Loading is not deciding.

### Worked example

`dev/decisions/0001-cosmic-ray-repair-at-load-time.md` is this principle applied end to
end: the correction is opt-in (`cosmic_rays=None`), the *default within it* is chosen for
batch-invariance rather than for aggressiveness, the mask comes back so the repair can be
inspected, `spectra` is never reassigned, and the one case where the safe default does
damage is warned about by name.

### The stated exception

`baseline="constant"` in the `fit_*` functions defaults **on**. It is a model term rather
than a modification of the data — omitting it does not preserve anything, it silently
migrates the pedestal into the fitted amplitude and FWHM. `"none"` is available and
reproduces pre-2026-07 numbers.

Do not "fix" this to `"none"` to comply with the section above.

---

## 2. Parameters earn their place

> **A function exposes the minimum set of parameters its callers cannot readily supply
> themselves. Everything else is not a parameter.**

This bites hardest in `plotting`, where the temptation is one argument per matplotlib
property.

### The test

**Does the argument change the numbers, or only the pixels?** What changes what the data
*is* — which correction ran, which array was plotted, what the axis means — belongs in
the signature. What changes only how it looks does not, because there are already three
better places for it.

### The three better homes

**The returned handles — the first thing to reach for.** `plotting` returns
`(fig, ax, <artist>)`, and that return contract *is* the styling API:
`line.set_color("k")`, `ax.set_xlim(1.6, 1.8)`, `mesh.set_clim(0, 1)` are one line each
at the call site. Never add a parameter whose entire body is
`artist.set_<thing>(value)`.

The corollary matters as much as the rule: **a function that draws several artists must
return them**, or callers have no route to restyle and the parameters grow back.
Enumerated style arguments are a *symptom of a broken return contract* — fix the return
first, and most of the parameters stop being wanted.

**One `**kwargs` passthrough, where a single artist dominates.**
`plot_spectrum(..., **line_kwargs)` forwards to `ax.plot` and therefore supports every
line property matplotlib has, in one parameter, with no docstring to maintain. Keep this
to the one-artist case: a bad key raises from deep inside matplotlib, so a function with
a passthrough per artist is both harder to introspect and worse to debug than one that
returns its artists.

**`set_style()` and rcParams.** Fonts, line widths, spine visibility and DPI are
figure-wide look, set once per session. A `contour_lw=0.9` default silently overrides the
`lines.linewidth` the user just configured — and hardcoding is not the alternative:
`ax.legend(fontsize=5)` in `plot_diffusion_cloud` overrides `set_style`'s
`legend.fontsize` with no way to opt out at all. Style with a sensible global home
belongs in that home.

### What does earn a parameter

- **Corrections and processing** — `median_kernel`, `threshold="1/e"`, `smooth_sigma`,
  `keep_largest`, `bg_stat`, `rescale_img`. These change the numbers, and are governed by
  §1 above.
- **Which data is shown** — `x_axis`, `spectra_source`, `normalize`, and the point
  selectors.
- **Physical context the function cannot infer** — `pixel_scale`, `origin`, `laser_ref`.
  The caller knows the µm/px; the array does not.
- **Composition and structure** — `panels`, `ax`, `n_frames`, `save`. `animate_panels`
  takes a *list of panel objects* rather than a flag per panel type, so any subset, order
  or combination works with no special-casing. That is the shape to aim for: **one
  structural parameter absorbing a combinatorial space.**
- **Axis and colour-bar labels.** A label states what the numbers *are*, so a wrong one
  is a misread rather than an ugly figure. The contract — `None` derives, a string is
  verbatim, nothing is ever appended — is
  `dev/decisions/0011-label-contract-derive-or-verbatim.md`.

### Why this is a library rule and not a matter of taste

**Every parameter is a promise.** It needs a docstring entry, it constrains refactoring,
its default reads to users as a recommendation, and it cannot be withdrawn later without
breaking callers.

Twenty independent booleans imply a million configurations you have implicitly claimed
work and have never once run. A smaller signature is both less to learn and less to keep
honest.

### The boundary case, stated honestly

A style argument that exists so a feature stays **legible** is not decoration.
`laser_halo=True` draws a white halo so the laser circle survives being drawn over a dark
colormap — that is correctness-of-reading, and it stays.

The question to ask: **could the plot be misread without this argument?** If so, it is
not trivial.

### The standing counter-example

`plot_diffusion_cloud` has ~30 parameters, about half of them enumerated styling
(`contour_*`, `centroid_*`, `roi_color`, `bg_region_color`, `laser_*`, `xlabel`/`ylabel`),
and it returns `result` instead of its artists. It predates this rule and is the live
demonstration of the corollary above: the broken return is *why* the signature grew.

New code must not copy it. The planned fix — return the artists, then delete the style
parameters — is E11 in the audit.

---

## 3. Reuse before adding, delete before documenting

> **This package grows by copying and by never deleting, not by over-building.**

The duplication section of the audit is its largest category. Three habits, in order of
how much they cost:

**Search for the concept before writing it.** The second copy is the bug, not merely a
maintenance burden: if a helper is wanted in two modules, the first one is in the wrong
place — move it and import, don't fork it. The live example is the laser-circle drawers
with different styling defaults, of which two remain to unify. That is how 6k lines
becomes 10k.

**Prefer composing an existing entry point over adding a near-duplicate one.**
`animate_wl_pl_spectra` builds its panels and returns `animate_panels(...)` — a new
public function, no new engine. That is the shape to copy. For contrast, `plotting` has
16 public entry points and exactly one such delegation.

**Anything nothing reaches gets deleted, not documented.** A dead parameter with a
docstring entry is worse than no parameter, because it reads as supported. See the whole
*dead parameters* section of the audit, and `fitting.voigt_approx`, which is implemented
and reachable from no `fit_*` function. Status is alpha, pre-adoption: deletion is free
here in a way it will not be later. Take it while it is.

### This is not a licence to write the terse version

Vectorised code with named shapes, returned masks and diagnostics, and error messages
held to a shared-library standard all cost lines deliberately. The target is **less
duplication, not less code.**

---

## 4. A docstring is a contract, not a changelog

> **A docstring describes the thing as it is, to someone who has never seen the source
> and does not know the project has a history.**

mkdocstrings renders these onto the docs site, so the reader has no access to the audit,
no memory of what a function used to be called, and no interest in which design was
rejected. They are deciding whether to call it, and how.

### One test, applied sentence by sentence

> Would this still be true, and still worth reading, if the code had always been this
> way?

If not, it belongs elsewhere. Three homes — the split is about **audience**, not
importance, and nothing is thrown away:

| Text | Home | Read by |
|---|---|---|
| What it does, takes, returns, refuses; units; limits | **docstring** | someone about to call it |
| Why this line is odd — the mechanism, the trap, the measurement | **comment** | someone editing it |
| What was chosen and what was rejected | **`dev/decisions/`** | someone deciding what to do next |
| What is broken, and when it was fixed | **`dev/defects.md`** | same |

Note the third and fourth rows: most displaced text is *not* comment material either.

### What earns a place

- The purpose, in a line that names the thing rather than the change.
- Parameters with units and accepted types; what is returned; what it raises and warns,
  and on what.
- **Limitations and non-goals** — what it does *not* handle, and where to go instead.
  State them flat: *"does not resample; pass a pre-aligned array"* beats a paragraph on
  why resampling would be wrong.
- Constraints on *use* a caller could otherwise get wrong — that a contrast needs a
  matched exposure, that the Jacobian cancels in a ratio. These are the reader's problem,
  so they stay.
- Conventions the array cannot state: axis order, ascending-ness, view vs copy, which
  arrays are never mutated, what unit a bare float is in.
- Examples, where the call shape is not obvious from the signature.

### What does not belong, however true

- Dates, commit hashes, audit IDs, and pointers into `dev/`. A caller cannot follow them
  and is not the audience.
- `was` / `now` / `used to be` / `previously` / `since <date>` / `pre-rename`, and more
  generally any sentence that only parses against what preceded it.
- **Arguing with the design that was not chosen.** *"Why this is a separate class rather
  than a mode of it"* is a decision record. Its **consequence** — which attributes
  therefore do not exist — is documentation. Keep the consequence, move the argument.
- **"Deliberately", "on purpose", "don't add this back".** These answer a maintainer who
  suspects an oversight, so their presence is a reliable signal the sentence is in the
  wrong file. Describe the behaviour and drop the defence.
- Evidence for the implementation — *"verified against the committed pair"*, *"measured
  at 13.8 s against 10.7 s"*.

### One file is not the format

Every figure in the export record came from one particular export, and the next one will
differ — different raster dimensions, different sweep length, possibly a different
acquisition version. So docstrings *and* comments describe the **shape of the thing** and
never present one file's numbers as what a caller should expect.

- **Generalise the mechanism, not the measurement.** *"declares twice as many blocks as
  it writes"* is behaviour, holds across every export seen, and is what the code keys on;
  *"a 2091-point raster exported with 4182 blocks"* is one file's arithmetic. State the
  first, drop the second. Where the prose is more specific than the logic, the prose is
  wrong.
- **Where an illustration genuinely helps, make the numbers obvious stand-ins.** Symbolic
  is best (*"an `n_x` × `n_y` raster gives `n_x·n_y` points"*); round and small is fine
  (*"e.g. 10 × 10 exported as 100 points"*). A figure that matches the committed data
  reads as a specification, not an example — and a doctest whose expected output is one
  file's value is a specification with a test attached.
- **Exact figures from real files belong in the record** — `dev/instruments/`, or the
  audit. There they are dated, attached to the file they came from, and correctable when
  the next acquisition version disagrees.
- **They rot, and silently.** `_read_block_layout` says a raster is 314 MB while the
  comment four lines below it says 300 MB. Nothing failed, no test caught it, and the
  reader now has two facts about a file they do not have.

### Cross-references: does it help the reader act?

`See Also`, *"the maths lives in `processing.spectral_contrast`"*, *"pass this to
`animate_panels`"* are ordinary good documentation: they route someone to the next thing
they need. A reference used to *justify* the code does not.

**Never cite a private helper from a public docstring** — the reader cannot call it, so it
reads as API and dead-ends. Inline the fact instead.

### Worked example

`best_energy_spectra`, before — ten lines, half of them defending a decision:

```
Return the best available energy-axis spectra.
...
**A contrast array is deliberately not returned here**, even when a *reference*
was given.  "Best" means the same physical quantity, better corrected — not a
different quantity.  Contrast is negative-going, so feeding it to
:func:`fit_scan_peak`, whose peak models decay to zero in their wings, would give
quietly meaningless fits; and a PL map's colour bar would silently start meaning
ΔR/R₀.  Ask for :attr:`energy_contrast` explicitly, or ...
```

After — same information a caller needs, none of the argument:

```
Background-corrected energy-axis spectra when available, else uncorrected.

Returns :attr:`energy_spectra_bg` if a background was supplied at load time and
:attr:`energy_spectra` otherwise, so downstream code need not know which.

Never returns the contrast, even when a *reference* was supplied: that is a
different quantity rather than a better-corrected one, and it is negative-going,
which peak fits and intensity colour bars both misread. Use
:attr:`energy_contrast`.
```

The non-goal survives, and so does the one clause a caller needs in order to understand
why they must ask explicitly. What goes is the word *deliberately*, the mechanism of the
fit failure, and the defence of the choice.

---

## Citations

A reference to a thesis, a paper, or an equation **requires a citation a reader can
follow**. An equation number without the document it indexes points nowhere, and a bare
surname with a year is not enough to find a table in a paper.

Where a claim rests on a reference that cannot currently be followed, record the claim as
*inherited group practice* and list what is missing — see `dev/physics-conventions.md`
§9, which does exactly this for three of them.
