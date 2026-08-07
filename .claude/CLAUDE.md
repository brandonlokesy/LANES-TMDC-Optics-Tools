# tmdc_optics_tools

## What this is

A Python toolkit for TMDC optoelectronics/photonics measurements, developed in the
LANES research group. Maintained by one person (Brandon), but written to become the
standard analysis workflow for the group (~15 people max) — so error messages,
defaults, and docstrings are held to a shared-library standard, not a personal-script
one.

**Samples:** TMDCs in FET structures (bottom-, top-, dual-gated) or in cavities.

**Measurements the group makes:** photoluminescence, micro-PL, reflectance /
differential reflectance, absorption, real-space PL imaging (exciton diffusion
clouds), back-focal-plane measurements (k-space dispersion) as real images on CCDs,
Raman spectroscopy.

**Measurements this package currently supports (as of 2026-08-06):** PL,
reflectance / reflectance contrast, time-resolved PL, real-space imaging, and
Raman — both single-spectrum (`RamanSpectrum`) and 2-D spatial maps
(`RamanMap`), both LabRAM `.txt` exports with different row shapes — see
`examples/example-Raman.ipynb`. Absorption, cavity, and BFP/k-space data are
measured in the lab but have no loader. Don't propose speculative
multi-modality abstractions unasked
— but do flag where a PL assumption will resist a future loader (the hardcoded
"PL intensity" strings in `plotting.py` are the live example; `scan.signal_label`
and `scan.contrast_label` exist for them — see E12).

## The AttoCube export format

Established from real files on 2026-07-30, which closed most of E9. **This is the
record — don't re-derive it by inference.**

One column *block* per sweep point, after a `"Parameters Labels"` label column.
Two block layouts, told apart by the header's field names (`_read_block_layout`):

| Layout | Block | Used by |
|---|---|---|
| spectral | `[Par_i, Wavelength{i}, ExpROI1_{i}, ExpROI2_{i}]` | PL, R, RC |
| temporal | `[Par_i, Wavelength{i}, Exp_{i}]` | TRPL |

- **In the temporal layout, the column named "Wavelength" holds TIME** (ns, 4 ps
  bins, ~12.8 ns range). An acquisition-software misnomer. Read it as time; do
  not "fix" the name in the file.
- **`ExpROI1`/`ExpROI2` are two spatial ROIs on the CCD** — the excitation spot
  and a remote, spatially-filtered spot, for two-spot galvo scans. `ExpROI2` is
  identically zero in every other measurement, which is why the loader warns when
  the *selected* ROI is all zeros. Both are always loaded; `roi=` only chooses
  which one `spectra` points at.
- **The 57 labelled parameter rows are format-fixed.** Verified identical across
  PL, R, TRPL and the TRPL companion. A missing row therefore means a different
  acquisition version, not routine variation — so the permissive load (E1) makes
  its error *diagnostic*, not merely tolerant.
- **The exporter over-allocates and zero-fills.** A 2091-point reflectance raster
  is exported with 4182 declared blocks, the surplus half filled with literal
  `0.0` in every field — numeric, not empty, so no NaN strip removes them.
  `_drop_unwritten_blocks` drops them using *the axis column being identically
  zero* as the sentinel, and reports the count via `n_declared_sweeps`. This is
  decoding, not a correction: keeping them fabricates measurements never taken.
- **It over-allocates the row width too, and that padding is empty, not zero.**
  Two separate over-allocations, easy to conflate. Beyond the 4182 *named* blocks
  (the last named field is `ExpROI2_4181`, so the named width is
  `1 + 4182×4 = 16729` fields), every row carries a further `4182×4 = 16728`
  unnamed, **empty** fields — 33457 in total, exactly twice the named width minus
  the label column. So the surplus *blocks* hold `0.0`, while the trailing
  *padding* holds nothing at all. Nothing needs to strip it: `_read_block_layout`
  counts columns matching `^Par_?\d+$`, and an empty field cannot match, so the
  block count comes out at 4182 regardless of how wide the row is.
- **`R`/`RC` need no parser work** — reflectance uses the identical spectral
  layout as PL. What it needs is a reference spectrum, which is a 2-row
  `SingleSpectrum` CSV of the bare substrate.
- **A TRPL sweep is a directory**, one file per point, each carrying its own full
  57-row parameter snapshot, plus a **metadata companion**: a *spectral*-layout
  file whose `Par_i` columns hold one snapshot per point and whose
  Wavelength/ROI columns are identically zero. The companion collides on
  `iter_0` with the first data file and is written last, so classify by
  **content, not filename**. Order data files by the integer in `_iter_N`;
  lexicographic order puts `iter_10` before `iter_2`.
- **A 2-D spatial raster is one flattened file** (41 X inside 51 Y for the
  reflectance example). `sweep_grid()` detects and reports the shape; it does
  **not** reshape — that is still open work.

Still unknown, and no file can answer it: which acquisition software and version
emits this, and whether the layout is version-stable.

**Status:** alpha, pre-adoption. Renaming and signature changes are acceptable;
prefer fixing a name now over carrying a compatibility shim. (`DeviceGeometry.from_single`
is labelled "backward compatibility" but predates any real user base.)

## Environment

```
Interpreter : C:\Users\Brandon\anaconda3\envs\viz-sci-plot\python.exe
Install     : pip install -e ".[docs]"
Docs        : python -m mkdocs build --strict
Tests       : python -m pytest -q
```

`pytest` **is installed** in `viz-sci-plot` (9.1.1, confirmed 2026-07-30 — the whole
suite runs) but is **not declared** in `pyproject.toml`: there is no `test` extra, so
the dependency exists only in this one environment. Tests are local-only by
deliberate choice — do not add a test job to CI. CI
(`.github/workflows/docs.yml`) builds and deploys docs only. Declaring the extra is
part of F1 in `dev/audit-2026-07.md` and is deferred, not forgotten.

TODO: `pyproject` says `requires-python = ">=3.9"` but the docs CI uses 3.12.
Ask which is authoritative before relying on version-specific syntax.

## Physics conventions

**Displacement field** — `DeviceGeometry.electric_field` is **exact as written**
(since 2026-07-30); do not "simplify" it:

```
ε_2D · E_2D = ε_stack · E_stack,   E_stack = (V_BG − V_TG) / d_tot
⇒ E_2D = (V_BG − V_TG) / d_tot · (ε_stack / ε_2D)
d_tot = d_2D + d_hBN,top + d_hBN,bottom
```

Exact because **D** is continuous with no free charge between the gates, so
`ε_i·E_i` is the same in every slab, and `ε_stack` (`DeviceGeometry.eps_stack`) is
by construction the ε of the homogeneous slab of thickness `d_tot` with the same
capacitance as the real stack. `ε_2D` is the TMDC-only series-capacitor value
(`DeviceGeometry.eps_2d`). Result is in mV/nm.

Two wrong forms will look tempting; both are recorded because both have already
been shipped somewhere in the group:

- **`ε_hBN` in place of `ε_stack`** (numerator) is the thin-TMDC approximation —
  *"pretend the whole stack is hBN"*. Low by `(d_2D/d_hBN)(1 − ε_hBN/ε_2D)`, i.e.
  0.59% for 53/46 nm hBN around a MoSe₂/WSe₂ bilayer, growing for thicker TMDC
  stacks or thinner hBN. This is eq. 3.13 of the senior's thesis, what the group's
  MATLAB scripts compute, and what this function did before 2026-07-30 — so current
  fields are ~0.6% higher than older results. Not wrong, just not exact.
- **`ε_stack` in the denominator** instead of the numerator is wrong by ~1.8×, and
  is *not* an approximation of anything. The old MATLAB reaches the correct answer
  only because its `eps_hs` line, `2*t*e_hbn*e_tmdc/(2*t*e_hbn)`, cancels
  algebraically to `eps_tmdc` for any input. Repairing that cancellation alone takes
  you from 0.6% low to 82% high.

Thesis eq. 3.12 keeps the `+ d_2D` term but substitutes `d_TOT` for `d_hBN`, which
counts the TMDC twice; it is 1.29% low, i.e. *worse* than the 3.13 that follows it.
Keeping `d_hBN` there instead makes 3.12 exact and equal to the form above.

**Jacobian** — `apply_jacobian` defaults to `False` and that is intended. The
docstrings and README still claim "True (default)"; the *docs* are wrong, not the
code. When the Jacobian is applied, background must be subtracted in wavelength
space **first**, because a flat pedestal `B` becomes `B·λ²/hc` — curved, not flat —
in energy space. The loader already does this in the right order.

**Reflectance contrast** — `ΔR/R₀ = (S − R)/R` against a bare-substrate
reference (`processing.spectral_contrast`, wired as `reference=` on the loader).
Two things about it are easy to get wrong, both recorded because both change the
numbers:

- **The Jacobian cancels in a ratio.** `(S·λ²/hc)/(R·λ²/hc) = S/R` exactly, so
  `energy_contrast` is built with the Jacobian **off** regardless of
  `apply_jacobian`, and applying it to the numerator alone would be an error.
  Verified numerically against the committed reflectance pair.
- **Sample and reference must share an exposure.** For a reference scaled by `k`,
  `(S − kR)/(kR)` is a *biased* contrast, not a rescaled one — so a reference
  taken at a different integration time or excitation power gives a wrong answer
  that no later normalisation repairs. A 2-row reference CSV carries **no
  parameter rows**, so the package cannot check this and cannot correct it:
  matching the acquisition, or supplying the ratio via `reference_scale=`, is the
  caller's responsibility. Same shape of problem as gate polarity. (Precedent:
  `reference/processors/Dijkstra2025.py:162` divides by integration time first.)

Background comes off both arrays *before* the ratio — a pedestal in either biases
a contrast non-linearly. Grid mismatch **raises**; interpolating would change the
numbers and smooth the data, so it is a correction and cannot be a default. Pass
a pre-aligned bare array if you have resampled it yourself.

**TRPL time axis** — ns, 4 ps bins. Consistent with the Picoharp rows and a
~78 MHz rep rate, but **not independently confirmed**; `_TRPL_TIME_UNIT` is the
single place to change it. Any fitted lifetime inherits this assumption. The
per-file time axes are *not* bit-identical (bin width varies in its seventh
figure), so assembly compares with a tolerance, never equality.

**Fit baselines** — peak models decay to zero in their wings, so an un-subtracted
dark-count pedestal is otherwise absorbed by inflating amplitude and FWHM. All
`fit_*` functions therefore take `baseline={"constant"|"linear"|"none"}`, default
`"constant"`. `"none"` reproduces pre-2026-07 numbers. A Lorentzian's 1/x² wings are
partly degenerate with a flat offset, so FWHM is more window-sensitive with a
baseline than without; the centre is set by symmetry and stays robust.

**Raw arrays are never mutated after load.** `scan.spectra` is the untouched file
contents; corrections produce new arrays (`energy_spectra`, `energy_spectra_bg`, …).

TODO — provenance not yet recorded, ask before documenting or changing:
- `power_scale = 0.303e6` ("calibrated by CdG") — when, which objective/filter set,
  does it vary per session?
- `Scanner X` / `Scanner Y` units (code says "assumed µm, scale 1.0 until confirmed").
- `EPS_TMDC["HS"] = 7.5` — unsourced, sits among per-material values. Note it is
  *not* the harmonic mean of a MoSe₂/WSe₂ bilayer (that is 7.299), and it exceeds
  both constituents, so it is not an average of the values above it. Also note that
  this key uses "HS" for the **TMDC-only** quantity, the opposite scope from the
  former `DeviceGeometry.eps_hs` — which is why those properties were renamed on
  2026-07-30. Don't reintroduce bare "hs".
- `T_MONOLAYER` = 0.65 nm for all four materials — deliberate approximation?
- `EPS_HBN = 3.9` is cited to Laturia et al. 2018, and the four TMDC values do match
  that paper's bulk out-of-plane figures — but its hBN out-of-plane value is usually
  quoted as **3.76**, and 3.9 is also the canonical SiO₂ value. Worth checking
  against the table: it propagates into `eps_stack`, though only weakly into
  `electric_field` now that the exact form is used.
- **Gate polarity is per-session wiring, not a property of the code.** The
  electrodes can be hooked up in either configuration, so which acquisition channel
  drove which gate is *not* inferable and must be recorded per scan; the old MATLAB
  used `dm(4,…)`/`dm(6,…)` with no note of which was which.
  `electric_field(v_top=, v_bot=)` puts the mapping on the caller by design —
  transposing them mirrors the field axis and flips the sign of any extracted
  dipole. The senior's thesis is itself inconsistent here (eq. 3.11 carries a
  leading minus, 3.12/3.13 do not), so don't inherit its sign.
  *Partly addressed 2026-07-30:* `AttoCubeSpectralSweep` now **records** the
  mapping — `curated_labels={"v_top": …, "v_bot": …}`, printed in `__repr__` as
  `(top ← 'V_A', bottom ← 'V_B')` and written into exported HDF5 — and
  `gate_mode` reports whether the two gates were driven anti-correlated
  (field-like) or correlated (doping-like). What is recorded is still the
  *channel-to-argument* mapping, not the physical wiring: the default remains
  `V_A`→top, `V_B`→bottom, which is a convention no file confirms. Ask per
  session; don't add a sign to the physics.

**Raman — WSe₂ bilayer and monolayer example modes.** `examples/data/Raman/*.txt`
(LabRAM export, loaded by `RamanSpectrum`) is WSe₂, bilayer or monolayer per the
filename (`*bilayer*` / `*monolayer*`) and per session identification — the file
headers carry no material or layer-count field, so neither is independently
checkable from the data alone. Fitting the six example spectra
(`unstrained_bilayer`, `strained_bilayer1`, `strained_bilayer2`,
`unstrained_monolayer`, `strained_monolayer1`, `strained_monolayer2`) with
`fitting.fit_raman_modes(..., material="WSe2", n_layers=2)` /
`fit_raman_modes(..., n_layers=1)` (see `examples/example-Raman.ipynb`)
consistently finds modes matching Pan et al. 2022 (*"Signature of lattice
dynamics in twisted 2D homo/hetero-bilayers"*, 2D Materials 9, 045018,
doi:10.1088/2053-1583/ac83d4). The mode identities, seed positions, and fit
tolerances behind that call live in `constants.RAMAN_MODES["WSe2"]` (per
layer count) rather than being hardcoded in the fitting function — the
values there are what to change for a different material or layer count,
not `fitting.py` itself; `constants.RAMAN_LAYER_DISCRIMINATOR["WSe2"]`
holds the matching data for `fitting.classify_raman_layer`, used on the
map below where the layer count is not known ahead of time:

| Fitted here (bilayer) | Fitted here (monolayer) | Literature | Assignment |
|---|---|---|---|
| ≈250.5–250.6 cm⁻¹ | ≈250.1 cm⁻¹ | ≈250 cm⁻¹ | E₂g/A₁g, nearly degenerate |
| ≈258.6–258.8 cm⁻¹ | ≈260.3–260.5 cm⁻¹ | ≈260 cm⁻¹ | 2LA(M), second-order double-resonance |
| ≈309–309.3 cm⁻¹ | **absent** | ≈309 cm⁻¹ | B₂g |

E₂g and A₁g being *nearly* degenerate is exactly why they don't split into two
resolvable peaks here — treating ≈250 cm⁻¹ as a splittable doublet (an
assumption tried and rejected during fitting, not sourced from this paper) was
the wrong model going in. The weak shoulder is not "the other half of a
doublet" either: 2LA(M) is second-order (double-resonance), a different
scattering mechanism from the first-order E₂g/A₁g and B₂g modes, which is why
it is ~10× weaker and why every attempt to seed it near ≈250–253 cm⁻¹ (as if
it were doublet-adjacent) either pinned at a fit bound or failed to converge —
only seeding near its actual position converges cleanly. That position was
found from data first in both materials (the residual of a main-peak-only fit,
via `fitting.locate_residual_peak`), and matches this paper's ≈260 cm⁻¹ 2LA(M)
value after the fact — it is not assumed equal between bilayer and monolayer
just because the mode has the same name in both (≈258.7 vs. ≈260.4 cm⁻¹, a real
difference, not fit noise).

B₂g is absent in every monolayer spectrum checked — flat baseline at ≈309 cm⁻¹,
not a small/unresolved peak — consistent with B₂g requiring interlayer coupling
that a single layer does not have.
`constants.RAMAN_MODES["WSe2"][1]["modes"]` therefore genuinely lists only two
modes; do not "fix" it to include B₂g and expect `fitting.fit_raman_modes` to
drop it — see that function's docstring.

## Design principle — corrections are opt-in

**The researcher looks at the result and decides whether a further correction is
warranted. The package never makes that decision on their behalf.** A default must
not apply a step that distorts the data or removes a feature from it. Anything
destructive is off until it is switched on deliberately.

This governs what defaults are allowed to be:

- A processing step that alters the data is a **parameter set to off/none**, not a
  behaviour that happens unless suppressed. `apply_jacobian=False`, background
  subtraction only when `bg_region` is supplied, and `median_kernel=1` are all this
  rule, not separate decisions.
- **Off means the least-assuming option, not merely the least code.** For a
  function whose whole purpose *is* a destructive correction, the opt-in already
  happened at the call site; its parameters then follow the same rule one level
  down. Default to the parameterisation that assumes least about the data — the one
  whose results do not depend on how the data happened to be batched, and that
  presumes nothing about what the sweep axis means.
- **Where a permitted default can still destroy a feature, it must say so.** Silent
  damage is the thing this principle actually forbids: a researcher cannot look at
  a feature that has been removed without trace. `remove_cosmic_rays` keeps
  `cross_sweep_veto=False` — conservative, assumption-free, shape-invariant — and
  warns when a pixel is flagged in most sweeps, because that is the case where the
  conservative default is the damaging one. Prefer a warning that names what was
  affected over a safer-looking default that hides it.
- **Return the evidence.** Masks, flags, and fit diagnostics come back to the caller
  so the decision can actually be made. Cf. raw arrays are never mutated after load.
- Never move a correction into a loader's default path, however obviously right it
  looks. Loading is not deciding.

**Stated exception:** `baseline="constant"` in the `fit_*` functions. It defaults
*on* because it is a model term rather than a modification of the data — omitting it
does not preserve anything, it silently migrates the pedestal into amplitude and
FWHM. Do not "fix" it to `"none"` to comply with this section.

## Design principle — parameters earn their place

**A function exposes the minimum set of parameters its callers cannot readily supply
themselves. Everything else is not a parameter.** This bites hardest in `plotting`,
where the temptation is one argument per matplotlib property.

The test is whether an argument changes **the numbers or only the pixels.** What
changes what the data *is* — which correction ran, which array was plotted, what the
axis means — belongs in the signature. What changes only how it looks does not,
because there are already three better places for it:

- **The returned handles — the first thing to reach for.** `plotting` returns
  `(fig, ax, <artist>)`, and that return contract *is* the styling API:
  `line.set_color("k")`, `ax.set_xlim(1.6, 1.8)`, `mesh.set_clim(0, 1)` are one line
  each at the call site. Never add a parameter whose entire body is
  `artist.set_<thing>(value)`. The corollary matters as much as the rule: **a
  function that draws several artists must return them**, or callers have no route to
  restyle and the parameters grow back. Enumerated style arguments are a symptom of a
  broken return contract — fix the return first.
- **One `**kwargs` passthrough, where a single artist dominates.**
  `plot_spectrum(..., **line_kwargs)` forwards to `ax.plot` and so supports every
  line property matplotlib has, in one parameter, with no docstring to maintain. Keep
  this to the one-artist case: a bad key raises from deep inside matplotlib, so a
  function with a passthrough per artist is both harder to introspect and worse to
  debug than one that returns its artists.
- **`set_style()` and rcParams.** Fonts, line widths, spine visibility, and DPI are
  figure-wide look, set once per session. A `contour_lw=0.9` default silently
  overrides the `lines.linewidth` the user just configured — and hardcoding is not
  the alternative: `ax.legend(fontsize=5)` in `plot_diffusion_cloud` overrides
  `set_style`'s `legend.fontsize` with no way to opt out at all. Style with a
  sensible global home belongs in that home.

What does earn a parameter:

- **Corrections and processing** — `median_kernel`, `threshold="1/e"`, `smooth_sigma`,
  `keep_largest`, `bg_stat`, `rescale_img`. These change the numbers, and are governed
  by *corrections are opt-in* above.
- **Which data is shown** — `x_axis`, `spectra_source`, `sweep_index`, `normalize`.
- **Physical context the function cannot infer** — `pixel_scale`, `origin`,
  `laser_ref`. The caller knows the µm/px; the array does not.
- **Composition and structure** — `panels`, `ax`, `n_frames`, `save`.
  `animate_panels` takes a list of panel objects rather than a flag per panel type,
  so any subset, order, or combination works with no special-casing. That is the
  shape to aim for: one structural parameter absorbing a combinatorial space.

Why this is a library rule and not a matter of taste: **every parameter is a
promise.** It needs a docstring entry, it constrains refactoring, its default reads
to users as a recommendation, and it cannot be withdrawn later without breaking
callers. Twenty independent booleans imply a million configurations you have
implicitly claimed work and have never once run. A smaller signature is both less to
learn and less to keep honest.

The boundary case, stated honestly: a style argument that exists so a feature stays
**legible** is not decoration. `laser_halo=True` draws a white halo so the laser
circle survives being drawn over a dark colormap — that is correctness-of-reading,
and it stays. Ask whether the plot could be *misread* without the argument; if so, it
is not trivial.

`plot_diffusion_cloud` is the standing counter-example — ~30 parameters, about half
of them enumerated styling, and it returns `result` instead of its artists. It
predates this rule. New code should not copy it; see *Decided but not yet
implemented*.

## Design principle — reuse before adding, delete before documenting

This package grows by **copying and by never deleting**, not by over-building. The
`D` section of `dev/audit-2026-07.md` is its largest category. Three habits, in
order of how much they cost:

- **Search for the concept before writing it.** The second copy is the bug, not
  merely a maintenance burden: if a helper is wanted in two modules, the first one is
  in the wrong place — move it and import, don't fork it. `_draw_region_box` exists
  verbatim in both `processing.py` and `diffusion.py` (D1), and there are three
  laser-circle drawers with different styling defaults (D2). That is how 6k lines
  becomes 10k.
- **Prefer composing an existing entry point over adding a near-duplicate one.**
  `animate_wl_pl_spectra` builds its panels and returns `animate_panels(...)` — a new
  public function, no new engine. That is the shape to copy. For contrast,
  `plotting.py` currently has 16 public entry points and exactly one such delegation.
- **Anything nothing reaches gets deleted, not documented.** A dead parameter with a
  docstring entry is worse than no parameter, because it reads as supported —
  cf. the whole `B` section, and `fitting.voigt_approx`, which is implemented and
  reachable from no `fit_*`. Status is alpha, pre-adoption: deletion is free here in
  a way it will not be later. Take it while it is.

**This is not a licence to write the terse version.** Vectorised code with named
shapes, returned masks and diagnostics, and error messages held to a shared-library
standard all cost lines deliberately — see *vectorised NumPy is wanted, but never
silent* below and *return the evidence* above. The target is less duplication, not
less code.

## Design principle — a docstring is a contract, not a changelog

**A docstring describes the thing as it is, to someone who has never seen the source
and does not know the project has a history.** mkdocstrings renders these onto the
docs site, so the reader has no access to the audit, no memory of what a function
used to be called, and no interest in which design was rejected. They are deciding
whether to call it and how.

One test, applied sentence by sentence:

> Would this still be true, and still worth reading, if the code had always been
> this way?

If not, it belongs elsewhere. Three homes — the split is about **audience**, not
importance, and nothing is being thrown away:

| Text | Home | Read by |
|---|---|---|
| What it does, takes, returns, refuses; units; limits | **docstring** | someone about to call it |
| Why this line is odd — the mechanism, the trap, the measurement | **comment** | someone editing it |
| What changed and when, why, what was rejected | **`dev/audit-2026-07.md`**, or this file for standing conventions | someone deciding what to do next |

Note the third row: most displaced text is *not* comment material either. This
project already keeps decision records properly, and the audit's
**[FIXED — date, commit]** discipline is where "what changed" lives. Moving history
from a docstring into a comment two lines below has not fixed anything.

**What earns a place**

- The purpose, in a line that names the thing rather than the change.
- Parameters with units and accepted types; what is returned; what it raises and
  warns, and on what.
- **Limitations and non-goals** — what it does *not* handle, and where to go
  instead. State them flat: *"does not resample; pass a pre-aligned array"* beats a
  paragraph on why resampling would be wrong.
- Constraints on *use* a caller could otherwise get wrong — that a contrast needs a
  matched exposure, that the Jacobian cancels in a ratio. These are the reader's
  problem, so they stay.
- Conventions the array cannot state: axis order, ascending-ness, view vs copy,
  which arrays are never mutated, what unit a bare float is in.
- Examples, where the call shape is not obvious from the signature.

**What does not belong, however true**

- Dates, commit hashes, audit IDs (`E9`, `A6`), and pointers into `dev/` or this
  file. A caller cannot follow them and is not the audience.
- `was` / `now` / `used to be` / `previously` / `since <date>` / `pre-rename`, and
  more generally any sentence that only parses against what preceded it.
- **Arguing with the design that was not chosen.** *"Why this is a separate class
  rather than a mode of it"* is a decision record. Its **consequence** — which
  attributes therefore do not exist — is documentation. Keep the consequence,
  move the argument.
- **"Deliberately", "on purpose", "don't add this back".** These answer a
  maintainer who suspects an oversight, so their presence is a reliable signal the
  sentence is in the wrong file. Describe the behaviour and drop the defence.
- Evidence for the implementation — *"verified against the committed pair"*,
  *"measured at 13.8 s against 10.7 s"*. Comment or audit.

**Cross-references: does it help the reader act?**

`See Also`, *"the maths lives in `processing.spectral_contrast`"*, *"pass this to
`animate_panels`"* are ordinary good documentation: they route someone to the next
thing they need. A reference used to *justify* the code (*"as `_order_by_iter` now
does"*) does not. And **never cite a private helper from a public docstring** — the
reader cannot call it, so it reads as API and dead-ends. Inline the fact instead.

**Worked example.** `best_energy_spectra`, before — ten lines, half of them
defending a decision:

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

The non-goal survives, and so does the one clause a caller needs to understand why
they must ask explicitly. What goes is the word *deliberately*, the mechanism of the
fit failure, and the defence of the choice — which is now in *Known issues* where a
maintainer will look for it.

## Code conventions

- NumPy-style docstrings (rendered by mkdocstrings); aligned-colon parameter blocks.
  What goes in one, and what belongs in a comment or the audit instead, is *a
  docstring is a contract, not a changelog* above.
- Relative imports within the package (`from . import processing`).
- `loaders` = I/O + geometry; `processing` = pure array functions; `fitting` returns
  dataclasses; `plotting` returns `(fig, ax, <artist>)` and never calls `plt.show()`.
  The artist comes back so that styling can stay out of the signature — a function
  that draws several returns all of them. See *parameters earn their place* above.
- Plotting must not re-implement maths that belongs in `processing`.
- New module → add `docs/api/<module>.md` with a `:::` directive **and** a `nav`
  entry in `mkdocs.yml`; `mkdocs build --strict` must stay green.

**Vectorised NumPy is wanted, but never silent.** Broadcasting, fancy indexing,
boolean masks, and comprehensions are preferred over explicit Python loops — write
the fast version, don't hand-roll a loop to be kind. The condition is that a reader
must never have to re-derive the trick from the expression. Name the shapes and say
what the operation is doing:

```python
# (n_pixels, 1) broadcast against (n_pixels, n_sweeps): one baseline per pixel,
# subtracted from every sweep.
corrected = spectra - baseline[:, None]

# Median along the sweep axis only — the (1, window) footprint means no
# information is ever mixed between detector pixels.
local_med = median_filter(spectra, size=(1, window))
```

A comprehension gets the same treatment: say what it is collecting and over what.
The bar is that someone reading it six months later can see the intent without
running it. Explain the same thing in prose in the reply that introduces it —
efficient code is fine, unexplained code is not.

## Guardrails

- Never run `reference/registry.py` — it downloads from Zenodo and mediaTUM FTP.
- Don't commit `.h5` files (gitignored); don't regenerate `examples/data/`.
  That data lives in the repo deliberately, to reproduce the example notebooks.
- Don't touch `site/` (mkdocs build output).

## Known issues — check before "helpfully" fixing

Full audit with fix sketches: `dev/audit-2026-07.md`

**Deferred by explicit decision — leave alone:**
- `diffusion._binary_area` has a wrong MATLAB `bwarea` weight table (diagonal pairs
  0.5, should be 0.75; 3-pixel patterns 0.75, should be 0.875), biasing cloud areas
  low. Parked pending a decision on whether bwarea semantics are wanted at all.
  *(The `AttoCubePLVabScan` `_CURATED` fail-fast that used to sit here was fixed
  by the 2026-07-30 rewrite — see below.)*

**Fixed — don't re-report:**
- `processing.remove_cosmic_rays` (A1, fixed 2026-07-28). Was uncallable
  (`UnboundLocalError`), and its iteration was a no-op. Now works, takes 2-D
  `(n_pixels, n_sweeps)` input, and has `cross_sweep_veto=` plus a persistent-flag
  warning. Covered by `tests/test_processing_cosmic_rays.py`.
- `DeviceGeometry.eps_hs` / `__repr__` (A2, fixed 2026-07-30). `eps_hs` iterated
  `_slabs()`'s `(d, ε)` tuples as `StackLayer` objects, so `print(geom)` raised for
  every geometry. Now unpacks the tuples — `_slabs()` returning tuples is the
  convention, not something to change: hBN gate flakes are ~50 nm, not *n*
  monolayers, and hBN's ε lives in `EPS_HBN`, not `EPS_TMDC`, so wrapping them in
  `StackLayer` would require asserting a false `n_layers`.
- `DeviceGeometry.optical_thickness` **deleted** (A2, 2026-07-30). It returned
  `d_tot × ε_2D` — full-stack thickness times a TMDC-only ε — under a local
  variable misleadingly named `d_2d`. It was neither an optical path length
  (`Σ nᵢdᵢ`, which these static out-of-plane ε values cannot give: TMDC in-plane
  `n ≈ 4–5` near resonance against `√7.2 ≈ 2.7`) nor any capacitive thickness
  (capacitance goes as `ε/d`; `d/ε` is what adds in series). It was a fragment of
  the pre-2026-07-30 `electric_field` denominator with `ε_hBN` not yet divided out,
  had no callers, and the name would have actively misled the planned
  reflectance/cavity work. Do not reinstate it — the future optics code needs
  dispersive `n(λ)` data of its own, not `EPS_TMDC`.
- Laser circle in `animate_real_space_PL_map` (A3, fixed 2026-07-30). Passed the
  `matplotlib.patheffects` *module* to `set_path_effects`, which raises
  `AttributeError` at **draw** time, not at call time. The hand-rolled block was
  replaced by `_draw_laser_circle(ax, scan.laser_ref, ls="--")`, so two drawers now
  remain for D2 rather than three. New code annotating a laser spot should call that
  helper, not build a `patches.Circle`. Covered by
  `tests/test_plotting_laser_circle.py` — the suite's first plotting tests, which
  force the `Agg` backend and assert on a real `fig.canvas.draw()`. Any future test
  of an artist's styling must render too; building the figure proves nothing.

- **`AttoCubePLVabScan` → `AttoCubeSpectralSweep`** (2026-07-30). Renamed and
  rewritten; the old name survives as a deprecated subclass that emits
  `FutureWarning` (**not** `DeprecationWarning` — Python filters that out by
  default outside `__main__`, so a library raising one warns nobody). What
  changed, and what to not re-litigate:
  - `spectra_type=` is **required, keyword-only, no default**. It is written into
    exported metadata and trusted thereafter, so a default would let a guess
    outlive the session. Use `scan.signal_label` instead of hardcoding "PL".
  - One `sweep=` argument takes a `_SWEEP_TYPES` key *or* any raw CSV row label.
    Undeclared → the sweep axis is the sweep **index**, never an auto-detected
    parameter: mislabelling an axis is worse than not labelling one, and
    `V_A`+`V_B` both varying is ambiguous between a field sweep and independent
    gating. `varying_parameters()` and `gate_mode` return the evidence instead.
  - `gate_axis` / `gate_axis_label` are kept as aliases of `sweep_axis` /
    `sweep_axis_label` so existing plotting keeps working. Don't delete them
    without updating `plot_pl_map_Vab_scan`.
  - E1 fixed: **no curated row is mandatory.** A file missing `Scanner X` loads;
    the property raises only if accessed. The one remaining fail-fast is the row
    the *declared* `sweep` needs — the requirement follows the declaration.
  - The eight `*_label` / `power_scale` arguments collapsed into two dicts,
    `curated_labels=` / `curated_scales=`.
  - Both ROIs are always loaded (`spectra_roi1`/`spectra_roi2`); `roi=` only
    chooses what `spectra` points at.
  - `SPECTROSCOPY_TYPES` **moved** from `reference/loader.py` to `constants.py`
    (with `"RC"` added) and is re-exported there. One vocabulary — don't fork it.
  - Covered by `tests/test_loaders.py` (now against the new class, plus shim
    tests) and `tests/test_hdf5_roundtrip.py`.
- **HDF5 storage** (`hdf5.py`, 2026-07-30). `scan.to_hdf5(path)`; the loader
  accepts `.h5`/`.hdf5` and dispatches on suffix, so **one class serves both
  formats** — do not add a second loader class for HDF5. The file stores raw
  signal arrays (both ROIs for a spectral sweep), every parameter row verbatim,
  and the measurement metadata. It deliberately does **not** store the energy
  axis, the energy-space spectra, or the sweep axis: all are derivable, and
  freezing them would put one session's corrections into the archive. Corrections
  — `apply_jacobian`, `bg_region_nm`/`_ns`, and the `bg_spectrum` / `reference`
  arrays — are recorded as provenance in `scan.source_metadata` and are **never
  replayed on read**; loading is not deciding. The auxiliary spectra are stored as
  *arrays not paths* so a contrast can still be rebuilt from the archive alone.
  `FORMAT_VERSION` 1.1 added the temporal axis kind (additive). A 4.59 MB PL CSV
  writes as 0.14 MB; the 4-file 11.57 MB TRPL sweep as 0.069 MB.
- **`AttoCubeTRPLSweep`** (2026-07-30). Sibling of `AttoCubeSpectralSweep` over a
  shared private base `_AttoCubeSweep`. Accepts one file *or* a directory. Do not
  merge it back into one class with a mode flag: a single decay is just
  `n_sweeps == 1`, but `energy = hc/t` is meaningless and divides by zero at
  `t = 0`, so a mode flag would leave a third of the public API conditionally
  meaningful. It has no `spectra` attribute **on purpose**, so a TRPL sweep handed
  to a spectral plot raises instead of drawing time as wavelength.
- **The TRPL metadata companion is evidence, not the source.** Parameters come
  from each data file's own snapshot, contemporaneous with its decay, so a sweep
  loads without the companion at all. The companion supplies `n_declared_sweeps`
  (an aborted sweep is then visible) and its table is exposed as
  `declared_parameters`. Its values are **deliberately not** cross-checked row by
  row: it is written seconds after the last decay, so drifting channels genuinely
  disagree — the leakage currents and `Fianium_Select_A6` do, while the swept
  gates agree to seven figures. Nothing in the file says which channels are
  stable, so a value check would fire on every real sweep, which is how warnings
  get ignored. Don't add one back.

**Open, not yet fixed:**
- `plot_diffusion_cloud` double-subtracts the background when handed an image object
  that already had `bg_region` applied at load.
- README §5/§6 reference APIs that don't exist (`AttoCubePLScan`, `plot_pl_map`,
  `bg_region=` on `fit_scan_peak`). The `__init__.py` quick-start was corrected on
  2026-07-30 as part of the rename.
- `plotting.py` hardcodes "PL intensity" in ~6 places and
  `plot_pl_map_Vab_scan` / `plot_current(ef_axis=)` are named for the gate-sweep
  era. Nothing breaks — `scan.signal_label` and `scan.sweep_axis` exist for them —
  but a plotting pass is owed. See E12.
- **A declared 1-D sweep on a raster gives a sawtooth axis, silently** (A8).
  `sweep="position_x"` on the reflectance raster succeeds and returns `Scanner X`
  repeated 51 times, non-monotonic; `__repr__` prints only min and max so it looks
  fine, and any plot against it overplots 51 times. `sweep_grid()` already detects
  the raster, so the fix is to compare the two and warn. Don't make it an error —
  one axis of a raster is legitimate when slicing a single row.
- **`_is_image_csv` accepts a two-row spectrum as an image** (A9), because it only
  tests whether the first line parses as floats — and a `SingleSpectrum`'s first row
  is its wavelength axis. A directory of single spectra therefore loads as 2×N
  "images". `_read_block_layout` already draws this distinction correctly (two rows →
  `SingleSpectrum`, more → image sequence); copy that rule. Same pass as A7.

**Decided but not yet implemented:**
- `plot_pl_map_Vab_scan`'s `median_kernel` should default to `1` (off). The current
  default of `3` runs a 2-D median filter that smooths across gate voltage, mixing
  physically independent sweeps. Keep 2-D available, just not by default.
- `plot_diffusion_cloud` should shed its ~15 enumerated style parameters
  (`contour_*`, `centroid_*`, `roi_color`, `bg_region_color`, `laser_*`,
  `xlabel`/`ylabel`, `colorbar_label`) and return its artists instead of only
  `result`. Signature change, acceptable pre-adoption. See E11.
- **An x/y raster gets a grid API, not a `_SWEEP_TYPES` entry.** Asked and settled
  2026-07-31, so don't re-propose `"position_xy"`: that registry answers *"which 1-D
  array of length `n_sweeps` is the sweep axis"*, and `sweep_axis` returns exactly one
  array that plotting calls `.min()`/`.max()` on. A raster has two axes, so there is
  nothing single to return — a tuple breaks the contract, a flat index is already
  `sweep=None`, and one-of-the-two is already `position_x`/`position_y`. What is
  missing is the **reshape** to `(n_points, n_y, n_x)` plus map plotting, on top of
  the shape `sweep_grid()` already reports. Give it an explicit
  `grid=("Scanner X", "Scanner Y")` declaration, **inner axis first**: detection needs
  `n_inner × n_outer == n_sweeps` exactly and so fails on an aborted scan with a
  partial final row, and declaring it settles the x/y loop-order flip by statement
  rather than inference — the same "put it on the caller" pattern as `sweep=` and gate
  polarity.

## Working style

Changes are made **one at a time**. Report adjacent problems found along the way
rather than fixing them unasked. For physics or analysis judgment calls, state the
mechanism and ask — don't pick a default.

**Depth of explanation follows the domain.** The same person is expert on one side
of a file and new to the other; calibrate per topic, and ask rather than assume.

- *Physics, optics, analysis maths* — full speed. This is Brandon's field.
- *Everyday programming* — competent. Don't explain syntax, control flow, NumPy
  indexing, or what a dataclass is.
- *Library design and long-term maintenance* — the real gap, and what this package
  is becoming. Designing signatures other people will depend on, deprecation and
  versioning, packaging and dependency resolution, release process, and CI /
  GitHub Actions especially (new; see F1 in `dev/audit-2026-07.md`). Here: slow
  down, say what each piece does and why before adding it, go a step at a time,
  and aim for him being able to debug it himself afterwards rather than for a
  working config landing in one move.
