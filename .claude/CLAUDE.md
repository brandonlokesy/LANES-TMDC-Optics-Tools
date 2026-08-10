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
clouds), back-focal-plane measurements (k-space dispersion) as real images on CCDs.

**Measurements this package currently supports (as of 2026-07-30):** PL,
reflectance / reflectance contrast, time-resolved PL, and real-space imaging.
Absorption, cavity, and BFP/k-space data are measured in the lab but have no
loader. Don't propose speculative multi-modality abstractions unasked — but do
flag where a PL assumption will resist a future loader (the hardcoded "PL
intensity" strings in `plotting.py` are the live example; `scan.signal_label` and
`scan.contrast_label` exist for them — see E12).

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
  reflectance example). `sweep_grid()` detects and reports the shape; `as_grid()`
  reshapes it, once the nest is declared with `fast_sweep=` / `slow_sweep=` (E14).
- **The committed reflectance export is a truncation, not a raster.**
  `examples/data/reflectance-contrast/sample_truncated_…csv` holds the first **50**
  points of that 41 × 51 scan — one complete X row plus 9 of the next — which the
  filename says and the numbers above do not. So it is a real fixture for the
  aborted-scan refusal and for nothing else: **no complete raster is committed**,
  and a test needing one must synthesise it (`tests/test_loaders_nesting.py`).

Still unknown, and no file can answer it: which acquisition software and version
emits this, and whether the layout is version-stable.

**Status:** alpha, pre-adoption. Renaming and signature changes are acceptable;
prefer fixing a name now over carrying a compatibility shim. (`DeviceGeometry.from_single`
is labelled "backward compatibility" but predates any real user base.)

## Environment

Conda env `viz-sci-plot`. **Activate it before running anything that imports numpy:**

```
conda activate viz-sci-plot

Install     : pip install -e ".[docs]"
Docs        : python -m mkdocs build --strict
Tests       : python -m pytest -q
```

Non-interactively (scripts, agents), one-shot instead of activating — there is no
persistent shell session to activate into:

```
conda run --no-capture-output -n viz-sci-plot python -m pytest -q
```

`--no-capture-output` matters: without it `conda run` buffers everything until the
process exits, so a crashing or hanging suite reports nothing.

**Do not invoke `C:\Users\Brandon\anaconda3\envs\viz-sci-plot\python.exe` directly.**
It is the right interpreter, but activation is what puts the env's `Library\bin` on
`PATH`, and conda's numpy keeps its BLAS there (`mkl_rt.3.dll`) rather than vendoring
it next to the extension module the way a pip wheel does. Unactivated, numpy imports
fine and then dies at the **first BLAS call** — matmul, `np.corrcoef` (so
`gate_mode`), skimage `regionprops` (so `_fit_laser_spot`) — with `Windows fatal
exception: code 0xc06d007f`. That is a delay-load failure: a native crash, not a test
failure, so there is no traceback, and it lands on whichever test reaches BLAS first,
i.e. a different one each run. Nothing is wrong with the environment when this
happens.

`pytest` **is installed** in `viz-sci-plot` (9.1.1; the whole suite runs — 336 tests
passing as of 2026-08-06) but is **not declared** in `pyproject.toml`: there is no `test` extra, so
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
- ~~`Scanner X` / `Scanner Y` units~~ **Answered 2026-08-04: volts.** The scanners
  are piezos and the rows carry their *drive voltage*, so `scanner_x`/`scanner_y`
  and the `piezo_x`/`piezo_y` sweep axes are in V, scale 1.0. A distance needs a
  per-stage µm/V calibration that no file contains — supply it via
  `curated_scales` rather than assuming one. The sweep keys were `position_*`
  until this was settled; they are `piezo_*` because the axis is a drive level.
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
- ~~**Gate polarity is per-session wiring**~~ **Enforced 2026-08-05 — the code no
  longer guesses, but you still have to know.** The electrodes can be hooked up in
  either configuration, so which acquisition channel drove which gate is *not*
  inferable and must come from the lab notebook per session; the old MATLAB used
  `dm(4,…)`/`dm(6,…)` with no note of which was which. The senior's thesis is itself
  inconsistent here (eq. 3.11 carries a leading minus, 3.12/3.13 do not), so don't
  inherit its sign, and don't add a sign to the physics.

  The loaders now take `gates=` and **refuse** to produce `v_top`, `v_bot`,
  `v_channel`, `ef`, or a `top_voltage`/`bottom_voltage`/`electric_field` sweep
  without it — closing E7b, which was opened by exactly this going wrong on a real
  measurement. The mapping is recorded on the scan (`scan.gates`), printed in
  `__repr__`, and written into exported HDF5 as its own attribute so a round trip
  cannot launder an unstated wiring into a stated one. Channel-level work needs no
  declaration: `scan["V_A"]` and `sweep="V_A"` are unaffected. Details, including
  what deliberately does *not* raise, are in E7b.

- **`gates` declares device topology, not just wiring.** The roles *present* say
  what the device is, and what is computable follows (E7c):

  | Declaration | Device | Available |
  |---|---|---|
  | `{"top": "V_A", "bottom": "V_B"}` | dual-gated | `ef`, `v_top`, `v_bot` |
  | `{"bottom": "V_A", "channel": None}` | bottom-gated, TMDC grounded | `carrier_density`, `v_bot`; `ef` raises |

  `"channel"` is a contact to the TMDC itself — **not a gate**: it sits inside the
  stack, carries no thickness, enters no field, and is excluded from `gate_mode`.
  A value of `None` means the electrode is hard-grounded with no row recording it.
  `is_dual_gated` is the one predicate for "a field is defined"; `plotting` branches
  on it. At least one gate is required, and a lone gate must name its `"channel"`,
  or a single-gated device could not be told from a forgotten second gate.

  **Do not make `ef` work for a single-gated device.** Its derivation fails twice
  there — no second equipotential to define `V_BG − V_TG`, and a grounded TMDC *is*
  the free charge the no-free-charge assumption excludes. One gate is also one
  degree of freedom, so field and density are locked; independent control is what
  the dual-gate anti-symmetric sweep buys. See E7c.

**Carrier density** — `DeviceGeometry.gate_capacitance(gate)` is `ε₀ε_hBN/d_hBN`.
The TMDC is the **counter-electrode**, not a slab inside the capacitor, so neither
its thickness nor `eps_stack` enters — that is the sign you have reached for the
wrong tool. `carrier_density` sums `C_i(V_i − V_ref)/e` over the supplied gates,
signed with electrons positive, in cm⁻².

- **`v_ref` is a gate voltage, not a threshold**, so the result is a density
  *difference*. Absolute `n` needs the threshold at which the channel populates —
  a transfer curve or the PL charging step, in no file. Pass it as `v_ref` if
  measured; don't add a guessed one.
- Geometric only: quantum and interface-trap capacitance are in series and make the
  effective value smaller, so this is an **upper bound**.
- `carrier_density` warns when the declared channel's row varies — the density is
  referenced to that contact, so a driven contact moves the reference under the
  axis. Legitimate for a source-drain bias, wrong for a doping sweep, and no file
  distinguishes them.

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
- **Which data is shown** — `x_axis`, `spectra_source`, `normalize`, and the point
  selectors `value` / `index`, whose contract is under *Settled* below.
- **Physical context the function cannot infer** — `pixel_scale`, `origin`,
  `laser_ref`. The caller knows the µm/px; the array does not.
- **Composition and structure** — `panels`, `ax`, `n_frames`, `save`.
  `animate_panels` takes a list of panel objects rather than a flag per panel type,
  so any subset, order, or combination works with no special-casing. That is the
  shape to aim for: one structural parameter absorbing a combinatorial space.
- **Axis and colour-bar labels — semantics, not styling.** Settled 2026-08-06.
  A label states what the numbers *are*, so a wrong one is a misread, not an
  ugly figure — it falls under the boundary case below, not under
  `artist.set_<thing>(value)`. **One contract, everywhere in `plotting`:
  `None` derives the label from the scan; a string is used verbatim; nothing is
  ever appended to a caller's string.** Derivation reads `signal_name` /
  `signal_unit` / `contrast_label` via `_signal_name_unit`, so it follows
  `spectra_type` and no plot hardcodes "PL". A `normalized` flag **substitutes**
  the unit and never adds one — a ratio such as ΔR/R₀ already reads as
  normalised, so `$\Delta R/R_0$ (norm.)` states it twice. The append is what the rule
  forbids: composing `caller_string + " (counts)"` forces a "pass it without a
  unit" convention that is undocumentable at the call site and has already
  shipped `"PL intensity (norm.) (norm.)"` once. The signal side gets **no**
  loader-level override — `spectra_type` is a closed validated vocabulary
  (G1); the sweep side does, and declares it once via `sweep=` /
  `sweep_label=` / `sweep_unit=`.

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
  in the wrong place — move it and import, don't fork it. The live example is the
  three laser-circle drawers with different styling defaults (D2). That is how 6k
  lines becomes 10k.
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

**One file is not the format.** Every figure in the AttoCube record above came from
one particular export, and the next one will differ — different raster dimensions,
different sweep length, different file size, possibly a different acquisition
version. So docstrings *and* comments describe the **shape of the thing** — that
the exporter declares twice as many blocks as it writes, that a raster arrives
flattened, that exports are large enough to make a one-line read worth it — and
never present one file's numbers as what a caller should expect.

- **Generalise the mechanism, not the measurement.** *"declares twice as many
  blocks as it writes"* is behaviour, holds across every export seen, and is what
  the code keys on; *"a 2091-point raster exported with 4182 blocks"* is one file's
  arithmetic. State the first, drop the second. Where the prose is more specific
  than the logic, the prose is wrong.
- **Where an illustration genuinely helps, make the numbers obvious stand-ins.**
  Symbolic is best (*"an `n_x` × `n_y` raster gives `n_x·n_y` points"*); round and
  small is fine (*"e.g. 10 × 10 exported as 100 points"*). A figure that matches
  the committed data reads as a specification, not an example — and a doctest whose
  expected output is one file's value (`(12.817, 3)`) is a specification with a
  test attached.
- **Exact figures from real files belong in the record** — the AttoCube section of
  this file, or `dev/audit-2026-07.md`. There they are dated, attached to the file
  they came from, and correctable when the next acquisition version disagrees. Same
  split as *evidence for the implementation* above.
- **They rot, and silently.** `_read_block_layout` already says a raster is 314 MB
  while the comment four lines below it says 300 MB. Nothing failed, no test caught
  it, and the reader now has two facts about a file they do not have.

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

**Fixed — don't re-report.** `remove_cosmic_rays` (A1), `DeviceGeometry.eps_hs` /
`__repr__` and `optical_thickness` (A2), the `animate_real_space_PL_map` laser
circle (A3), zero-filled blocks loaded as sweep points (A6), the silent sawtooth
sweep axis on a raster (A8), the zero-mean overflow in `varying_parameters` (A10),
the lexicographic frame order in `AttoCubePLScanRealSpace` (A7), duplicate
iteration indices passing unreported (A12), a valid nest refused because a
read-back level was wider than the axis tolerance (A13), the `_CURATED`
fail-fast (E1), the silently-defaulted channel-to-gate mapping (E7b), nested
sweeps (E14), and the
2026-07-30 rewrite — `AttoCubeSpectralSweep`, `hdf5.py`, `AttoCubeTRPLSweep`
(G1–G4). Diagnosis, fix and tests for each are in the audit.

**Settled — don't re-litigate, and don't "helpfully" restore.** Each line is a
decision; the argument for it is in the audit under the ID given.

- `DeviceGeometry.optical_thickness` is **deleted**; do not reinstate it. Optics
  code needs dispersive `n(λ)`, not `EPS_TMDC`. (A2)
- `_slabs()` returns `(d, ε)` **tuples**, not `StackLayer` — wrapping an hBN flake
  would assert a false `n_layers`. (A2)
- Annotate a laser spot with `_draw_laser_circle`, never a hand-built
  `patches.Circle`. Two drawers remain to unify. (A3, D2)
- `spectra_type=` is **required, keyword-only, no default**. (G1)
- An undeclared `sweep=` means the sweep **index** — never an auto-detected
  parameter. (G1)
- `gate_axis` / `gate_axis_label` stay as aliases until `plot_pl_map_Vab_scan` is
  updated. (G1, E12)
- `SPECTROSCOPY_TYPES` lives in `constants.py`. One vocabulary — don't fork it. (G1)
- One loader class reads both CSV and HDF5; don't add a second for HDF5. (G2)
- HDF5 stores no derived arrays, and never replays corrections on read. (G2)
- `AttoCubeTRPLSweep` stays a separate class and has no `spectra` attribute. (G3)
- No row-by-row value check against the TRPL metadata companion. (G4)
- The `AttoCubePLVabScan` shim raises `FutureWarning`, not `DeprecationWarning`. (G1)
- Background is subtracted in wavelength space **before** the Jacobian. (G5)
- The channel-to-gate mapping is declared through **`gates=` and nowhere else**;
  `v_top` / `v_bot` are rejected as `curated_labels` keys. One fact, one spelling.
  (E7b)
- **`v_top` / `v_bot` / `top_voltage` / `bottom_voltage` keep naming physical
  roles** — don't rename them to `V_A` / `V_B`. Asked and settled 2026-08-05: the
  role layer is where the field's sign convention lives, so channel names there
  would make `electric_field` unambiguous-looking and still wrong half the time,
  moving the ambiguity out of a recorded mapping and into the researcher's head.
  Channels are already reachable as raw rows. (E7b)
- `gate_mode` and `__repr__` **never raise** for an undeclared mapping, and
  `ef` returns `None` rather than raising when no geometry was supplied. A
  diagnostic that dies is no diagnostic. (E7b)
- **`"channel"` is a `gates` role, not a third gate.** Asked and settled
  2026-08-05: it records the contact that grounds the TMDC, so it makes a
  single-gate declaration unambiguous and is what a density is referenced to. It
  is excluded from `gate_mode` and from every field. (E7c)
- **`carrier_density` returns a density *difference*, referenced to a gate
  voltage.** Absolute `n` needs a threshold no file records; don't default one.
  (E7c)
- **Electrode currents are role-named (`i_top` / `i_bot` / `i_channel`) and come
  from `gates=`.** Settled 2026-08-07. A and B are source-meter channels, so `V_A`
  and `I_A` are one terminal and one declaration names both — the correspondence
  lives in `_CHANNEL_SIBLING_CURRENT`, a format fact, while `gates=` keeps naming
  **rows**. Don't re-propose `gates={"bottom": "A"}`: the row↔current pairing is
  format-stable and the role↔channel wiring is per-session, so fusing them puts a
  format convention into every notebook, and every other API (`sweep=`, `scan[…]`,
  `varying_parameters()`, the undeclared-gates error) speaks rows. `Ich1`/`Ich2` are
  **deleted**, not renamed to `i_a`/`i_b` — which electrode `I_A` flows into is the
  undeclared fact, so the currents refuse without `gates=` exactly as the voltages
  do. `scan["I_A"]` still works undeclared. A role declared `None` gives `v_*`
  zeros but makes `i_*` **raise**: grounding fixes a potential, not a current. (E15)
- **Cosmic-ray repair is a load-time `cosmic_rays=` dict, run first in the
  wavelength-space chain** — not a plotting argument and not a third flag with its
  own energy-space arrays. It feeds the background estimate, the contrast and the
  fits; `spectra` stays the file's own counts, `spectra_cr` holds the repair and
  `cosmic_ray_mask` says what moved. (E13)
- **Nested sweeps are declared with `fast_sweep=` / `slow_sweep=`, which are not
  aliases of `sweep=`.** Settled 2026-08-06, superseding the `grid=(inner, outer)`
  tuple. Everywhere in this package **"sweep" means the flattened measurement
  point** — `n_sweeps`, `sweep_mask`, `sweep_axis` (an array *of length
  `n_sweeps`*), and the `index=` that the accessors and `plot_spectrum` take. So
  `sweep=` answers *"which array labels each flat point"*
  and the nest declaration is a separate statement, *"those points are `n_fast`
  inside `n_slow`"*. Don't redefine `sweep=` to mean the fast axis: it would be the
  one place in the package the word meant an axis, two lines from `n_sweeps` still
  meaning the other thing. On a nested scan `sweep=` is normally omitted, so
  `sweep_axis` is the flat index. (E14)
- **Both nest axes resolve through `_resolve_sweep`, so a derived quantity can be
  an axis.** This is the point of the whole design: with both gates moved together
  to sweep the field, each gate row takes a different value at every point, so no
  row can be the axis while `electric_field` takes exactly `n_fast`.
  `sweep_grid()` still detects on raw rows only and so reports the channels — it
  says what to declare, as `gate_mode` does for `gates=`. It also returns the
  *first* pair that verifies, and on a raster taken during an anti-symmetric gate
  sweep two pairs verify equally well, so it may name the gates rather than the
  scanners. Not a defect to chase: nothing in the rows says which pair the
  experiment was about. (E14, A10)
- **`spectra` keeps shape `(n_points, n_sweeps)` whether or not a nest is
  declared**; the grid is a view from `as_grid(array)`. A declaration must not
  change the rank of an attribute — `n_sweeps` is `spectra.shape[1]`, and every
  consumer would have to branch on `is_nested`. Accessor results are views too,
  which is why the selectors are ints and slices rather than fancy indexing. (E14)
- **`axis=` locates a point by a quantity other than the declared sweep axis** —
  `get_spectrum_at(15.0, axis="top_voltage")` on a field sweep driven by both
  gates. Fourth entry point onto the same `_resolve_sweep` vocabulary; a
  parameter, not a `get_spectrum_from_parameter` sibling. **Flat sweeps only, and
  don't extend it to nests:** there an arbitrary quantity matches `n_slow` points
  or one depending on how the scan was driven, so the return rank would follow the
  data rather than the call. Declare `fast_sweep=` in the coordinate you want
  instead. (E14)
- **An ambiguous coordinate is warned by `nearest_index` and *refused* by the
  accessors.** Not an inconsistency: a single `int` is all `nearest_index` can
  return, whereas `get_spectrum_at` returns data and the API already has the
  complete answer — a declared nest gives every match at once through
  `fast=`/`slow=`, so handing back one of four would be a silent partial answer.
  Matches are compared on the coordinate, not the distance, so a request landing
  midway between two distinct points is not a tie. (E14)
- **`plot_spectrum` selects a point the way the accessors do, and resolves it
  through them.** Settled 2026-08-10. Researchers name a setting, not a column, so
  the coordinate takes the positional slot: `plot_spectrum(scan, 2.5)`. Six
  keyword-only selectors in two exclusive spellings — `value`/`fast`/`slow` are
  coordinates, `index`/`index_fast`/`index_slow` are positions — so the two never
  share a keyword and a request cannot be half of each. Naming both ways, or
  neither, raises; `axis=` applies to coordinates only.

  **Selection is not re-implemented in `plotting`.** `_select_sweep_point` forwards
  to `_sweep_selector`, so the settled policies above reach the figure unchanged:
  an ambiguous coordinate is refused rather than drawn, and a distant one warns.
  Composing `nearest_index` at the call site was rejected for exactly this — it
  *warns* where the accessors *refuse*, so recommending it as the idiom would route
  every value-based plot around the refusal.

  A free nest axis is **refused**, not drawn as N lines: it selects a spectrum per
  point, and the return contract is one artist. The legend names the coordinate
  addressed — both, for a nest, where the declared sweep axis is the flat index and
  says nothing. `_coordinate_text` reuses `sweep_axis_label`'s composition, so
  existing legends are unchanged and a raw-row axis correctly shows no unit.
- **The sweep axis must label each point individually, and the loader warns when
  it does not.** Asked as *"how many different values does this axis take?"*, not
  *"is it monotonic?"* — the latter is a side effect that catches a nest's inner
  quantity (sawtooth) and misses its outer one (staircase), which is the worse
  failure of the two. **No grid guard:** `sweep_grid()` returns `None` for a field
  × power nest, so a guarded check would be silent on exactly the case E14 exists
  for. Deliberate repeat measurements warn too — their map collapses the same way.
  (A8, E14)
- **`_order_by_iter` is one module-level helper shared by both directory loaders,
  and it warns rather than repairs.** Frame and point order come from the integer in
  `_iter_N`, never from `sorted()` on filenames: exports are zero-padded but the
  *width* varies between them, so alphabetical order is right only by luck. Three
  conditions warn and none is repaired — no `_iter_N` suffix, a gap, and an index
  claimed by more than one file. A gap is never closed up and a duplicate never
  resolved by picking a winner, because both would silently restore the mispairing
  the helper exists to catch; the duplicate message **names the colliding files**,
  since two acquisitions sharing a directory is the usual cause and a narrower
  prefix the usual fix. Duplicates are reachable with every file legitimate, so
  don't re-file this under A9. `stacklevel` is a required argument because the two
  callers sit at different depths — see A11 before trusting any warning's line
  number. (A7, A12)

**Open, not yet fixed:**
- `plot_diffusion_cloud` double-subtracts the background when handed an image object
  that already had `bg_region` applied at load.
- README §5/§6 reference APIs that don't exist (`AttoCubePLScan`, `plot_pl_map`,
  `bg_region=` on `fit_scan_peak`). The `__init__.py` quick-start was corrected on
  2026-07-30 as part of the rename.
- `plot_current(ef_axis=)` is still named for the gate-sweep era, as is
  `plot_spectrum`'s hand-rolled `E_F` legend default and `SpectrumLinePanel`'s
  `sweep_attr="scanner_y"` / `sweep_unit="V"`. The rename half of
  E12 is owed. The hardcoded-"PL intensity" half is **done** (2026-08-06); see
  *parameters earn their place* for the label contract that replaced it, and
  `dev/plan-E12.md` Step 3 for what remains.

  **Do the `plot_current` rename before the rest of E12.** E15 changed that
  function on 2026-08-07 — `color_ich1` / `color_ich2` deleted, `lines` appended to
  the return, and it now requires `gates=` — and Step 3 changes it again. Two
  breaking changes to one function, so land them together. The other two renames
  break nothing and can wait. Details and the corrected before-signature are in
  `dev/plan-E12.md` Step 3.
- **`_is_image_csv` accepts a two-row spectrum as an image** (A9), because it only
  tests whether the first line parses as floats — and a `SingleSpectrum`'s first row
  is its wavelength axis. A directory of single spectra therefore loads as 2×N
  "images". `_read_block_layout` already draws this distinction correctly (two rows →
  `SingleSpectrum`, more → image sequence); copy that rule. Now the whole of the
  `AttoCubePLScanRealSpace` pass, since A7 landed without it; take **B1**
  (`bg_region`/`bg_stat` ignored in `load_frame`) with it, and second.
- **Every `stacklevel` in `loaders.py` is unverified** (A11) — 15 `warnings.warn`
  calls, values 2 through 5, no test pinning where any of them points. The TRPL chain
  is confirmed wrong: it needs 6 and passes 4, so those warnings blame
  `loaders.py:1176` instead of the researcher's line. That also **suppresses
  repeats**, because Python's default filter shows a warning once per location, so
  many scans warning from one library line print one message. Its own pass, not a
  character changed while passing through — A7 deliberately left TRPL's 4 alone so
  its existing tests stay honest, and `plot_spectrum` left its own inherited chain
  alone on 2026-08-10 for the same reason.

  **Trace these by measuring, not by reading `def` lines.** The second confirmed
  chain is `get_spectrum_at`, which needs 6 and passes 5, and the uncounted frame
  is a **lambda** — `_sweep_selector` wraps `_index_for_value` in a closure before
  calling it, and a closure is a real frame with no `def` to scroll past. Anything
  routing through `_sweep_selector` inherits the miscount and adds its own depth:
  `plot_spectrum` needs 7. `nearest_index` is correct. A
  `catch_warnings(record=True)` harness asserting `caught[0].filename` settles each
  chain in one run; reading the call stack by eye is what produced the wrong values
  in the first place.

**Decided but not yet implemented:**
- `plot_pl_map_Vab_scan`'s `median_kernel` should default to `1` (off). The current
  default of `3` runs a 2-D median filter that smooths across gate voltage, mixing
  physically independent sweeps. Keep 2-D available, just not by default.
- `plot_diffusion_cloud` should shed its ~15 enumerated style parameters
  (`contour_*`, `centroid_*`, `roi_color`, `bg_region_color`, `laser_*`,
  `xlabel`/`ylabel`, `colorbar_label`) and return its artists instead of only
  `result`. Signature change, acceptable pre-adoption. See E11.
- ~~An x/y raster gets a `grid=(inner, outer)` declaration~~ **Built 2026-08-06 as
  `fast_sweep=` / `slow_sweep=` (E14).** The "not a `_SWEEP_TYPES` entry" half of the
  2026-07-31 decision stands — don't re-propose a `"piezo_xy"`. The tuple spelling
  does not: see *nested sweeps* below.

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
