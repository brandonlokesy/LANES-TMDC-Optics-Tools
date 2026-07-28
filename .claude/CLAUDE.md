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

**Measurements this package currently supports:** PL and real-space imaging only.
Reflectance, absorption, cavity, and BFP/k-space data are measured in the lab but
have no loader here (`reference/` is the only place `"R"` appears, as a
spectroscopy-type tag). Treat the PL-shaped design as deliberate-for-now. Don't
propose speculative multi-modality abstractions unasked — but do flag where a PL
assumption will resist a future loader.

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

`pytest` is **not currently installed** in `viz-sci-plot` and is not declared in
`pyproject.toml`. Tests are local-only by deliberate choice — do not add a test job
to CI. CI (`.github/workflows/docs.yml`) builds and deploys docs only.

TODO: `pyproject` says `requires-python = ">=3.9"` but the docs CI uses 3.12.
Ask which is authoritative before relying on version-specific syntax.

## Physics conventions

**Displacement field** — `DeviceGeometry.electric_field` is **correct as written**;
do not "fix" it:

```
E_2D ≈ (V_BG − V_TG) / d_tot · (ε_hBN / ε_2D)
d_tot = d_2D + d_hBN,top + d_hBN,bottom
```

The full-stack thickness in the denominator (including the TMDC layers) is
deliberate. `ε_2D` is the TMDC-only series-capacitor value (`DeviceGeometry.eps_2d`).
Result is in mV/nm.

**Jacobian** — `apply_jacobian` defaults to `False` and that is intended. The
docstrings and README still claim "True (default)"; the *docs* are wrong, not the
code. When the Jacobian is applied, background must be subtracted in wavelength
space **first**, because a flat pedestal `B` becomes `B·λ²/hc` — curved, not flat —
in energy space. The loader already does this in the right order.

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
- `EPS_TMDC["HS"] = 7.5` — unsourced, sits among per-material values.
- `T_MONOLAYER` = 0.65 nm for all four materials — deliberate approximation?

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

## Code conventions

- NumPy-style docstrings (rendered by mkdocstrings); aligned-colon parameter blocks.
- Relative imports within the package (`from . import processing`).
- `loaders` = I/O + geometry; `processing` = pure array functions; `fitting` returns
  dataclasses; `plotting` returns `(fig, ax, <artist>)` and never calls `plt.show()`.
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
- `AttoCubePLVabScan` refuses to load if any `_CURATED` row is missing (including
  `Scanner X`/`Scanner Y`). A larger overhaul of this class is planned.

**Open, not yet fixed:**
- `processing.remove_cosmic_rays` raises `UnboundLocalError` (`cosmic_mask` vs
  `cr_mask`) — cannot be called at all.
- `plotting.py` (~line 681) `circle.set_path_effects([path_effects])` passes the
  module, not a path effect; fails on draw.
- `plot_diffusion_cloud` double-subtracts the background when handed an image object
  that already had `bg_region` applied at load.
- `DeviceGeometry.eps_hs` iterates `(d, ε)` tuples as `StackLayer` objects →
  `AttributeError`, which also breaks `__repr__`.
- `__init__.py` quick-start and README §5/§6 reference APIs that don't exist
  (`AttoCubePLScan`, `plot_pl_map`, `bg_region=` on `fit_scan_peak`).

**Decided but not yet implemented:**
- `plot_pl_map_Vab_scan`'s `median_kernel` should default to `1` (off). The current
  default of `3` runs a 2-D median filter that smooths across gate voltage, mixing
  physically independent sweeps. Keep 2-D available, just not by default.

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
