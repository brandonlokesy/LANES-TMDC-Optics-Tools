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

## Code conventions

- NumPy-style docstrings (rendered by mkdocstrings); aligned-colon parameter blocks.
- Relative imports within the package (`from . import processing`).
- `loaders` = I/O + geometry; `processing` = pure array functions; `fitting` returns
  dataclasses; `plotting` returns `(fig, ax, <artist>)` and never calls `plt.show()`.
- Plotting must not re-implement maths that belongs in `processing`.
- New module → add `docs/api/<module>.md` with a `:::` directive **and** a `nav`
  entry in `mkdocs.yml`; `mkdocs build --strict` must stay green.

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
