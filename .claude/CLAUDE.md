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

**Measurements this package supports:** PL, reflectance / reflectance contrast,
time-resolved PL, and real-space imaging. Absorption, cavity, and BFP/k-space data are
measured in the lab but have no loader. Don't propose speculative multi-modality
abstractions unasked — but do flag where a PL assumption will resist a future loader.

**Status:** alpha, pre-adoption. Renaming and signature changes are acceptable; prefer
fixing a name now over carrying a compatibility shim.

## Where things are written down

**This file holds rules. Everything else has a home.** Before writing a paragraph
anywhere, find its home here.

| File | Holds | Write here when… |
|---|---|---|
| `.claude/CLAUDE.md` | rules, conventions, do-nots | it is an instruction that applies to any future task |
| `dev/decisions/NNNN-*.md` | one record per decision: chosen · rejected · consequences | you chose between real alternatives, or refused one |
| `dev/design-principles.md` | the argument behind a rule, with worked examples | you are explaining *why* a rule is the rule |
| `dev/architecture.md` | how the package works now — vocabulary, shapes, call flow | it describes current behaviour |
| `dev/physics-conventions.md` | what a quantity means, its units, when it is invalid | it is about the physics, not the code |
| `dev/instruments/<system>.md` | export format and hardware facts | it is about what the instrument or its exporter did |
| `dev/defects.md` | defects, open and fixed | something is *wrong* |
| `dev/TODO.md`, `dev/plan-*.md` | work not yet done | it is about the future |
| a docstring | what it does, takes, returns, refuses; units; limits | the reader is about to call it |
| a comment | why this line is odd — the mechanism, the trap | the reader is editing that line |

**Three tests, in order.**

1. **Instruction, or fact?** An instruction that applies regardless of task belongs
   here. A fact about the format, the physics, or the code belongs in its record.
2. **Would it still be true if the code had always been this way?** If not it is
   history — a decision record or the audit, never a docstring.
3. **Caller, editor, or decider?** Docstring, comment, or `dev/`.

Two rules that keep this file from re-accumulating:

- **Keep the rule here, move the reasoning out.** If a bullet in this file needs a
  paragraph to justify it, the paragraph goes to `dev/` and a one-line pointer stays.
- **Nothing in this file carries a date, a commit hash, or an audit ID.** Those belong
  in dated documents where they can be corrected.

Decision records are **append-only**: never rewrite an accepted one. Superseding writes
a new record and points the two at each other. Conventions in
`dev/decisions/README.md`.

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
i.e. a different one each run. Nothing is wrong with the environment when this happens.

`pytest` **is installed** in `viz-sci-plot` and the whole suite runs, but it is **not
declared** in `pyproject.toml`: there is no `test` extra, so the dependency exists only
in this one environment. Tests are local-only by deliberate choice — do not add a test
job to CI. CI (`.github/workflows/docs.yml`) builds and deploys docs only.

`pyproject` says `requires-python = ">=3.9"` but the docs CI uses 3.12. Ask which is
authoritative before relying on version-specific syntax.

## Code conventions

- NumPy-style docstrings (rendered by mkdocstrings); aligned-colon parameter blocks.
- Relative imports within the package (`from . import processing`).
- `loaders` = I/O + geometry; `processing` = pure array functions; `fitting` returns
  dataclasses; `plotting` returns `(fig, ax, <artist>)` and never calls `plt.show()`.
  A function that draws several artists returns all of them.
- Plotting must not re-implement maths that belongs in `processing`.
- **Raw arrays are never mutated after load.** `scan.spectra` is the untouched file
  contents; corrections produce new arrays.
- New module → add `docs/api/<module>.md` with a `:::` directive **and** a `nav` entry
  in `mkdocs.yml`; `mkdocs build --strict` must stay green.

**Vectorised NumPy is wanted, but never silent.** Broadcasting, fancy indexing, boolean
masks, and comprehensions are preferred over explicit Python loops — write the fast
version, don't hand-roll a loop to be kind. The condition is that a reader must never
have to re-derive the trick from the expression. Name the shapes and say what the
operation is doing:

```python
# (n_pixels, 1) broadcast against (n_pixels, n_sweeps): one baseline per pixel,
# subtracted from every sweep.
corrected = spectra - baseline[:, None]

# Median along the sweep axis only — the (1, window) footprint means no
# information is ever mixed between detector pixels.
local_med = median_filter(spectra, size=(1, window))
```

A comprehension gets the same treatment: say what it is collecting and over what. The
bar is that someone reading it six months later can see the intent without running it.
Explain the same thing in prose in the reply that introduces it — efficient code is
fine, unexplained code is not.

## Design rules

The argument for each, with worked examples, is in `dev/design-principles.md`.

**Corrections are opt-in.** The researcher decides whether a further correction is
warranted; the package never decides for them. A step that alters the data is a
parameter set to off, not a behaviour you suppress. Off means the *least-assuming*
option, not the least code. Where a permitted default can still destroy a feature, warn
and name what was affected. Return the masks, flags and diagnostics. Never move a
correction into a loader's default path — loading is not deciding.
*Stated exception:* `baseline="constant"` in the `fit_*` functions defaults **on** —
it is a model term, and omitting it migrates the pedestal into amplitude and FWHM.
Don't "fix" it to `"none"`.

**Parameters earn their place.** A function exposes the minimum set its callers cannot
readily supply themselves. The test is whether the argument changes **the numbers or
only the pixels**. Never add a parameter whose entire body is
`artist.set_<thing>(value)` — return the artist instead. One `**kwargs` passthrough at
most, and only where a single artist dominates. Style with a global home belongs in
`set_style`/rcParams. Enumerated style arguments are a symptom of a broken return
contract; fix the return first.

**Reuse before adding, delete before documenting.** Search for the concept before
writing it — if a helper is wanted in two modules, the first one is in the wrong place;
move it and import. Prefer composing an existing entry point over adding a
near-duplicate. Anything nothing reaches gets deleted, not documented. This is not a
licence to write the terse version.

**A docstring is a contract, not a changelog.** It describes the thing as it is, to
someone who has never seen the source and does not know the project has a history. No
dates, no audit IDs, no `was`/`now`/`previously`, no pointers into `dev/`, no arguing
with the design that was not chosen, no "deliberately". Keep limitations, units, axis
order, and constraints on use. Keep the consequence, move the argument. Never cite a
private helper from a public docstring.

## References must be followable

**Someone reading a reference must be able to identify and find the source.**

- A reference to a thesis, paper, or equation needs enough to find it: author, year,
  title; a DOI for a paper.
- **An equation number is meaningless without the document it indexes.** If the
  document cannot be cited, state the claim without the number.
- Surname + year is not enough when a specific table or figure carries the value — say
  which.
- Where a claim rests on a source that cannot be cited (inherited group material, an
  undocumented calibration), label it **inherited group practice** and list what is
  missing in `dev/physics-conventions.md` §9. Do not present it as a literature value.
- **Internal references too:** a pointer to a file or section must resolve. Deleting or
  renaming a file means fixing every reference to it in the same change.

## Guardrails

- Never run `reference/registry.py` — it downloads from Zenodo and mediaTUM FTP.
- Don't commit `.h5` files (gitignored); don't regenerate `examples/data/`. That data
  lives in the repo deliberately, to reproduce the example notebooks.
- Don't touch `site/` (mkdocs build output).

## Do not re-propose

Each of these is a settled decision with a record in `dev/decisions/` — find it by
keyword in the index there. Don't re-litigate, and don't "helpfully" restore.

- Don't reinstate `DeviceGeometry.optical_thickness`, or make `_slabs()` return
  `StackLayer` objects.
- Don't hand-build a `patches.Circle` for a laser spot; use `_draw_laser_circle`.
- Don't give `spectra_type=` a default, or make it positional.
- Don't auto-detect the sweep axis; an undeclared `sweep=` means the sweep **index**.
- Don't fork `SPECTROSCOPY_TYPES` out of `constants.py`.
- Don't add a second loader class for HDF5, store derived arrays in it, or replay
  corrections on read.
- Don't give `AttoCubeTRPLSweep` a `spectra` attribute, or merge it into the spectral
  class; don't add a row-by-row value check against the TRPL metadata companion.
- Don't change the deprecation shim's `FutureWarning` to `DeprecationWarning`.
- Don't apply the Jacobian before subtracting the background.
- Don't declare the gate mapping anywhere but `gates=`, and don't re-propose
  `gates={"bottom": "A"}`.
- Don't rename `v_top`/`v_bot`/`i_top`/`i_bot` to channel names.
- Don't make `ef` work for a single-gated device, or treat `"channel"` as a third gate.
- Don't default a `v_ref` threshold to make `carrier_density` absolute.
- Don't return zeros for `i_*` on a role declared `None`.
- Don't make `gate_mode` or `__repr__` raise for an undeclared mapping.
- Don't respell a nest as `grid=(inner, outer)`, alias `fast_sweep=` to `sweep=`, or add
  a `"piezo_xy"` sweep type. A nest's shape is asserted as **named** `n_fast=` / `n_slow=`,
  never as a `sweep_shape=(fast, slow)` tuple — a reversed tuple cannot be detected.
- Don't resolve a nest by loosening `_AXIS_RTOL`, or by any single per-axis tolerance;
  don't trim outliers before comparing levels. Don't make an asserted shape skip the
  overlap checks, or store `n_fast`/`n_slow` in HDF5 for a nest the readings resolved.
- Don't name an instrument in a parameter (`force_power_by_fianium=`); which row drives
  what is declared per session, through `*_group_by=`. Don't make a level's coordinate the
  mean or the first reading instead of the median.
- Don't decide whether an axis was driven from one sign. Both are needed and either is
  sufficient — dropping sign 1 collapses a flattened nest, dropping sign 2 collapses a
  narrow sweep on a large offset (300.0–300.2 K). Don't floor `_axis_atol` at the row's
  magnitude, respell sign 2 as a gap *ratio*, or scale its threshold by the point count:
  all three are measured to collapse a real axis. Don't add `sweep_atol=` until a
  committed file needs it.
- Don't reshape `spectra` when a nest is declared; don't extend `axis=` to nests.
- Don't make the accessors return one match for an ambiguous coordinate.
- Don't give `plot_spectrum` a positional `value`, or truncate a fractional index.
- Don't append to a caller's label string — `None` derives, a string is verbatim.
- Don't make cosmic-ray removal a plotting argument or a third array branch.
- Don't repair a gap or pick a winner among duplicate `_iter_N` indices.
- `gate_axis` / `gate_axis_label` stay as aliases until `plot_pl_map_Vab_scan` is
  updated.

## Known issues — check before "helpfully" fixing

Full detail and fix sketches: `dev/defects.md`.

**Deferred by explicit decision — leave alone:** `diffusion._binary_area` has a wrong
MATLAB `bwarea` weight table (diagonal pairs 0.5, should be 0.75; 3-pixel patterns
0.75, should be 0.875), biasing cloud areas low. Parked pending a decision on whether
bwarea semantics are wanted at all.

**Open:**

- `plot_diffusion_cloud` double-subtracts the background when handed an image object
  that already had `bg_region` applied at load. It also has ~30 parameters and returns
  `result` instead of its artists — the standing counter-example to *parameters earn
  their place*. New code must not copy it.
- README §5/§6 reference APIs that don't exist (`AttoCubePLScan`, `plot_pl_map`,
  `bg_region=` on `fit_scan_peak`).
- `plot_current` is still named for the gate-sweep era, as are `plot_spectrum`'s
  hand-rolled `E_F` legend default and `SpectrumLinePanel`'s `sweep_attr`/`sweep_unit`.
  **Do the `plot_current` rename first** — two breaking changes to that one function are
  owed, so land them together. See `dev/plan-E12.md`.
- `_is_image_csv` accepts a two-row spectrum as an image, so a directory of single
  spectra loads as 2×N "images". `_read_block_layout` already draws this distinction
  correctly; copy that rule.
- **Every `stacklevel` in `loaders.py` is unverified** — 15 `warnings.warn` calls, no
  test pinning where any of them points, and two chains confirmed wrong. A wrong value
  also *suppresses repeats*, so it is a diagnostics failure rather than a cosmetic one.
  Its own pass, not a character changed while passing through. **Trace by measuring,
  not by reading `def` lines.**
- `plot_pl_map_Vab_scan`'s `median_kernel` should default to `1` (off). The current `3`
  runs a 2-D median filter that smooths across gate voltage, mixing physically
  independent sweeps. Keep 2-D available, just not by default.

## Working style

Changes are made **one at a time**. Report adjacent problems found along the way rather
than fixing them unasked. For physics or analysis judgment calls, state the mechanism
and ask — don't pick a default.

**Depth of explanation follows the domain.** The same person is expert on one side of a
file and new to the other; calibrate per topic, and ask rather than assume.

- *Physics, optics, analysis maths* — full speed. This is Brandon's field.
- *Everyday programming* — competent. Don't explain syntax, control flow, NumPy
  indexing, or what a dataclass is.
- *Library design and long-term maintenance* — the real gap, and what this package is
  becoming. Designing signatures other people will depend on, deprecation and
  versioning, packaging and dependency resolution, release process, and CI / GitHub
  Actions especially. Here: slow down, say what each piece does and why before adding
  it, go a step at a time, and aim for him being able to debug it himself afterwards
  rather than for a working config landing in one move.
