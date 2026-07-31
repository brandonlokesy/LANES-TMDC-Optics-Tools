# TODO

## Refactoring
- ~~AttoCubePLVabScan needs a rename~~ → `AttoCubeSpectralSweep` (2026-07-30).
  Old name kept as a deprecated subclass emitting `FutureWarning`.
- ~~AttoCubePLVabScan needs a rewrite~~ (2026-07-30)
    - ~~accept PL, R, RC etc. spectra~~ (2026-07-30) — `spectra_type=` is
      required, from `constants.SPECTROSCOPY_TYPES`, and drives `signal_label`.
      **R/RC now genuinely load**: reflectance uses the identical 4-column layout
      as PL, so it needed no parser work — only `reference=` and the contrast
      maths (`processing.spectral_contrast`). **TRPL loads too**, via
      `AttoCubeTRPLSweep`. Absorption/cavity/BFP still have no loader.
    - ~~check format/dimension of spectra files~~ — `_validate_payload`.
    - ~~verify only one gate is being used~~ — `gate_mode` reports
      dual/top-only/bottom-only and, for dual, whether the two are
      anti-correlated (field-like) or correlated (doping-like); shown in
      `__repr__`. Backed by `varying_parameters()`.
    - ~~accept any independent parameter~~ — one `sweep=` argument taking a
      `_SWEEP_TYPES` key *or* any raw CSV row label. Undeclared means the sweep
      index, never a guessed axis.
    - ~~`roi=1` vs `2`~~ — answered: two spatial ROIs on the CCD, the excitation
      spot and a remote spatially-filtered spot (two-spot galvo scans). Both are
      always loaded; the loader warns when the *selected* one is all zeros.
- Still open on this class:
    - Nested sweeps: a 2-D raster arrives as **one flattened file** (41×51 in the
      reflectance example) and a TRPL sweep as a **directory**. `sweep_grid()`
      detects and reports the raster shape; reshaping is the item below.
    - Field × power (a genuinely 2-parameter sweep) still unseen in a real file.
- HDF5 export/import: done (`hdf5.py`, `scan.to_hdf5()`, `.h5` accepted by the
  loader, both axis kinds). `dev/hdf5`'s `converters.py` remains unmerged — its CLI
  for bulk CSV→HDF5/TIFF conversion is the part not covered by this work, and the
  11.57 MB → 0.069 MB TRPL result is a decent argument for having one.
- AttoCubeRealSpace ...
    - Select the range of frames of interest -> work on a subset
    - **File ordering is lexicographic** (A7): `iter_10` sorts before `iter_2`.
      Escapes notice today only because the committed example is zero-padded.
      Copy `AttoCubeTRPLSweep._order_by_iter`, gap warning included.
- Position in scans (x,y).
    - for x (..) for y (...) can be flipped to for y (...) for x (..). reverse the nested loop
    - Reshape a raster sweep to `(n_points, n_y, n_x)` and plot it as a map.
      `sweep_grid()` already reports the shape and which axis is inner, so the
      loop-order flip is readable off the data rather than guessed.
- TRPL follow-ups (see E12):
    - No plotting: `_resolve_x_axis` knows only energy/wavelength. Let it ask the
      scan for its own axis instead of adding a third string.
    - No lifetime fitting: `fitting.py` has no exponential model. The baseline
      machinery is already model-agnostic; only `_fit_single_peak` hardcodes the
      3-parameter peak shape.
    - Time unit (ns) is inferred, not confirmed — `_TRPL_TIME_UNIT`. Any fitted
      lifetime inherits it.
- Reference class
    - Format spectra to return array, not per value sweeps

## Fitting
- Multiple ROIs for diffusion E.g. 2-3 diffusion spots. Track each for area and centre of mass
- Multiple ROIs for dipole length. E.g. in hybrid intralayer excitons in homobilayers with switchable dipole lengths based on the transitions

## Documentation
- Add mathematical formulas to functions that assume certain physics. E.g. the calculation of the electric field within the heterostructure stack.
