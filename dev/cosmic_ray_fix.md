# Wiring `remove_cosmic_rays` into `AttoCubeSpectralSweep`

**Date:** 2026-08-05 · **Audit ID:** E13 · **Branch:** `feature/AttoCube-loader-fix`

A review document. Line numbers are as of this change and will drift.

Files touched:

| File | What changed |
|---|---|
| `src/tmdc_optics_tools/loaders.py` | the feature — new `cosmic_rays=` argument, three new attributes, one behavioural fix to an existing sentinel |
| `src/tmdc_optics_tools/hdf5.py` | records the declaration as provenance |
| `src/tmdc_optics_tools/plotting.py` | one consistency fix in `_resolve_spectra` |
| `tests/test_loaders.py` | two optional arguments on the CSV fixture builder |
| `tests/test_loaders_cosmic_rays.py` | new, 14 cases |
| `dev/audit-2026-07.md` | E13 decision record |
| `.claude/CLAUDE.md` | one-line standing rule in *Settled* |

No existing public behaviour changes for a caller who does not pass `cosmic_rays=`,
with one exception noted in §6.

---

## 1. The starting position

`processing.remove_cosmic_rays` has existed since A1, with 18 tests. It is a pure
array function:

```python
remove_cosmic_rays(spectra, sigma_threshold=5.0, median_window=7, max_iter=3,
                   cross_sweep_veto=False, cross_sweep_window=5, axis=0)
    -> (cleaned, cr_mask)
```

Nothing called it. To use it a researcher wrote

```python
scan = AttoCubeSpectralSweep(path, spectra_type="PL", bg_region_nm=(700, 710))
clean, mask = processing.remove_cosmic_rays(scan.spectra)
```

and then carried `clean` by hand. Two things are wrong with that, and they are the
whole motivation:

1. **It arrives too late.** `scan.energy_spectra_bg` and `scan.contrast` were already
   built, from the *unrepaired* array, inside `__init__`. So `clean` is a fourth
   array that agrees with none of them, and any `fit_scan_peak` or plot that reaches
   for `best_energy_spectra` still sees the spikes.
2. **Nothing records it.** The scan does not know a repair happened, `__repr__` does
   not say so, and `to_hdf5` cannot write it down.

---

## 2. Decision one — where the correction runs

The question asked was: apply it at instance initialisation, or at plotting time?

**Answer: at initialisation, at the head of the wavelength-space correction chain.**
Three independent arguments rule out plotting time; each is sufficient on its own.

### 2.1 Detection space

Detection is built on the discrete Laplacian

```
L[i] = f[i-1] - 2·f[i] + f[i+1]
```

which is the uniform-spacing finite difference. It assumes the samples are evenly
spaced. That is true of the detector axis (CCD pixels are evenly spaced in λ to a
good approximation) and false of the energy axis, since `E = hc/λ` compresses the
spacing towards the blue. `plotting._resolve_spectra` already warns about this for
`spectra_source="raw"` — *"descending energy order and unequal pixel spacing"*.

So a plot-time call would have to re-enter wavelength space, clean, re-sort onto
ascending energy and re-apply the Jacobian. That is `_build_energy_spectra`'s job,
inside the loader; doing it in `plotting` breaks the module split (*"Plotting must not
re-implement maths that belongs in `processing`"*).

### 2.2 Consistency with the corrections that follow

The repair must precede three things, all of which read the counts as if they were
signal:

- **The `bg_region` window mean.** `subtract_background` takes the mean over a
  wavelength window. A spike inside that window inflates the mean and the excess is
  then subtracted from *every pixel of that sweep*. For a 4000-count spike in a
  31-pixel window that is a ~129-count pedestal error across the whole spectrum —
  quantified and pinned in the tests (§7).
- **The contrast ratio.** `(S − R)/R` is non-linear in both arms, so a spike does not
  merely add a local artefact; and `spectral_contrast`'s non-positive-reference guard
  decides which pixels become NaN, so a spike in `R` changes *which* pixels are
  excluded.
- **Any `fit_*` call**, which fits whichever array it was handed.

Clean at plot time and the figure is clean while `scan.energy_contrast`,
`scan.energy_spectra_bg` and every fit still contain the spikes. Two datasets under
one name — the failure mode is that the plot looks right.

### 2.3 Batching

`cross_sweep_veto` and the `PERSISTENT_FLAG_FRACTION` warning are both defined *over
the sweep axis* — see `processing.py:771-780`. `plot_power_series` takes
`sweep_step`, so a plot-time pass would run detection over a decimated subset and its
verdict would depend on the plot's stride. CLAUDE.md rules that out explicitly: a
default must produce *"results that do not depend on how the data happened to be
batched"*. Load time is the only place where the full `(n_pixels, n_sweeps)` block is
guaranteed present, which is the only place that diagnostic means anything.

### 2.4 The precedent that does not extend

`plot_power_series` already accepts a post-load `bg_region=`. That is not a licence
for a post-load `cosmic_rays=`: subtracting a scalar is space-agnostic, repeatable and
reversible, whereas replacing pixel values with local medians is none of the three.

### 2.5 What was given up, and the honest cost

Load time means **retuning `sigma_threshold` costs a re-read of the CSV.** That is a
real cost on a 300 MB export and it is the one thing plot-time would have been better
at. It is not paid in practice, because the exploratory loop needs no new code:

```python
scan = AttoCubeSpectralSweep(path, spectra_type="PL")      # read once
clean, mask = processing.remove_cosmic_rays(scan.spectra, sigma_threshold=4)
# look at mask, iterate on sigma_threshold in memory ...
```

and then the settled value is baked into the load call, where it is recorded and
where it feeds the rest of the chain. If that loop turns out to chafe, the fix is a
copy-returning `scan.with_cosmic_rays_removed(...)`, which needs `__init__` split into
a decode stage and a correct stage — a worthwhile refactor, a separate change, and not
needed for this to be correct.

---

## 3. Decision two — the shape of the argument

### 3.1 One dict, not six flags

```python
cosmic_rays : dict = None      # None = off, {} = defaults, {...} = tuned
```

The alternative was `remove_cosmic_rays: bool = False` plus five `cr_*` arguments on
an already 17-parameter `__init__`. Rejected under *parameters earn their place*: one
structural parameter absorbing a combinatorial space, the same shape as
`animate_panels(panels=…)`. It also keeps the docstring honest — the five keys are
documented once, in the function that implements them, rather than duplicated into a
second signature that can drift.

The cost of a dict is that a mistyped key would surface as a `TypeError` raised from
a function the caller never called. §4.3 pays that cost back with an explicit check.

### 3.2 Not a third axis in the array namespace

This is the part worth reviewing most carefully. The loader currently exposes five
arrays for two corrections crossed together:

```
                    apply_jacobian=False        apply_jacobian=True
no background       energy_spectra          ×   energy_spectra
                                                energy_spectra_pre_jacobian
background          energy_spectra_bg       ×   energy_spectra_bg
reference           contrast, energy_contrast
```

Adding cosmic-ray removal as a **third independent flag** would nominally double that
to ten names, and force `best_energy_spectra` to pick among them.

It avoids that because a repair is **not an alternative representation** of the signal
the way the Jacobian and the background branch are — those two give you the same
counts expressed differently or with a pedestal removed, and you legitimately want
both versions available. A repair is a claim that certain pixel values were never
signal at all. So it belongs *upstream of the branch point*, replacing the array that
feeds every branch, rather than adding a branch.

Net cost: **two arrays, and no new energy-space names.**

| New attribute | Type | Meaning |
|---|---|---|
| `spectra_cr` | `(n_pixels, n_sweeps)` float, or `None` | wavelength-space counts with flagged pixels replaced by local medians |
| `cosmic_ray_mask` | `(n_pixels, n_sweeps)` bool, or `None` | which pixels were replaced |
| `cosmic_rays` | `dict` or `None` | the declaration, as given |

`cosmic_ray_mask` exists because of *return the evidence*: the one risk of this
correction is that a real narrow feature is replaced and the result simply looks
clean. `cosmic_ray_mask.mean(axis=1)` is the diagnostic — a pixel flagged in most
sweeps is a detector defect or a real spectral line, because a cosmic ray cannot
recur at the same pixel in the next exposure.

`spectra` is **not** reassigned. It stays the file's own counts, per *raw arrays are
never mutated after load*.

---

## 4. Trace — `loaders.py`

### 4.1 The accepted-key set (`loaders.py:1934-1942`)

```python
_COSMIC_RAY_KEYS = frozenset(
    inspect.signature(processing.remove_cosmic_rays).parameters
) - {"spectra", "axis"}
```

Resolves at import to

```python
{'cross_sweep_veto', 'cross_sweep_window', 'max_iter', 'median_window', 'sigma_threshold'}
```

Read off the target function rather than written out, so the two cannot drift apart
when that signature grows. `import inspect` was added at `loaders.py:30`.

Two keys are excluded, for different reasons:

- **`spectra`** — the array is the one this loader just read. Accepting it would let a
  caller point the repair at something else entirely.
- **`axis`** — the class stores `(n_pixels, n_sweeps)`, so `axis=0` is a property of
  the class, not a choice. Accepting it would let a caller pass `axis=1` and transpose
  the *detection* while the stored shape stayed the same: cleaning along the sweep
  direction, which is not what a cosmic ray is anomalous along. Silently wrong
  numbers, so it is refused rather than documented.

### 4.2 Signature (`loaders.py:2336`)

```python
        geometry        : DeviceGeometry = None,
        cosmic_rays     : dict  = None,        # <-- new
        bg_region_nm    : tuple = None,
```

Placed **before** the background arguments so the signature reads in pipeline order:
repair → background → reference/contrast → Jacobian. Keyword-only (the whole block is,
after `*`), so the insertion cannot break a positional call.

### 4.3 Validation, before the read (`loaders.py:2349-2364`)

```python
        # Both checks precede the read: an export is large enough that a mistyped
        # argument should not cost the decode before it is reported.
        if bg_region_nm is not None and bg_region_eV is not None:
            raise ValueError(...)
        if cosmic_rays is not None:
            unknown = set(cosmic_rays) - _COSMIC_RAY_KEYS
            if unknown:
                raise ValueError(
                    f"cosmic_rays received unknown key(s) {sorted(unknown)}. ..."
                )
```

Produces, for `cosmic_rays={"sigma": 4}`:

```
cosmic_rays received unknown key(s) ['sigma']. Accepted: ['cross_sweep_veto',
'cross_sweep_window', 'max_iter', 'median_window', 'sigma_threshold'] — these are
forwarded to processing.remove_cosmic_rays, whose 'spectra' and 'axis' are set by
the loader. Pass cosmic_rays={} to accept every default.
```

The comment is the reason for the *placement*, not just the check: `_decode_and_describe`
is the expensive call, and a typo should not cost a multi-hundred-MB read before it is
reported. The existing `bg_region` mutual-exclusion check was already here; the new
comment now covers both.

### 4.4 The repair, at the head of the chain (`loaders.py:2445-2464`)

Position: after `self.spectra` has been selected by `roi` (line 2382) and after the
`apply_jacobian`-with-no-background warning, immediately before the energy axis is
built.

```python
        self.cosmic_rays = dict(cosmic_rays) if cosmic_rays is not None else None
        if cosmic_rays is None:
            self.spectra_cr      = None
            self.cosmic_ray_mask = None
        else:
            self.spectra_cr, self.cosmic_ray_mask = processing.remove_cosmic_rays(
                self.spectra, axis=0, **cosmic_rays
            )

        signal = self.spectra if self.spectra_cr is None else self.spectra_cr
```

Three points:

- `dict(cosmic_rays)` **copies** the declaration. The caller's dict cannot mutate what
  the scan reports afterwards, and what `to_hdf5` writes is what was actually used.
- `axis=0` is passed positionally in the call, not taken from `cosmic_rays` — which is
  why `axis` is excluded from the accepted keys. A caller who passed it would have hit
  a duplicate-keyword `TypeError` anyway; the explicit check turns that into a
  sentence that explains itself.
- **`signal` is a local, not an attribute.** It is the single point at which "which
  array does everything downstream read" is decided. There is deliberately no
  `self.signal`: it would be a fourth name for an array already reachable as either
  `spectra` or `spectra_cr`, and callers choosing between them would have to know
  which.

### 4.5 What `remove_cosmic_rays` does with it

For completeness of the trace, the callee (`processing.py:637-790`) for 2-D input:

```
remove_cosmic_rays(spectra=(n_pixels, n_sweeps), axis=0)
  │  arr = np.asarray(spectra, float);  flip = False   (axis==0, so no transpose)
  │  median_window forced odd
  ├─ for each sweep column j:
  │     _detect_cosmic_rays_1d(work[:, j], sigma_threshold, median_window, max_iter)
  │         iterate up to max_iter:
  │           laplacian = f[i-1] - 2f[i] + f[i+1]
  │           sigma_lap = MAD(laplacian over unflagged pixels) / 0.6745
  │           flag where laplacian < -sigma_threshold · sigma_lap
  │           replace flagged with local median, recompute        <- catches flat-top
  │         -> (cr_mask[:, j], working[:, j])                        multi-pixel CRs
  ├─ if cross_sweep_veto:  cr_mask = _cross_sweep_veto(...)   (can only remove flags)
  │  elif n_sweeps >= 3:   _warn_persistent_flags(cr_mask)    (UserWarning)
  ├─ cleaned = work.copy();  cleaned[flagged] = median_filter(working)[flagged]
  └─ return (cleaned, cr_mask)
```

Two consequences the loader relies on:

- Detection is **per exposure**. The MAD noise estimate is taken within one spectrum,
  because PL intensity can move by an order of magnitude across a gate sweep, so a
  sweep-wide noise estimate would be meaningless. That is also why the result does not
  depend on batching (§2.3).
- `cleaned` is a fresh `work.copy()`, so `scan.spectra_cr is not scan.spectra` always
  holds — which §4.6 depends on.

With the default `cross_sweep_veto=False` and ≥3 sweeps, a repair at load time can now
emit a `UserWarning` from `_warn_persistent_flags`. That is intended and consistent
with the other load-time warnings (the all-zero ROI, `apply_jacobian` without a
background): the conservative default is the one that can quietly remove a real
feature, so it says so.

### 4.6 Everything downstream reads `signal` (`loaders.py:2467-2527`)

Four call sites changed from `self.spectra` to `signal`:

```python
        self.energy_spectra = self._build_energy_spectra(
            signal, self.wavelength, _sort_idx, apply_jacobian)              # (1)

        if apply_jacobian:
            self.energy_spectra_pre_jacobian = self._build_energy_spectra(
                signal, self.wavelength, _sort_idx, apply_jacobian=False)    # (2)
        else:
            self.energy_spectra_pre_jacobian = self.energy_spectra

        corrected = signal                                                   # (3)
        if self.bg_region_nm is not None:
            corrected = subtract_background(corrected, bg_region=self.bg_region_nm,
                                            x=self.wavelength, axis=0)
        if self.bg_spectrum is not None:
            corrected = processing.subtract_spectrum(corrected, self.bg_spectrum, axis=0)

        if corrected is not signal:                                          # (4)
            self.energy_spectra_bg = self._build_energy_spectra(
                corrected, self.wavelength, _sort_idx, apply_jacobian)
        else:
            self.energy_spectra_bg = None
```

`contrast` is built from `corrected` further down and so inherits the repair through
(3), as does `energy_contrast`.

**(4) is the one that would have been a bug.** It is an *identity* test asking "did
either background mechanism actually run?", relying on `subtract_background` /
`subtract_spectrum` returning new arrays. Left as `corrected is not self.spectra`,
then with `cosmic_rays=` declared and **no** background:

```
corrected  is  signal  is  self.spectra_cr        # nothing subtracted
self.spectra_cr  is not  self.spectra             # always true, §4.5
⇒ the test passes, energy_spectra_bg is set from an array with no background removed
⇒ best_energy_spectra returns it (it prefers energy_spectra_bg when non-None)
⇒ plotting's spectra_source="energy_bg" becomes available and lies
```

i.e. a repair would silently present itself as a background subtraction. Changing the
comparison to `signal` makes the test mean what it always claimed to mean. Pinned by
`test_repair_alone_leaves_no_background_array`.

### 4.7 Resulting state, by declaration

| Loaded with | `spectra` | `spectra_cr` | `signal` is | `energy_spectra_bg` |
|---|---|---|---|---|
| nothing | file counts | `None` | `spectra` | `None` |
| `cosmic_rays={}` | file counts | repaired | `spectra_cr` | `None` |
| `bg_region_nm=…` | file counts | `None` | `spectra` | from `spectra` |
| `cosmic_rays={}`, `bg_region_nm=…` | file counts | repaired | `spectra_cr` | from `spectra_cr` |
| `cosmic_rays={}`, `reference=…` | file counts | repaired | `spectra_cr` | `None`; `contrast` from `spectra_cr` |

### 4.8 `__repr__` (`loaders.py:2774-2779`)

```python
        if self.cosmic_ray_mask is not None:
            n_flagged = int(self.cosmic_ray_mask.sum())
            lines.append(f"  {'Cosmic rays':<{w}}: {n_flagged} pixel"
                         f"{'' if n_flagged == 1 else 's'} replaced")
```

Placed before the `BG region` line, so the repr lists corrections in the order they
ran. Absent entirely when no repair was declared — the repr states what happened, and
does not advertise what did not.

### 4.9 Docstrings

`cosmic_rays` in *Parameters*; `spectra_cr`, `cosmic_ray_mask`, `cosmic_rays` in
*Attributes*; `source_metadata` extended. Written to the *contract, not a changelog*
rule: what it takes, what it refuses, that it runs first and why that matters to a
caller (the pedestal and the ratio), and no reference to this document or to E13.

---

## 5. Trace — `hdf5.py`

Governed by G2: **HDF5 stores no derived arrays and never replays corrections on
read.** The repaired array is a derived array, so it is not written; the declaration
is a loading choice, so it is written as provenance beside `apply_jacobian`.

Write path (`hdf5.py:305-311`):

```python
        if scan._LAYOUT_KIND == "spectral":
            meta.attrs["roi"]            = int(scan.roi)
            meta.attrs["apply_jacobian"] = bool(scan.apply_jacobian)
            if scan.cosmic_rays is not None:
                meta.attrs["cosmic_rays"] = json.dumps(scan.cosmic_rays)
```

Written only when declared, so a round trip cannot turn "no repair" into "a repair
with default parameters".

Read path — **one line**, because the reader is already generic over the mapping-valued
attributes (`hdf5.py:144`):

```python
_JSON_ATTRS = ("curated_labels", "curated_scales", "gates", "cosmic_rays")
```

`read_sweep` loops over that tuple, `json.loads`-ing each present key and setting
`None` otherwise, and the result lands in `scan.source_metadata`. Nothing reads
`source_metadata["cosmic_rays"]`, which is the point: it is a record, not an
instruction.

**The consequence, stated plainly:** `to_hdf5` → reload loses the repair. The
reloaded scan has `cosmic_rays is None` and `spectra_cr is None`, and
`source_metadata["cosmic_rays"]` says what the writing session did. Pass
`cosmic_rays=` again to repeat it. This matches `apply_jacobian` and `bg_region_nm`
exactly, so it is consistent rather than surprising — but it is behaviour worth
knowing, so it is in the class docstring and pinned by
`test_hdf5_records_the_declaration_without_replaying_it`.

---

## 6. Trace — `plotting.py`, and the one behaviour change

`_resolve_spectra` maps `spectra_source=` onto an array. `"raw"` still means
`scan.spectra`, the file's own counts — that is what the word means, and no new key
was added. `_SPECTRA_SOURCES` is unchanged.

`"best"` did change. Before:

```python
    if src == "best":
        arr = scan.best_energy_spectra if x_axis == "energy" else scan.spectra
```

The `x_axis="wavelength"` branch returned raw spectra because wavelength space had
nothing better to offer — `best_energy_spectra` picks between two *energy*-space
arrays, and no `spectra_bg` is stored. A cosmic-ray repair changes that: `spectra_cr`
is a better wavelength-space array. After:

```python
    if src == "best":
        if x_axis == "energy":
            arr = scan.best_energy_spectra
        else:
            arr = getattr(scan, "spectra_cr", None)
            if arr is None:
                arr = scan.spectra
```

Leaving it alone would have meant `spectra_source="best"` on a wavelength axis showing
spikes that no other source shows, on a scan that declared a repair — the same silent
inconsistency §2.2 rules out, reintroduced one layer up. `getattr` with a default
because `_resolve_spectra` may be handed an object without the attribute
(`SingleSpectrum`), and it should degrade rather than raise.

`_SPECTRA_SOURCE_LABELS["best"]` updated to `"best available (repaired, bg-corrected
if set)"` to stay true.

**This is the only change visible to a caller who never passes `cosmic_rays=`** — and
for them `spectra_cr` is `None`, so the returned array is identical.

---

## 7. Trace — the tests

### 7.1 Why the fixture builder changed

The MAD noise estimate is taken on the Laplacian of one spectrum. `make_spectral_csv`
writes `_roi1(r, i) = 100 + 10r + i` — a perfectly linear ramp, whose Laplacian is
identically zero, so `sigma_lap == 0` and `_detect_cosmic_rays_1d` breaks out on its
first iteration (`processing.py:526-527`). Detection cannot be exercised against it.

Rather than fork a second builder — *reuse before adding* — `make_spectral_csv` gained
two optional overrides:

```python
def make_spectral_csv(path, params=None, zero_blocks=0, interleave=False,
                      roi1=None, wavelength=None):
```

`wavelength` also sets the pixel count (`n_pixels = wl.size`, replacing the module
constant in the row loop), because a realistic spectrum needs more pixels than the
10-row default. Rows past the labelled parameters are written unlabelled, which is
what the real export does. Existing callers pass neither and are unaffected.

### 7.2 `tests/test_loaders_cosmic_rays.py` — 14 cases

Fixture: 400 pixels × 3 sweeps, a Gaussian peak (centre 250, σ 25) on a 100-count
pedestal with σ=3 noise, amplitudes 400/1200/2000 across the sweeps. Two 4000-count
spikes planted — one in the continuum at pixel 80, one at pixel 20 which lies **inside
the background window**.

*Opt-in*

- `test_absent_by_default` — all three attributes `None`, and the spike is still in
  `spectra`, proving the fixture would have shown a repair had one run.
- `test_empty_dict_opts_in` — `{}` is a declaration; mask set, value reduced.
- `test_spectra_is_never_repaired_in_place` — `spectra` still equals the planted
  array; `spectra_cr is not spectra`.
- `test_broad_peak_is_left_alone` — nothing in pixels 240–260 flagged. A 2000-count
  peak is not a spike, and this is the test that fails if the threshold logic ever
  starts keying on amplitude rather than curvature.

*Position in the chain*

- `test_energy_spectra_are_built_from_the_repair` — `energy_spectra` and
  `energy_spectra_pre_jacobian` both equal `spectra_cr` reordered onto ascending
  energy. This is §4.6 (1) and (2).
- `test_repair_alone_leaves_no_background_array` — `energy_spectra_bg is None` and
  `best_energy_spectra is energy_spectra`. This is the §4.6 (4) sentinel.
- `test_spike_in_the_window_does_not_bias_the_pedestal` — **the ordering test.** Loads
  the same file twice, with and without the repair, both with `bg_region_nm` set. The
  window holds 31 pixels, so the unrepaired load over-subtracts 4000/31 ≈ 129 counts
  from every pixel of the affected sweep; the test pins that difference to 5% over
  pixels 200–300 (away from the repaired pixel, where the arrays also differ by the
  replacement itself), and pins that the unaffected sweep is identical in both loads.
  This is the numerical statement of §2.2.
- `test_contrast_is_formed_from_the_repair` — with a flat 500-count reference,
  `contrast[CR_PIXEL, CR_SWEEP]` equals `(spectra_cr - 500)/500`, i.e. the ratio is
  formed from the repaired numerator.

*The declaration*

- `test_unknown_key_raises`, `test_loader_owned_arguments_are_rejected`
  (parametrised over `spectra`, `axis`) — §4.1 and §4.3.
- `test_declaration_is_checked_before_the_file_is_read` — points at a path that does
  not exist and asserts the `ValueError` about the key, not a `FileNotFoundError`.
  That is what pins the *ordering* of the check against the decode, which is otherwise
  invisible.
- `test_repr_reports_the_repair`.
- `test_hdf5_records_the_declaration_without_replaying_it` — provenance present,
  repair absent, stored spectra raw.

### 7.3 Result

```
tests/test_loaders_cosmic_rays.py   14 passed
full suite                          238 passed, 1 failed
python -m mkdocs build --strict     green
```

The one failure is **pre-existing and unrelated**:
`tests/test_contrast.py::test_spectra_type_not_mutated_by_supplying_a_reference`
expects `signal_label == "Reflectance (counts)"` while `SIGNAL_LABELS["R"]` gives
`"Reflected intensity (counts)"`. Verified by stashing this change and re-running:
it fails identically on the clean tree. It is not in the audit's known-issues list, so
it looks like a genuine unreported mismatch between the test and `constants.py`.
Which of the two is right is a naming call, so it was left alone — **your decision**.

---

## 8. Deliberately not done

- **`SingleSpectrum`** — has `apply_jacobian` and a background already, and
  `remove_cosmic_rays` handles 1-D input, so this is a straightforward follow-up. Left
  out to keep the change to one class.
- **`AttoCubeTRPLSweep`** — needs its own judgement, not this signature copied. A
  decay's sharp rise at t₀ is a large negative Laplacian over 1–3 bins, which is
  exactly the signature the detector keys on, so the default threshold would attack
  the physics. Whether a time-axis repair is wanted at all, and with what guard on the
  rise, is a physics question.
- **A `spectra_source="cosmic_rays"` plotting key** — `spectra_cr` is reachable on the
  scan, `"best"` already prefers it in wavelength space, and `plotting` is owed a pass
  under E12. Adding a key now would be a second decision inside this one.
- **A `with_cosmic_rays_removed()` method** — §2.5. Wanted only if the retune loop
  turns out to chafe, and it needs the `__init__` decode/correct split first.

## 9. Points to push back on

If any of these read wrong, they are the load-bearing choices:

1. **`axis` is refused rather than forwarded** (§4.1). Defensible as protecting the
   class's shape convention; arguable as over-restriction.
2. **`"best"` now prefers `spectra_cr` on a wavelength axis** (§6). The only behaviour
   change to existing plotting code paths, though inert without a declaration.
3. **HDF5 does not round-trip the repair** (§5). Consistent with G2 and with
   `apply_jacobian`, but it means an archived, repaired scan reloads unrepaired.
4. **The `dict` over flat flags** (§3.1) — trades discoverability for a signature that
   does not grow, and buys the loss back with an explicit key check.
5. **One line was added to `.claude/CLAUDE.md`** in *Settled — don't re-litigate*.
   Drop it if you would rather the rule lived only in E13.
