# 0001 — Cosmic-ray repair runs at load time, declared by one dict

| | |
|---|---|
| **Status** | Accepted · amended by A17 (the correction-ladder naming; record not yet migrated) |
| **Date** | 2026-08-05 |
| **Audit** | E13 |

## Context

`processing.remove_cosmic_rays` already existed as a pure array function with tests,
and nothing called it. To use it, a researcher loaded a scan, called the function on
`scan.spectra`, and carried the cleaned array by hand.

Two things are wrong with that, and together they are the whole motivation:

1. **It arrives too late.** The derived arrays — background-corrected, energy-axis,
   contrast — are built inside `__init__` from the *unrepaired* counts. So the cleaned
   array is one more array that agrees with none of them, and anything reaching for the
   "best available" spectra still sees the spikes.
2. **Nothing records it.** The scan does not know a repair happened, `__repr__` cannot
   say so, and the HDF5 writer cannot write it down.

## Decision

### 1. The repair runs at initialisation, at the head of the wavelength-space chain

Three independent arguments, each sufficient on its own.

**Detection space.** Detection is built on the discrete Laplacian
`L[i] = f[i-1] - 2·f[i] + f[i+1]`, the uniform-spacing finite difference. It assumes
evenly spaced samples — true of the detector axis, false of the energy axis, since
`E = hc/λ` compresses the spacing towards the blue. A plot-time call would therefore
have to re-enter wavelength space, clean, re-sort onto ascending energy and re-apply
the Jacobian, which is the loader's job; doing it in `plotting` also breaks the rule
that plotting must not re-implement maths belonging in `processing`.

**Consistency with the corrections that follow.** The repair must precede three things,
all of which read the counts as if they were signal:

- *The background window mean.* A spike inside the window inflates the mean, and the
  excess is then subtracted from every pixel of that sweep — turning a local artefact
  into a pedestal error across the whole spectrum.
- *The contrast ratio.* `(S − R)/R` is non-linear in both arms, and the
  non-positive-reference guard decides which pixels become NaN, so a spike in the
  reference changes *which* pixels are excluded.
- *Any fit*, which fits whichever array it was handed.

Clean at plot time and the figure is clean while the contrast, the
background-corrected array and every fit still contain the spikes. Two datasets under
one name, and the failure mode is that the plot looks right.

**Batching.** The cross-sweep veto and the persistent-flag warning are both defined
*over the sweep axis*. Plot functions accept a sweep stride, so a plot-time pass would
run detection over a decimated subset and its verdict would depend on the plot's
stride. Load time is the only place the full `(n_pixels, n_sweeps)` block is guaranteed
present, and therefore the only place that diagnostic means anything.

### 2. The declaration is one dict, and adds no branch to the array namespace

```python
cosmic_rays: dict = None      # None = off, {} = defaults, {...} = tuned
```

A repair is **not an alternative representation** of the signal the way the Jacobian
and the background are. Those give you the same counts expressed differently or with a
pedestal removed, and you legitimately want both versions available. A repair is a
claim that certain pixel values were never signal at all — so it belongs *upstream of
the branch point*, replacing the array that feeds every branch, rather than adding a
branch.

| New attribute | Type | Meaning |
|---|---|---|
| `spectra_cr` | `(n_pixels, n_sweeps)` float, or `None` | wavelength-space counts with flagged pixels replaced by local medians |
| `cosmic_ray_mask` | `(n_pixels, n_sweeps)` bool, or `None` | which pixels were replaced |
| `cosmic_rays` | `dict` or `None` | the declaration, as given |

## Rejected

**Applying it at plotting time.** Ruled out by all three arguments above. The existing
post-load `bg_region=` on a plot function is not a precedent for it: subtracting a
scalar is space-agnostic, repeatable and reversible, whereas replacing pixel values
with local medians is none of the three.

**A `remove_cosmic_rays: bool` flag plus five `cr_*` arguments.** On an `__init__` that
already took seventeen parameters. Rejected under *parameters earn their place* — one
structural parameter absorbing a combinatorial space. It also keeps the documentation
honest: the keys are described once, in the function that implements them, rather than
duplicated into a second signature that can drift.

**A third independent correction flag.** Crossed with the Jacobian and the background,
it would nominally double the energy-space array names to ten and force the
"best available" accessor to choose among them.

**Reassigning `spectra` to the repaired array.** `spectra` stays the file's own counts,
per *raw arrays are never mutated after load*.

**A `spectra_source="cosmic_rays"` plotting key.** `spectra_cr` is reachable on the
scan and `"best"` already prefers it in wavelength space. Adding a key would be a
second decision inside this one.

**A `with_cosmic_rays_removed()` copy-returning method.** Wanted only if the retune
loop below turns out to chafe, and it needs `__init__` split into a decode stage and a
correct stage first — a worthwhile refactor, and a separate change.

**Extending it to TRPL by copying this signature.** A decay's sharp rise at t₀ is a
large negative Laplacian over one to three bins, which is exactly the signature the
detector keys on, so the default threshold would attack the physics. Whether a
time-axis repair is wanted at all, and with what guard on the rise, is a physics
question and not answered here.

**Extending it to `SingleSpectrum` in the same pass.** The array function handles 1-D
input and that class already has a Jacobian and a background, so it is a
straightforward follow-up; left out to keep this change to one class.

## Consequences

- **Two new arrays and no new energy-space names.** The "best available" accessors never
  have to choose.
- **The mask comes back to the caller**, per *return the evidence*. The one risk of this
  correction is that a real narrow feature is replaced and the result simply looks
  clean. `cosmic_ray_mask.mean(axis=1)` is the diagnostic: a pixel flagged in most
  sweeps is a detector defect or a real spectral line, because a cosmic ray cannot
  recur at the same pixel in the next exposure.
- **Retuning the threshold costs a re-read of the file.** The real cost of load time,
  and the one thing plot time would have been better at. It is not paid in practice,
  because the exploratory loop needs no new code: load once, call the pure function on
  `scan.spectra` in memory, iterate on the threshold while looking at the mask, then
  bake the settled value into the load call where it is recorded and where it feeds the
  rest of the chain.
- **A mistyped dict key would otherwise surface as a `TypeError` raised from a function
  the caller never called**, so the accepted keys are checked explicitly at the call
  site.
- **HDF5 records the declaration as provenance but does not replay the repair on read.**
  Consistent with the rule that HDF5 stores no derived arrays, but it means an
  archived, repaired scan reloads unrepaired.
- **`"best"` prefers `spectra_cr` on a wavelength axis.** The only behaviour change to
  existing plotting paths, and inert without a declaration.

## Load-bearing choices

If something downstream later feels wrong, these are the parts to revisit first.

1. **`axis` is refused rather than forwarded** to the array function. Defensible as
   protecting the class's shape convention; arguable as over-restriction.
2. **`"best"` preferring `spectra_cr`** in wavelength space.
3. **HDF5 not round-tripping the repair.**
4. **The dict over flat flags** — trades discoverability for a signature that does not
   grow, and buys the loss back with the explicit key check.
