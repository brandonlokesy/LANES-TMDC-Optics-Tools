# 0023 — The cosmic-ray replacement median ignores the pixels it is repairing

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-18 |

## Context

`remove_cosmic_rays` flags a spike and replaces each flagged pixel with a local median.
Both the per-iteration fill inside `_detect_cosmic_rays_1d` and the final replacement
took that median with `scipy.ndimage.median_filter` over the whole spectrum, so a
flagged pixel was still inside its own median window. Once a flagged run reached half
the window, the median was drawn from the spike itself, and the value it wrote was then
self-consistent: every later pass returned the same contaminated number.

The boundary was a run of `(median_window - 1) / 2` pixels — 3 at the default window of
7, which is exactly the "1–3 pixels wide" the docstring gave as the definition of a
cosmic ray. Nothing wider had ever worked, and the tests planted spikes only 1 and 3
pixels wide, so nothing said so.

Measured on `examples/data/1L-WSe2-PL/PL_power_sweep_26_08_05_15_59_21_iter_0.csv`,
sweep index 2, a spike 4 pixels wide on a 590-count baseline: **[verified by running]**

```
 px    raw   repaired   flagged
643    606      606        0
644    920      920        1     <- kept its own raw value
645   1980      920        1
646   3412      920        1
647   2503      920        1
648    614      614        0
```

All four were detected; three were replaced with 920, a value taken from the fourth. A
330-count plateau survived. After the Jacobian and the `bg_region_eV` subtraction that
residual still stood at 149 714 against a whole-map maximum of 551 135, so the spike
remained the brightest point on the low-power end of the spectral map.

On a **flat-topped** spike the same contamination also stalled detection: the fills the
Laplacian was recomputed against were themselves spike values, so the interior was never
reached and only the two edge pixels were ever flagged.

## Decision

A flagged pixel takes the median of the pixels in its window that are **not** flagged.
`_fill_flagged` gathers each flagged pixel's window, sets the flagged neighbours to NaN,
and takes `np.nanmedian`; it is the single implementation, used by the detection
iterations and by both final-replacement paths.

Where a whole window is flagged there is no median. Those pixels keep their raw values
and `remove_cosmic_rays` raises a `UserWarning` naming them and pointing at
`median_window`. It does not widen the window on their behalf: how wide a window is
still local is a property of the spectrograph and the measurement, not of the array.

## Rejected

- **Widen the default `median_window` to 11.** One line, no new logic, and it repairs the
  file that prompted this. It moves the boundary and leaves the cause: the median is
  still drawn from the spike as soon as a run reaches half of the wider window, and the
  failure is still silent. Measured, a 6-pixel run fails at `median_window=11` exactly as
  a 4-pixel run fails at 7.
- **Interpolate linearly across each flagged run.** Robust and conventional, and it has
  no majority-contamination failure at all. Rejected because it abandons the documented
  local-median semantics for a different estimator, which changes every repaired value in
  every existing analysis, not only the ones that were wrong.
- **Grow the window automatically until a clean pixel falls inside it.** Always returns a
  value, but the window a caller asked for is then not the window used, and nothing says
  by how much it grew. A repair that cannot be done is worth reporting.
- **Warn whenever a fresh detection pass still flags something in the cleaned array.**
  This was measured as a way to catch the flat-top stall as well. It fires 459 times on
  `PL_20uW_Vbot_sweep_*.csv`, because the per-spectrum sigma is estimated over the whole
  dispersion axis and over-flags the shot-noise on a bright peak — see the open finding on
  the noise estimate. A diagnostic that fires hundreds of times on good data trains the
  reader to ignore it.

## Consequences

- A cosmic ray up to `median_window - 1` pixels wide is repaired to the local baseline
  where its profile is curved at every pixel, so every pixel is detected. That is 6 at
  the default window, against 3 before.
- A **flat-topped** run is bounded more tightly, and still quietly: repair is complete up
  to `median_window // 2 + 3` pixels (6 at the default), and a wider one keeps its
  interior with no warning, because the pixels that were flagged each had a clean
  neighbour to draw a median from. Stated in the `Notes` of `remove_cosmic_rays`. Closing
  this needs a local noise estimate, which is the same missing piece as the over-flagging
  finding, and is left to that work.
- The fill no longer depends on where the medians are drawn from, since flagged positions
  are excluded either way. `_detect_cosmic_rays_1d` therefore no longer returns its
  partially-cleaned spectrum, the `workings` array is gone from `remove_cosmic_rays`, and
  the veto's separate "rebuild the fills against the surviving mask" step is gone with it
  — a vetoed pixel is simply not in the mask, so the fill of a copy of the raw counts
  leaves it alone.
- Three copies of the replacement (the detection fill, the 1-D path, the 2-D path) are one.
- `median_window` is documented as what it is: the replacement window only. The noise
  estimate is the MAD of the Laplacian and never used it.
