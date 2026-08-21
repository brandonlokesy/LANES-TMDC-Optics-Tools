# 0031 — The spectral map's mesh runs one row per sweep point

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-21 |
| **Audit** | E3 |

## Context

`plot_spectral_map` built its coordinate grids by hand:

```python
x_m = np.tile(x[:, np.newaxis], (1, y.size))
y_m = np.tile(y[np.newaxis, :], (scan.n_pixels, 1))
mesh = ax.pcolormesh(x_m, y_m, data, shading="auto", ...)
```

`pcolormesh` accepts a 1-D `x`/`y` pair and builds the mesh itself, so the two
`(n_pixels, n)` arrays were work the call already does. At 1340 × 161 — a real scan from
`examples/data/1L-WSe2-PL` — the pair is 3.5 MB per call. Matplotlib copies whatever
coordinates it is given (`_pcolorargs`, `matplotlib/axes/_axes.py:6028`) and then expands
a 1-D pair with `repeat` (`:6042-6047`), so the tiled form cost two arrays plus two
copies where the 1-D form costs one expansion; the cell-edge interpolation that follows
is identical either way.

The obstacle is not memory but orientation. With a 2-D coordinate pair, `pcolormesh`
treats the arrays as an unordered list of cell corners and never asks which axis is
which. With a 1-D pair it does: `C` must be `(len(y), len(x))`. The map's `data` is
`(n_pixels, n)` — detector pixels down the rows — and `x` is the spectral axis, so the
block has to be handed over transposed. That is almost certainly why the tiled form was
written: it sidesteps having to decide.

The transpose is observable. `mesh` is the third member of the return, and
`mesh.get_array()` and `mesh._coordinates` come back in whichever order the call
established.

## Decision

1. `plot_spectral_map` passes the 1-D pair and a transposed block:
   `ax.pcolormesh(x, y, data.T, ...)`.
2. `mesh.get_array()` therefore runs `(n_sweep_points, n_pixels)`, and
   `mesh._coordinates` runs `(n_sweep_points + 1, n_pixels + 1, 2)`. The `Returns` block
   states this.
3. `tests/test_plotting_spectral_map.py` gains a test that pins the orientation by name,
   separate from the tests that depend on it.

## Rejected

**`np.broadcast_to` instead of `np.tile`.** Zero-storage read-only views of the same
shape, so the 2-D call and the existing orientation both survive and no test or caller
changes at all. It saves the same memory, because matplotlib copies the views into real
arrays at `_axes.py:6028` either way — so this was a genuine candidate, not a worse
version of the same thing. Rejected on readability:
`np.broadcast_to(x[:, None], (scan.n_pixels, y.size))` states a broadcasting trick where
`pcolormesh(x, y, data.T)` states what is being drawn, and the 2-D form keeps the
property that made the original bug possible — matplotlib accepts *any* two equal-shaped
arrays as coordinates, so a wrong pair draws a wrong figure rather than raising. With the
1-D pair both lengths are checked against `C`.

**Keeping `np.tile` and documenting the cost.** 3.5 MB is not a lot. Rejected because E3
recorded it as a defect and the 1-D call is shorter, not longer: nothing is being traded
for the saving except the orientation, which is now written down.

**Transposing inside the function to preserve the old return.** `data.T` in, then
handing back something re-transposed. Rejected because the artist is matplotlib's and
holds one array; the only way to preserve the old orientation would be to keep the 2-D
call.

## Consequences

- A caller reading data back off the mesh sees the other orientation. Nothing in the
  repository did except the tests, and `examples/example_1L_WSe2.ipynb` reads only
  `.min()` and `.max()` off it, which are orientation-free.
- **`tests/test_cosmic_rays_downstream.py` held the one place this could have gone wrong
  quietly.** `_map_column` did
  `np.asarray(mesh.get_array()).reshape(scan.n_pixels, scan.n_sweeps)`. A `reshape` is
  not a transpose: after the change it would have scrambled the array rather than
  failing. It now indexes the row.
- The comment warning that `scan.n_sweeps` describes the whole flat sweep while a pinned
  nest draws fewer points is gone with the `np.tile` lines. The new call never reaches
  for a length, so the trap no longer exists to be documented.
- Verified on `examples/data/1L-WSe2-PL` at 1340 × 161: the quads, the colour values,
  the colour limits and both axis limits are identical to what the tiled call drew.

## Load-bearing choices

That the orientation is worth stating in the `Returns` block rather than left as
matplotlib's business. If a second plotting function ever returns a `QuadMesh`, the two
should agree, and the statement is what makes disagreement visible.
