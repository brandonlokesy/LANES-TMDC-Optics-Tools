# 0026 — `plot_image` carries the coordinate mapping, as two named parameters

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-19 |

## Context

A `RamanMap` is counts on a position grid. The micrometres live in `RamanMap.x` and
`RamanMap.y`; the `counts` array carries only row and column numbers. Drawing that map
in physical units therefore needs two things `imshow` takes at creation time: the data
coordinates of the image's outer edges (`extent`), and which end of the array's row axis
is drawn at the top (`origin`).

`origin` carries more weight than it looks. `RamanMap` builds
`y = np.unique(xy[:, 1])` ascending and fills `counts[iy, ix]`, so **row 0 holds the
smallest Y**. With `extent=(x.min(), x.max(), y.min(), y.max())` the bottom edge is
labelled `y.min()`. Under `origin="upper"` — matplotlib's default — row 0 is drawn at
the *top*, so the map is vertically mirrored against axis labels that read correctly.
Nothing about the resulting figure looks wrong.

`dev/design-principles.md` §2 pulls two ways on `extent`, and this record exists to
settle which way. Its test — "what changes what the data *is* — which correction ran,
which array was plotted, what the axis means — belongs in the signature" — puts `extent`
in. Its styling subsection — "Never add a parameter whose entire body is
`artist.set_<thing>(value)`" — appears to keep it out, because `AxesImage.set_extent`
exists and updates the axes limits as well as the image, so a caller can already reach
it through the returned `ImagePlot.im`.

The test governs. The prohibition sits inside "The three better homes", a subsection
about **style**, introduced by "the temptation is one argument per matplotlib property".
`extent` is not style. It is the same class of argument as `xlabel`, which `plot_image`
already takes and which §2 defends on the grounds that "a label states what the numbers
*are*, so a wrong one is a misread rather than an ugly figure". A wrong `extent` is a
misread of exactly that kind, and a signature carrying `xlabel="X (µm)"` while the
micrometres themselves arrive through a different door is not coherent.

## Decision

1. `extent` and `origin` are named, documented parameters on `plot_image`.
2. `origin` keeps matplotlib's spelling and its `{"upper", "lower"}` values.
3. `plot_image`'s docstring states the mirroring trap, naming `RamanMap` as the case
   that meets it.

## Rejected

**One `**imshow_kwargs` passthrough.** The smallest signature, and it would have carried
`interpolation` and `aspect` for free. Rejected because §2 scopes a passthrough to
"where a single artist dominates", and `plot_image` draws three things — the image, the
laser circle, the colorbar — which is why `ImagePlot` has five members. A passthrough
also has no docstring entry, so the mirroring trap would have had nowhere to be written
down; and `clim` and `cmap` already own `vmin`/`vmax` and the colormap, so a caller
passing one of those through would get a duplicate-argument `TypeError` raised from
inside matplotlib rather than from here.

**`extent` set afterwards on the returned artist.** This works — `AxesImage.set_extent`
updates the axes limits too, measured, not assumed. Rejected because it splits one
concept across two doors. `origin` has no setter at all: there is no `set_origin`, and
`origin` is not a property, so it has to be an argument whatever happens to `extent`. A
caller who passes `origin` at the call and sets `extent` afterwards has two chances to
leave the pair inconsistent, and the failure mode is a mirrored map with correct-looking
labels — the one outcome this record is trying to prevent.

**Renaming to `row_order={"top", "bottom"}`.** Avoids the name collision below, and
describes what actually moves better than "upper"/"lower" does. Rejected because it
diverges from `imshow`, making it a name nobody reaching for this behaviour would guess.

## Consequences

- **`plotting.py` now spells `origin` two unrelated ways.** `plot_diffusion_cloud` and
  `diffusion.py` use `origin` for `{"corner", "center", "image_center"}` — where a
  cloud's coordinate zero sits. `plot_image` uses matplotlib's `{"upper", "lower"}`.
  **Do not unify them.** Each docstring states its own permitted values, and matching
  `imshow` was judged worth this cost.
- A caller who passes `extent` without considering `origin` gets a mirrored map. The
  docstring says so, and there is no detection: the array cannot say which of its ends
  is which.
- The two parameters are the last that earn a place on `plot_image`'s coordinate side.
  Anything further about how the image *looks* goes through the returned `ImagePlot`.

## Load-bearing choices

Keeping matplotlib's `origin` rather than a name unique inside this package. If
`plotting.py` ever acquires a third meaning for `origin`, that trade should be
re-examined for all of them together, rather than this one drifting to a third spelling.
