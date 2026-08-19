# 0024 — A plotting return longer than `(fig, ax, artist)` is a named tuple

| | |
|---|---|
| **Status** | Accepted · extended by [0025](0025-plot-image-carries-its-colorbar.md) |
| **Date** | 2026-08-19 |

## Context

`plotting` returns handles because the return contract is the styling API: there is no
`color=` parameter because `line.set_color("k")` is one line at the call site. Applied
honestly, that rule makes returns grow. Four functions had outgrown the
`(fig, ax, <artist>)` shape the module was written around:

| Function | Return |
|---|---|
| `plot_spectrum` | `fig, ax, line, ax_twin` |
| `plot_current` | `fig, ax_left, ax_right, lines` |
| `plot_image` | `fig, ax, im, circle` |
| `plot_spectral_series` | `fig, ax, cb, lines, ax_twin` |

Two of those members are usually nothing. `twin_axis` and `laser_annotation` are both off
by default — corrections and annotations are opt-in — so `ax_twin` and `circle` are `None`
in the ordinary call. `tests/test_plotting_labels.py` had five lines reading
`_, ax, _, _, _ = plotting.plot_spectral_series(...)`: four discarded positions to reach
the second one.

The complaint that forced the decision was that returning something usually unused reads as
a design mistake, and that a reader has to count to five to find out what `ax_twin` was.
Both halves are fair, but they are different complaints, and only the second one is about
the return *shape*:

- **The `None` is not the problem.** [0021](0021-the-conjugate-axis-has-one-implementation.md)
  rejected a return whose length depends on an argument's value, because a caller then
  cannot unpack the result without knowing what it asked for, and every helper downstream
  needs the same branch. That argument has not weakened. A fixed length with a `None` in it
  is the alternative, not the defect.
- **The anonymity is the problem.** Position four of five carries no information. Nothing
  about `fig, ax, cb, lines, ax_twin` tells a reader that `cb` precedes `lines`, and nothing
  catches them swapping.

Where the wider ecosystem has settled, checked rather than recalled:

- Matplotlib returns a **fixed-field container** once a plot draws several artists.
  `ax.bar` gives a `BarContainer` carrying `.patches`, `.errorbar`, `.datavalues` — always
  those three, with `.errorbar` `None` when no error bars were drawn. `ax.errorbar` and
  `ax.stem` do the same; `ax.boxplot` returns a dict whose `"means"` key is always present
  and empty unless means were asked for. Its older, simpler functions still return plain
  tuples (`ax.hist` gives `(n, bins, patches)`), so matplotlib migrated toward containers as
  returns grew and never went back.
- SciPy's best-behaved result is `scipy.stats.linregress`, which returns a `LinregressResult`
  that **is** a tuple and also carries
  `_fields = ('slope', 'intercept', 'rvalue', 'pvalue', 'stderr')`.
- SciPy's `OptimizeResult` is the counter-example, and it is the exact failure this record
  is careful to avoid: its fields genuinely differ by solver — BFGS supplies `hess_inv`,
  `jac`, `njev`, Nelder-Mead supplies `final_simplex` instead — so a caller cannot know the
  shape of what it is holding.

NumPy is no guide here. Where it has optional outputs it changes the number of return values
from an argument (`np.unique(x, return_counts=True)`), which is what 0021 already rejected.

## Decision

1. **A plotting return longer than `(fig, ax, <artist>)` is a `NamedTuple`**, named
   `<Thing>Plot`: `SpectrumPlot`, `CurrentPlot`, `ImagePlot`, `SpectralSeriesPlot`. The nine
   functions that return `fig, ax` or `fig, ax, artist` are unambiguous and unchanged.
2. **The length stays fixed and an absent artist is a `None` member.** 0021 part 3 is
   untouched and needs no amendment: the conjugate axis is still returned, at fixed arity,
   `None` when `twin_axis=False`. What changes is that the member now has a name.
3. **Field order is part of the contract.** Every existing caller unpacks positionally, so
   the order is what those callers depend on and `tests/test_plotting_return_shapes.py`
   pins it — both the `_fields` tuple and, separately, the *kind of object* each member
   holds. The second check is the one that matters: `_fields` being correct does not prove
   the return statement passes its arguments in the matching order, and a swap there would
   be silent.
4. **The per-member contract lives on the class.** Each function's `Returns` block names its
   class in one line; the units, the axis each artist belongs to, and the condition under
   which a member is `None` are stated once, in the class `Attributes` block.

## Rejected

**Keeping the plain tuple and adopting `fig, ax, *_ =` at call sites.** Free, breaks
nothing, ordinary Python, and it does remove the row of underscores. Rejected because it
only fixes the *writing* of a call. The documented signature still presents five anonymous
positions, so a reader who wants the conjugate axis still counts, and a member added later
still has no name. It is a habit, not a contract — and habits are not what a shared library
hands to fifteen people.

**A mutable dataclass**, matching `fitting.FitResult` and matplotlib's containers directly.
Rejected because it is not a tuple, so roughly forty positional unpackings across the suite,
the README and two example notebooks break at once — for a gain the `NamedTuple` already
delivers. Worth revisiting only if a fifth or sixth member is ever wanted, since a
`NamedTuple` still lengthens when extended and that is the one thing a dataclass fixes.

**An accessor on the axes**, `conjugate_axis(ax)`, modelled on `ax.get_legend()`. Matplotlib
does keep the link: because `_conjugate_x_axis` builds the axis with `secondary_xaxis`, the
parent holds it in `ax.child_axes` — verified, and `tests/test_x_axis_vocabulary.py` already
asserts `ax.child_axes == []` to prove absence. So 0021's "walk `fig.axes` and guess"
objection would have been met. Rejected on scope, not on mechanism: it introduces a third
return convention into a module that already has two, and it fixes only the conjugate axis
while `plot_image`'s `circle` and `plot_current`'s `ax_right` stay anonymous. It also needs
a private label on the secondary axis so the accessor can tell it from a future inset or
secondary y-axis on the same host — machinery in service of one member.

**Monkey-patching `ax.conjugate_axis`.** The most direct reading of "reach it through `ax`".
Rejected because matplotlib never asks a caller to read an attribute the library did not
put there, nothing in this package sets an attribute on a matplotlib object, and an
attribute invented by us is invisible to `dir(ax)` readers, to type checkers, and to anyone
reading the matplotlib documentation.

**One shared class for every plotting return.** The four field sets differ, so a single
class needs optional members, and which of them are populated would then depend on which
function you called — `OptimizeResult`'s failure, rebuilt.

**Variable-length returns.** A 4-tuple with `twin_axis=True` and a 3-tuple without. Already
rejected by 0021; restated here because the ecosystem evidence is now on the record against
it, in `OptimizeResult` and in `np.unique(return_counts=True)`.

## Consequences

- Nothing breaks. A `NamedTuple` is a tuple, so all existing unpackings, the README
  snippets and the example notebook cells keep working. The suite passed with no edit to any
  of its roughly forty positional unpackings, and that is the regression check.
- `res.ax_twin`, `res.circle`, `res.cb` are reachable by name, and `fig, ax, *_ = res`
  discards the rest without naming it.
- The shape no longer varies with what was asked for, which was the objection: five members
  with `twin_axis=True` and five without.
- `plot_current` acquired its first tests, since the return-shape suite needed a scan with
  the wiring declared in order to check its members.
- Two stale unpackings that described these exact returns were corrected in the same change:
  the `Examples` block of `plot_spectral_series`, which unpacked four of five, and the
  `plot_current` snippet in the README, which unpacked three of four.

## Load-bearing choices

Field order is the whole risk. Because callers unpack positionally, a reordered field hands
them the wrong objects without an error anywhere — the names would still read correctly. The
type-of-member assertions in `tests/test_plotting_return_shapes.py` are what stand between
that mistake and a silent wrong plot, so they must be kept aligned with the classes rather
than trimmed as redundant.

`plot_image` builds a colorbar and does not return it. That is untouched here and remains a
gap in the same contract this record is about.
