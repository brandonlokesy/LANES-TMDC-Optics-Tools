# `plot_power_series` — sweep thinning and spectrum stacking

Implementation record for a change made 2026-08-05 to `plot_power_series` in
`src/tmdc_optics_tools/plotting.py`. Nothing else in the package was touched.

Two new parameters, one changed default:

| | |
|---|---|
| `sweep_step : int = 1` | Draw every *n*-th sweep instead of all of them. |
| `spectrum_offset : float = 0.0` | Stack the drawn spectra by a cumulative vertical shift. |
| `ylabel : str = None` | Was the literal `"PL intensity (counts)"`. Now derived from the scan. |

Line references are to `src/tmdc_optics_tools/plotting.py` as of this change.

---

## The request, stated precisely

A power series with many sweeps draws many overlapping lines, and overlapping
lines hide each other. Two independent remedies were wanted:

1. **Thin the set** — plot sweeps `0, n, 2n, …` so fewer lines compete.
2. **Separate them vertically** — add `0` to the first drawn spectrum,
   `spectrum_offset` to the second, `2·spectrum_offset` to the third, and so on,
   turning the overplot into a waterfall.

Both are display-only: they change which lines appear and where they sit on the
canvas, never the arrays behind them.

The third consequence was spotted in the request itself and is the reason the
change is not purely additive: **once spectra are stacked, "counts" is a false
y-axis label.** The tick values no longer read as signal.

---

# Part 1 — The three decisions

## 1.1 `skip_idx` → `sweep_step`

The parameter was originally proposed as `skip_idx`, with `skip_idx=1` meaning
*no skip*. That semantics is exactly a NumPy slice step — `data[:, ::n]` — but the
name says something else: `skip_idx` reads as *"the index of a sweep to skip"*,
i.e. one sweep dropped, which is close to the opposite of the behaviour. Under
that name the docstring has to spend two sentences arguing against the name.

`sweep_step` names the mechanism, inherits the meaning of `1` from a convention
every NumPy reader already has, and matches the existing `sweep_index` in this
module.

## 1.2 `spectrum_offset` is absolute, in the units of the plotted array

The alternative considered was a *fraction* — `0.2` meaning 20 % of the plotted
data's full span — which is easier to pick blind because it scales itself to
whatever the counts happen to be.

It was rejected because that self-scaling reads off the plotted subset. The
divisor would then depend on `x_range`, on `bg_region`, and on `sweep_step`, so
the **same scan cropped differently would stack differently**. A figure whose
spacing silently tracks the crop is one you cannot reproduce from the call alone.

Absolute is shape-invariant: the same value gives the same figure regardless of
how the data was cropped or thinned. The cost is real and is stated in the
docstring rather than engineered away — you must read a peak height off an
unstacked plot before you can choose a sensible number.

## 1.3 `ylabel` default becomes `None` → derived

Three options were on the table:

| Option | Verdict |
|---|---|
| Keep the literal `"PL intensity (counts)"` | **Rejected.** With a stack applied, the default prints a unit that is wrong, with no signal to the reader. |
| `None` → choose between two hardcoded PL strings | Solves the unit; leaves reflectance scans labelled as PL. |
| `None` → derive from `scan.signal_label` | **Chosen.** |

The literal default was already one of the hardcoded "PL intensity" sites that
finding **E12** enumerates — a reflectance sweep through this function came out
labelled as photoluminescence. Since the stacking work had to touch that exact
line anyway, deriving the label fixes that instance in passing rather than
leaving a known-wrong string behind for the E12 pass to find again.

`ylabel=None` meaning *derive it* is not a new convention here: `plot_centroid`
already does the same at `plotting.py:1711-1719`. There is no shared helper to
reuse — each function derives its own labels locally — so this follows the
pattern without adding an abstraction.

---

# Part 2 — The mechanism: why the loop has two counters

This is the only part of the change with any subtlety, and it is worth reading
carefully because both plausible one-counter versions are wrong.

**Before:**

```python
for i, p in enumerate(power):
    colour = sm.to_rgba(p)
    a = ... power_norm_linear[i] ...
    y = data[:, i]
```

**After** (`plotting.py:2338-2347`):

```python
# Two counters, and they differ once sweep_step > 1: i indexes the scan, so
# colour and alpha keep tracking each line's own power, while j counts drawn
# lines, so the offsets stack contiguously instead of leaving gaps where a
# skipped sweep would have been.
for j, i in enumerate(range(0, len(power), sweep_step)):
    colour = sm.to_rgba(power[i])
    a = float(alpha_min + (1.0 - alpha_min) * power_norm_linear[i]) \
        if alpha_by_power else float(alpha)
    y = data[:, i] + j * spectrum_offset
```

`range(0, len(power), sweep_step)` yields the **scan** indices that survive
thinning; `enumerate` numbers them in **draw** order. The two indices answer two
different questions, and each is used for exactly one thing:

| Counter | Means | Used for |
|---|---|---|
| `i` | which sweep in the scan this is | the data column, the colour, the alpha |
| `j` | how many lines have been drawn so far | the offset multiplier |

### Worked example

Six sweeps at `power = [1.0, 2.8, 4.6, 6.4, 8.2, 10.0]` µW, with
`sweep_step=2` and `spectrum_offset=500`. `range(0, 6, 2)` gives `0, 2, 4`:

| `j` | `i` | power used for colour | y drawn |
|---|---|---|---|
| 0 | 0 | 1.0 µW | `data[:, 0] + 0` |
| 1 | 2 | 4.6 µW | `data[:, 2] + 500` |
| 2 | 4 | 8.2 µW | `data[:, 4] + 1000` |

### Why one counter alone fails

**Using `i` for the offset** (`data[:, i] + i * spectrum_offset`) would shift the
three lines by 0, 1000, 2000 — leaving a blank 500-count gap wherever a skipped
sweep would have gone. The stack would grow sparser the harder you thinned it,
which is the opposite of the intent: thinning is meant to *declutter*.

**Using `j` for the colour** (`sm.to_rgba(power[j])`) is the damaging one. It
would paint the spectra actually taken at 1.0, 4.6 and 8.2 µW in the colours of
1.0, 2.8 and 4.6 µW. Every line would look plausible, the colorbar would look
correct, and **two of the three powers read off the figure would be wrong** —
a silently mislabelled figure, not a visibly broken one. This is the failure the
two-counter split exists to prevent, and the loop comment says so.

---

# Part 3 — The derived y-axis label

At `plotting.py:2364-2373`:

```python
if ylabel is None:
    if spectrum_offset:
        # Stacking destroys the absolute scale, so the unit goes.  A
        # dimensionless signal has none to drop -- signal_label then equals
        # signal_name -- and is only marked as shifted.
        has_unit = scan.signal_label != scan.signal_name
        ylabel = (f"{scan.signal_name} (a.u., offset)" if has_unit
                  else f"{scan.signal_name} (offset)")
    else:
        ylabel = scan.signal_label
```

Two branches, for two reasons.

## 3.1 Why not simply swap "counts" for "a.u."

Because half the entries in `constants.SIGNAL_LABELS` have no unit to swap:

```python
"PL":   ("PL intensity",     "counts"),
"RC":   (r"$\Delta R/R_0$",  ""),        # dimensionless ratio
"A":    ("Absorbance",       ""),        # dimensionless
```

`signal_label` composes `"name (unit)"` when a unit exists and returns the bare
name when it does not. So a string replacement of `"counts"` would do nothing at
all to `$\Delta R/R_0$`, and appending `"(a.u.)"` to it would be worse than
nothing: it would **invent a unit for a quantity that never had one**. ΔR/R₀ is a
ratio; a shifted ratio is not "in arbitrary units", it is a ratio plus a
constant. So the dimensionless branch says only `(offset)`.

## 3.2 How the branch is detected

`has_unit = scan.signal_label != scan.signal_name` uses only the two public
properties. `signal_name` is the bare name, `signal_label` is the name plus a
parenthesised unit when there is one — so they are equal exactly when the unit
is empty.

The alternative was to import `SIGNAL_LABELS` into `plotting` and unpack
`(name, unit)` directly. That was avoided because it would put the
label-composition rule in two modules: `loaders` already owns how a name and a
unit become a label, and `plotting` reading the tuple would be a second
implementation of it, free to drift. Comparing the two public properties consumes
that rule instead of restating it.

Resulting labels, all four verified by running:

| `spectra_type` | `spectrum_offset = 0` | `spectrum_offset ≠ 0` |
|---|---|---|
| `PL` | `PL intensity (counts)` | `PL intensity (a.u., offset)` |
| `RC` | `$\Delta R/R_0$` | `$\Delta R/R_0$ (offset)` |

## 3.3 Why "offset" is in the label at all, not just "a.u."

"a.u." says the numbers are not absolute. It does not say *why*, and a reader who
does not know the figure is a waterfall may try to compare two lines' heights.
`(a.u., offset)` states the transformation, so the axis is self-describing:
spacing is still meaningful, position is not.

An explicit `ylabel=` still wins over both branches — the derivation only fills a
`None`.

---

# Part 4 — Validation

At `plotting.py:2295-2299`, before any figure is created:

```python
if not isinstance(sweep_step, (int, np.integer)) or sweep_step < 1:
    raise ValueError(
        f"sweep_step must be a positive integer, got {sweep_step!r}.  "
        f"Use 1 to plot every sweep, 2 for every other one, and so on."
    )
```

It rejects rather than coerces because every invalid value fails *interestingly*
if passed through to a slice:

- `0` → `range(0, n, 0)` raises `ValueError: range() arg 3 must not be zero`, from
  inside the loop, after a figure exists.
- Negative → `range(0, n, -1)` is empty, so an **empty plot with a colorbar** and
  no error at all.
- `1.5` → `TypeError` from `range`, naming `range` rather than the parameter.

The message states the rule and then gives the two values a caller actually
wants, because `1` meaning "everything" is the one thing about this parameter
that is not self-evident.

`spectrum_offset` is unvalidated: any float is meaningful, `0.0` is off, and
negatives stack downward, which is a legitimate house style for waterfalls.

**Known gap:** `sweep_step=True` does *not* raise, because `bool` subclasses
`int` and `True >= 1`, so it silently means `1`. Harmless in effect but the
plausible misuse — the original `skip_idx` name suggested a flag, and someone who
remembers it may reach for `True`. Left unfixed pending a call: a bool-check in a
numeric parameter is unusual enough to be worth asking about rather than adding
quietly.

---

# Part 5 — What deliberately did not change

Each of these was considered and left alone. They are recorded because each looks
like an oversight.

**The colorbar spans the whole scan, not the drawn subset.** `power.min()` /
`power.max()` still run over every sweep. Thinning is a display choice, so tying
the colour scale to it would make one scan render two different power→colour
mappings depending on `sweep_step`, and two figures from the same data would not
be comparable. The docstring states this.

**`peak_marker` needs no adjustment.** It computes `x[np.argmax(y)]` on the
already-offset `y`, and `argmax` is invariant under adding a constant, so the
position is identical to the unshifted spectrum's. A comment says so at the call
site, because the obvious "bug fix" here — recomputing from the unshifted array —
is a no-op that implies the offset was a hazard.

**The arrays are untouched.** The offset is added to a temporary inside the loop;
`scan.spectra` and friends are never written. Verified by array comparison.

**Order of operations is unchanged.** `bg_region` and `x_range` still run before
the loop, on the full array. Stacking is the last thing that happens to a y
value, after every correction, which is what makes it purely cosmetic.

**No parameters added for styling.** The tick values are meaningless under a
stack and hiding them is reasonable, but `ax.set_yticks([])` is one line at the
call site on an `ax` this function already returns — so it is a docstring note,
not a `hide_yticks` parameter.

---

# Part 6 — Verification

`python -m pytest -q`: **224 passed, 1 failed.** The failure is pre-existing and
unrelated — confirmed by stashing the change and re-running the single test,
which fails identically on a clean tree. See §7.3.

`python -m mkdocs build --strict`: **green.** The new docstring text renders; the
`Raises` section was added to a docstring whose parameter block is split across
the function's own pseudo-headings, and mkdocstrings accepts it.

Behaviour was checked with a throwaway script driving a minimal stand-in object
exposing only what the function reads (`power`, `energy`, `wavelength`, the
spectra arrays, `signal_name`, `signal_label`). What it confirmed:

| Check | Result |
|---|---|
| Label, all 4 `type` × `offset` combinations | as tabulated in §3.2 |
| `lines[j]` y data equals `data[:, ::2][:, j] + j·offset` | max residual 2.3e-13 (float addition) |
| Line count under `sweep_step=2` on 6 sweeps | 3 |
| Colorbar range under `sweep_step=4` | still 1.0 → 10.0 µW |
| `alpha_by_power` under `sweep_step=4` | `[0.2, 0.84]` — the alphas of 1.0 and 8.2 µW, i.e. tracking `i`, not `j` |
| `scan.spectra` after a stacked plot | bit-identical to before |
| `sweep_step` ∈ `{0, -1, 1.5, "2"}` | `ValueError` each |
| `sweep_step=True` | no raise — the §4 gap |
| Explicit `ylabel="custom"` with an offset | `"custom"` |

The alpha row is the one that actually discriminates the two-counter logic from
the mislabelling failure in §2: under a one-counter version it would read
`[0.2, 0.36]`.

**Tests are owed.** That script was scratch and is not committed, so none of the
above is guarded against regression. The two worth having are the `alpha_by_power`
/ colour-tracks-`i` check and the four label combinations, since those encode the
decisions rather than the arithmetic.

---

# Part 7 — Follow-ups found along the way

Reported, not fixed.

## 7.1 `sweep_step=True` is accepted

See §4. Needs a yes/no on rejecting bools.

## 7.2 `_AttoCubeSweep` defines `signal_label` twice

`loaders.py:1332` is a stub whose body is a bare `return`, so it evaluates to
`None`. `loaders.py:1637` is the real implementation. Both are on the same class,
so the second silently wins and the stub is dead.

It matters slightly more now: `ylabel=None` here depends on `signal_label`
returning a string. If the definition order were ever swapped — by a merge, or by
someone tidying the file — every derived label in this function would become
`"None"` with no error raised. The fix is to delete `loaders.py:1331-1336`.

## 7.3 `SIGNAL_LABELS["R"]` disagrees with its test

`tests/test_contrast.py:251` asserts `signal_label == "Reflectance (counts)"`;
`constants.SIGNAL_LABELS["R"]` is `("Reflected intensity", "counts")`. This is
the pre-existing failure in §6.

It is now load-bearing for this function: `ylabel=None` on an `R` scan renders
whichever string wins. And it is not merely cosmetic —
`dev/plan-E12.md` (Step 1) argues the physics: reflectance is *defined* as a
dimensionless ratio R = I_r/I_i, while the loader reads R as raw CCD counts
through the same spectral layout as PL. So `"Reflectance (counts)"` asserts a
normalisation that has not happened, which is why the constant reads
`"Reflected intensity"`. On that argument the **test** is the stale side, but it
is a naming call on E12's territory, so it stays open rather than being settled
here.

---

# Part 8 — Rejected alternatives

Recorded so they are not re-proposed.

**Fractional offset.** §1.2. Rejected for depending on the crop and the thinning.

**Per-spectrum normalisation before stacking.** A true waterfall of a power
series usually wants each spectrum scaled to its own maximum, because the
intensity spans decades and one absolute offset cannot suit both ends of the
sweep. That is a genuine need and this change does not meet it — but
normalisation *changes the numbers*, so under *corrections are opt-in* it is a
separate parameter with its own default of off, not something to fold into a
display shift. It was not requested and was not added.

**Returning the drawn indices.** *Return the evidence* would suggest handing back
which sweeps survived. Not added: it is exactly `range(0, len(scan.power), sweep_step)`,
derivable from arguments the caller already holds, and the docstring states the
`lines[j]` → `scan.power[::sweep_step][j]` correspondence instead. A fourth return
value would break every existing call site to restate an argument.

**`hide_yticks` parameter.** §5.
