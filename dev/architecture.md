# Architecture and vocabulary

**Audience:** someone who has just been handed this package and has to change
something in it. It assumes you know Python and NumPy and you know the physics; it
assumes you know nothing about *this* codebase's conventions.

**What this file is for.** Most of the confusion here is vocabulary, not logic. The
code says *role*, *curated*, *payload*, *source*, *block*, *declared* — each with a
narrow meaning that is never spelled out in one place because every docstring
reasonably assumes you already know. This file is that one place.

**Where the other dev files fit.** Nothing here duplicates them; follow the pointer
when you need the argument rather than the mechanism.

| File | Holds |
|---|---|
| `.claude/CLAUDE.md` | standing conventions, physics record, design principles, settled decisions |
| `dev/audit-2026-07.md` | the defect register: what is broken, what was fixed and when |
| `dev/E7b-E7c-gates.md` | the gate machinery line by line — the deep dive this file summarises |
| `dev/plan-E12.md`, `dev/cosmic_ray_fix.md`, … | one change each |
| **this file** | how the pieces fit and what the words mean |

**On line numbers.** This file names functions and attributes rather than line
numbers, because names survive edits. `dev/E7b-E7c-gates.md` uses line numbers
deliberately (it is pinned to one dated change); this one is meant to stay true.

---

# 1. The shape of the package

Eight modules, ~9.5k lines, one dependency direction. Nothing below imports
anything above it.

```
        constants.py         physical constants, material tables,
             │               controlled vocabularies
             ▼
        loaders.py           I/O + device geometry.  Turns files into objects.
        ├── hdf5.py          one archive format, both axis kinds
        └── processing.py    pure array functions — no objects, no files
             │
             ├──► fitting.py     arrays → dataclasses (FitResult, DipoleResult)
             ├──► diffusion.py   image arrays → DiffusionResult
             └──► plotting.py    objects/arrays → (fig, ax, artist)
```

The contracts, in one line each:

- **`constants`** — data, no logic. If a number is physical, it lives here.
- **`loaders`** — the only module that reads files, and the only one that knows what
  a device is. Everything it produces is a plain NumPy array plus metadata.
- **`processing`** — functions from arrays to arrays. No knowledge of scans, files,
  or matplotlib. This is where maths goes so that plotting can't reimplement it.
- **`hdf5`** — serialise a loader object and read it back. Imported *lazily* inside
  `loaders._decode` so `h5py` is only needed if you actually touch an `.h5`.
- **`fitting`** — curve fits, returning dataclasses that carry parameters,
  uncertainties, and diagnostics.
- **`plotting`** — draws. Returns `(fig, ax, <artist>)`, never calls `plt.show()`.
- **`diffusion`** — real-space exciton cloud analysis (segment, centroid, area).

Two rules that come up constantly and are easy to break by accident:

1. **`plotting` must not do maths that belongs in `processing`.** If a plot needs a
   number computed, the computation goes in `processing` and the plot calls it.
2. **Raw arrays are never mutated after load.** A correction produces a *new*
   attribute; it never overwrites the array the file held. See §7.

---

# 2. Glossary

Read this section first. Everything after it uses these words precisely.

## 2.1 The file format words

### Block

The AttoCube exporter writes **one column block per sweep point**. A block is a
fixed group of consecutive columns carrying everything about that one point:

```
Parameters Labels │ Par_0 Wavelength0 ExpROI1_0 ExpROI2_0 │ Par_1 Wavelength1 … │ …
   ↑ label column │ ←──────── block 0 (one sweep point) ─→ │ ←── block 1 ───────→ │
```

`block_width` is 4 here. The block's field names — with the trailing index stripped
— are what identify the **layout**.

### Layout kind — `"spectral"` vs `"temporal"`

Two block shapes exist, and the *field names* tell them apart. This is the table
`_BLOCK_LAYOUTS`:

| Kind | Block fields | Written by |
|---|---|---|
| `spectral` | `Par`, `Wavelength`, `ExpROI1`, `ExpROI2` | PL, R, RC |
| `temporal` | `Par`, `Wavelength`, `Exp` | TRPL |

**In the temporal layout the column named `Wavelength` holds time.** That is an
acquisition-software misnomer. Read it as time; do not "fix" the name.

Each loader class declares which layout it accepts as `_LAYOUT_KIND`, and refuses
the other by name in the error.

### Declared vs written blocks

The exporter **over-allocates**: it declares more blocks in the header than it
actually fills, and the surplus are filled with literal `0.0` in every field — not
blank, not NaN. So no NaN-strip removes them, and keeping them would fabricate
measurements that were never taken.

`_drop_unwritten_blocks` finds them using **the axis column being identically zero**
as the sentinel (a spectrometer axis never contains zeros; a time axis has only its
first bin at zero, never the whole column). Hence two counts on every scan:

- `n_declared_sweeps` — what the header claimed.
- `n_sweeps` — what was really written, and the width of every array.

`__repr__` prints the difference when they disagree.

There is a **second, separate** over-allocation: the rows are also padded to twice
the named width with *empty* fields. Nothing needs to strip that — `_read_block_layout`
counts columns matching `^Par_?\d+$`, and an empty field cannot match.

### Payload

**The dict every decoder returns.** It is the seam between "how this format is
stored" and "everything else", and it is why HDF5 input needs no second loader
class: `_decode_csv` and `hdf5.read_sweep` produce the same shape, and the code
after that point cannot tell which one ran.

```python
{
    "wavelength" : (n_points,)              # or "time" for TRPL
    "roi1"       : (n_points, n_sweeps)     # or "counts" for TRPL
    "roi2"       : (n_points, n_sweeps),
    "parameters" : {label: (n_sweeps,) array, …},
    "metadata"   : {…},        # empty {} for a raw CSV, populated for HDF5
    "n_declared" : int,
}
```

If you add an input format, you write a decoder that returns this. Nothing else
changes.

## 2.2 The measurement words

### Sweep point, `n_sweeps`, `n_points`

A **sweep point** is one step of whatever was scanned — one spectrum, plus a full
snapshot of every instrument channel at that moment.

Two counts, and mixing them up is the most common shape bug:

| | Meaning | Also called |
|---|---|---|
| `n_points` | length of the **measured axis** | `n_pixels` (spectral), `n_bins` (TRPL) |
| `n_sweeps` | number of **sweep points** | — |

**Every signal array in the package is `(n_points, n_sweeps)`** — measured axis down
the rows, sweep across the columns. Every parameter array is `(n_sweeps,)`. Nothing
is transposed anywhere; if you find yourself wanting `.T`, check first.

### Parameter, row, label

The export carries **57 labelled scalar rows** — every instrument channel, one value
per sweep point. `scan.parameters` is those rows:

```python
scan.parameters                 # {"V_A": (n_sweeps,), "Scanner X": (n_sweeps,), …}
scan.parameters["V_A"]          # raw file units
scan.get_parameter("V_A")       # same, with an error message if absent
scan["V_A"]                     # sugar for get_parameter
scan.parameter_labels           # the list of available names
```

**"Label" always means the string as it appears in the file** — `"V_A"`,
`"Scanner X"`, `"Excitation Power"`. It is the instrument's word, not ours.

`varying_parameters()` reports which rows actually moved, ranked by span relative to
their own **RMS** magnitude — RMS rather than mean, so a row straddling zero is
measured by how large it is rather than by how nearly it cancels (A10). It is
**evidence, not a detector**: a small channel swinging across its whole range
outranks a large one stepped through part of its, so the top entry is often a
leakage current. Which axis was swept is the caller's to declare.

### Curated parameter / curated attribute / the curated registry

A handful of rows are **analysis-primary**, so they get promoted to first-class
properties with a unit conversion attached. That promotion is the **curated
registry**.

The class-level table `_AttoCubeSweep._CURATED` maps

```
curated attribute  ->  (default row label, scale, unit)
```

```python
_CURATED = {
    "v_top":     ("V_A",              1.0,      "V"),
    "v_bot":     ("V_B",              1.0,      "V"),
    "power":     ("Excitation Power", 0.303e6,  "µW"),
    "i_top":     (None,               1e9,      "nA"),
    "i_bot":     (None,               1e9,      "nA"),
    "i_channel": (None,               1e9,      "nA"),
    "scanner_x": ("Scanner X",        1.0,      "V"),
    "scanner_y": ("Scanner Y",        1.0,      "V"),
}
```

So a **curated attribute** is the Python-side name (`power`), and the row label is
the file-side name (`"Excitation Power"`).

A `None` label means the row is not a fixed property of the format but comes from
the session's `gates=` declaration. The two voltages carry a default anyway, because
`_gate_candidates()` and `gate_mode` both describe an *undeclared* scan and need
somewhere to look; the three currents have no such reader, so a default there would
be a guess nothing consults. Their labels are filled in from `gates` through
`_CHANNEL_SIBLING_CURRENT`, which records that a source-meter channel's bias row and
current row are one terminal — `{"bottom": "V_A"}` therefore also makes `I_A`
reachable as `i_bot`. A gate declared on some other row keeps its voltage and has no
current.

Reading a curated entry is:

```python
def _curated_value(self, name):
    label, scale, _ = self._curated[name]
    return self.get_parameter(label, scale)      # parameters[label] * scale
```

Four things to know:

- **It is a lookup, not a copy.** Nothing is stored; `scan.power` reads
  `parameters["Excitation Power"]` and multiplies, every time you access it. Mutating
  the returned array changes nothing.
- **A curated row a file lacks is not an error.** The property raises only if you
  access it. A file from a different instrument configuration still loads.
- **`self._CURATED` is the class default; `self._curated` is this instance's
  resolved copy** — same keys, but labels and scales may have been overridden. It is
  built as lists so it can be mutated during construction, then frozen to tuples so
  it cannot be after.
- **Everything else in the file is still reachable** through `parameters`. Curation
  buys you a name and a unit, nothing more.

Overriding, per instance:

```python
scan = AttoCubeSpectralSweep(..., curated_labels={"scanner_x": "Galvo_X"})  # which row
scan = AttoCubeSpectralSweep(..., curated_scales={"scanner_x": 12.5})       # µm per V
```

`v_top` and `v_bot` are the exception — see roles, next.

### Role, electrode, channel

This is the vocabulary that trips people up most, so here is the whole chain in one
picture. Four different names for what is arguably "the same voltage":

```
  "V_A"                acquisition channel      what the instrument wrote in the file
    │                                           ← the only one the file knows
    │  gates={"bottom": "V_A"}                  ← you declare this, per session
    ▼
  "bottom"             role                     where it sits in the device
    │
    │  _GATE_ROLE_CURATED                       ← fixed table, role → curated attr
    ▼
  "v_bot"              curated attribute        the Python property, in volts
    │
    │  _SWEEP_TYPES["bottom_voltage"]           ← fixed table, sweep type → attr
    ▼
  "bottom_voltage"     sweep type               what you pass as sweep=
```

A **role** is a *physical position in the device*: `"top"`, `"bottom"`, `"channel"`.
Three roles exist, and they split in two:

- **Gate electrodes** (`"top"`, `"bottom"`) — metal that gates the TMDC *across a
  dielectric*. A potential difference between two of them is what defines a
  displacement field, so a field needs both. Constant: `_GATE_ELECTRODES`.
- **The channel** (`"channel"`) — a contact to the **TMDC itself**. It sits *inside*
  the stack rather than across a dielectric from it, so it has no thickness and
  enters no field. It is **not a gate** and is excluded from `gate_mode`. Its job is
  to record that the device is contacted, which is what a carrier density is
  referenced to.

Together: `_GATE_ROLES = _GATE_ELECTRODES + ("channel",)`.

**Why the declaration exists at all.** Which acquisition channel reached which
electrode is set by where someone plugged a wire. No export records it. The package
used to guess (`v_top` ← `V_A`), which produced a mirrored field, a wrong-signed
dipole, and a plot that looked entirely normal. So it now **refuses**:

```python
gates={"top": "V_A", "bottom": "V_B"}      # dual-gated
gates={"bottom": "V_A", "channel": None}   # bottom-gated, TMDC hard-grounded
gates=None                                 # undeclared → v_top/v_bot/ef/... all raise
```

**The keys present describe the device**, not just the wiring. A missing `"top"` is a
statement that there is no top gate, not an omission — which is why `ef` correctly
raises on a single-gated device rather than returning a number for a quantity that
does not exist there. A value of `None` means the electrode is tied to ground with no
row recording it.

Channel-level work needs no declaration at all: `scan["V_A"]` and `sweep="V_A"` are
unaffected by any of this.

Full treatment, including every refusal and why each one is where it is:
`dev/E7b-E7c-gates.md`.

### Sweep type vs sweep source

**Sweep type** is the string you pass as `sweep=` — a key of `_SWEEP_TYPES`, or a raw
row label, or `None`.

**Sweep source** is the internal `(kind, name)` pair that `_resolve_sweep` produces
and `sweep_axis` consumes. Three kinds:

| kind | name | `sweep_axis` returns |
|---|---|---|
| `"index"` | `None` | `np.arange(n_sweeps)` |
| `"row"` | a row label | `get_parameter(label)`, file units |
| `"curated"` | a **property name** | `getattr(self, name)` |

The third is the trick worth knowing: `_SWEEP_TYPES` maps a sweep type to *the name
of a property*, and `sweep_axis` just calls `getattr`. That is why adding a sweep
type needs no special-casing anywhere — write the property, add the table row.

```python
_SWEEP_TYPES = {
    "index"          : (None,              "Sweep index",       ""),
    "electric_field" : ("ef",              r"$E_F$",            "mV/nm"),
    "carrier_density": ("carrier_density", r"$\Delta n$",       r"cm$^{-2}$"),
    "top_voltage"    : ("v_top",           r"$V_\mathrm{top}$", "V"),
    "bottom_voltage" : ("v_bot",           r"$V_\mathrm{bot}$", "V"),
    "power"          : ("power",           "Power",             "µW"),
    "piezo_x"        : ("scanner_x",       r"Piezo $x$",        "V"),
    "piezo_y"        : ("scanner_y",       r"Piezo $y$",        "V"),
}
```

**An undeclared `sweep=` means the sweep index — never an auto-detected parameter.**
That is a settled decision, not an oversight. The package will not guess what you
scanned.

Anything *not* in `_SWEEP_TYPES` is looked up as a raw row label, which is why a new
scanned quantity (`"Galvo_Y"`, `"T"`) needs no code change to be usable as an axis.

### `_SWEEP_REQUIRES`

A companion table: which curated rows each sweep type depends on, so a bad `sweep=`
fails **at load time** with the file's available labels listed, rather than at the
first plot.

```python
_SWEEP_REQUIRES = {
    "electric_field": ("v_top", "v_bot"),
    "top_voltage"   : ("v_top",),
    ...
}
```

Note what is **not** here: there is deliberately no `_GATE_SWEEPS` list. Which sweep
types need a gate declaration is *derived* by mapping `_SWEEP_REQUIRES[sweep]`
through `_ROLE_FOR_CURATED`, so a gate-backed sweep type added next year inherits the
requirement automatically instead of being forgotten in a second list.

`"carrier_density"` has no entry, also deliberately: its requirement depends on which
roles were declared, which a static table cannot express. It is checked inline
instead.

### `spectra_type`, signal, spectroscopy

`spectra_type` is what the spectra **are** — a key of `constants.SPECTROSCOPY_TYPES`
(`"PL"`, `"R"`, `"RC"`, `"T"`, `"A"`, `"TRPL"`). It is **required, keyword-only, with
no default**: the value is written into exported metadata and trusted thereafter, so
a guess would outlive the session that made it.

It drives labelling and metadata, not parsing:

```python
scan.spectra_type    # "PL"
scan.spectroscopy    # "Photoluminescence"        SPECTROSCOPY_TYPES
scan.signal_name     # "PL intensity"             SIGNAL_LABELS[type][0]
scan.signal_label    # "PL intensity (counts)"    name + unit, unit omitted if ""
scan.contrast_label  # "ΔR/R₀"                    independent of signal_label
```

Use `scan.signal_label` in a plot rather than hardcoding `"PL intensity"` — a
reflectance scan handed to the same function should not be labelled PL. (`plotting`
currently hardcodes it in about six places; that is open work, see E12.)

### Axis kind

The HDF5-side counterpart of layout kind. The axis dataset is **named for the
physical quantity it holds** (`axes/wavelength` or `axes/time`), so `h5ls` alone says
what a file contains, and each loader rejects the other's archives by name.

```
layout kind (CSV, from header)   spectral  ←→  temporal
axis kind   (HDF5, from dataset) wavelength ←→ time
```

## 2.3 The correction words

### Correction chain

The ordered set of optional operations applied at load. Every one is **opt-in**:
the default is always "off", and off means the least-assuming option, not merely the
least code. Loading is not deciding.

The order is physics, not preference — see §7.

### Provenance / `source_metadata`

What the *writing* session chose, recorded but **never replayed**. An HDF5 file
records that it was written with `apply_jacobian=True` and a `bg_region_nm`, and
reading it back gives you those facts in `scan.source_metadata` — but the corrections
do not re-run. Re-applying a correction because a file mentions one would make
loading a decision.

The distinction that matters:

| Restored on read | Recorded only |
|---|---|
| `spectra_type`, `sweep`, `gates`, `geometry`, `curated_labels`/`scales`, `roi` | `apply_jacobian`, `bg_region_nm`, `cosmic_rays`, the aux spectra |
| *what the measurement was* | *what one session did to it* |

---

# 3. The `_resolve_*` convention

There are seven functions named `_resolve_*` across three modules, and they all mean
the same thing. Once you see the pattern the loaders get much easier to read.

> **A `_resolve_X` takes what the caller said, what the file said, and the defaults,
> and returns the one settled value of X — or raises with an actionable message. It
> is the single place a given ambiguity is decided.**

Precedence is the same everywhere: **explicit argument > file metadata > default**.

| Function | Module | Resolves | Raises when |
|---|---|---|---|
| `_resolve_spectra_type` | loaders | what the spectra are | neither argument nor file states it |
| `_resolve_gates` | loaders | the electrode mapping | unknown role; no gate; a lone gate with no channel |
| `_resolve_sweep` | loaders | the sweep axis | unknown sweep; missing row; missing role/geometry |
| `_resolve_aux_spectrum` | loaders | a background or reference onto this scan's grid | axis mismatch (never resamples) |
| `_resolve_baseline` | fitting | `"constant"`/`"linear"`/`"none"` → model terms | unrecognised key |
| `_x_axis_name_unit` | constants | `"energy"`/`"wavelength"` → `(name, unit)` | anything else |
| `_resolve_x_axis` | plotting | an axis name → `(array, label)` | delegates the refusal |
| `_resolve_spectra` | loaders | `spectra_source=` / `source=` → the actual array | unknown axis; source unavailable on this scan |

Two design habits visible in all of them:

- **They are the only place that ambiguity is decided.** The `x_axis` vocabulary is
  the worked example of why that matters, since it is the one ambiguity decided in
  three modules: `constants.X_AXES` holds the two axes and `_x_axis_name_unit` is the
  only thing that refuses a third, so `_resolve_x_axis` (plotting), `_resolve_spectra`
  and `pixel_slice` (loaders) and `fit_scan_peak` (fitting) share one message. A
  branch on `x_axis == "energy"` still picks the arrays — that mapping is not uniform
  — but it must sit *after* the refusal, or the `else` is a two-way test on a
  free-form string and every misspelling reads as wavelength (A14).
- **A disagreement is not swallowed.** `_resolve_spectra_type` warns when the
  argument and the file disagree, then uses the argument. A relabelled measurement is
  exactly the error that survives into every downstream figure.

There is also `_bind_sweep_axis`, which is *not* a resolver — it is the thin wrapper
that applies the metadata fallback and unpacks `_resolve_sweep`'s four-tuple onto
`self`. Worth its own name because *when* it is called matters (see §5).

## `_resolve_sweep` in detail

The most involved of them; it runs six checks in a fixed order, and the order is the
point — each check assumes the previous one passed.

```python
def _resolve_sweep(self, sweep, label, unit) -> (key, source, label, unit):
    if sweep is None: sweep = "index"

    if sweep in _SWEEP_TYPES:
        source, default_label, default_unit = _SWEEP_TYPES[sweep]

        # 1. Role check, FIRST: without a mapping there is no row to look for.
        for name in _SWEEP_REQUIRES.get(sweep, ()):
            role = _ROLE_FOR_CURATED.get(name)      # "v_bot" -> "bottom"
            if role is None: continue                # not a gate-backed row
            self._require_role(role, f"sweep={sweep!r}")

        # 2. Grounded check: a declared-None electrode is all zeros, not an axis.
        # 3. Row check: does this file actually contain the row that backs it?
        #    Error branches on whether the row is a gate, so it points at
        #    gates= or curated_labels — never at the door that now raises.
        # 4. electric_field needs a geometry.
        # 5. carrier_density needs a geometry, a channel, and the hBN of every
        #    declared gate.  Checked here, not in _SWEEP_REQUIRES, because the
        #    requirement depends on which roles exist.
        return (sweep, ("curated", source) if source else ("index", None), …)

    if sweep in self.parameters:
        return (sweep, ("row", sweep), label or sweep, unit or "")   # raw row

    raise ValueError(...)   # lists both the sweep types and the file's rows
```

Everything a `sweep=` needs therefore fails **during construction**, with the
available labels printed. Property access (`scan.v_top`) fails on **first access**
instead — a file that lacks a curated row is not an error until you ask for it.

---

# 4. `DeviceGeometry` — the device, separate from the data

`DeviceGeometry` knows nothing about files. It is the stack: hBN thicknesses, TMDC
layers, dielectric constants, and the four quantities you can compute from them.

```python
geom = DeviceGeometry.from_single("WSe2", d_hbn_top=53, d_hbn_bottom=46)

geom.eps_stack              # ε of the homogeneous slab with the same capacitance
geom.eps_2d                 # TMDC-only series-capacitor value
geom.d_stack                # d_2d + d_hbn_top + d_hbn_bottom
geom.electric_field(v_top, v_bot)   # mV/nm
geom.gate_capacitance("bottom")     # F/m², ε₀ε_hBN/d_hBN — that gate's hBN only
geom.carrier_density(v_bot=…)       # cm⁻², electrons positive
```

Two things about this class that a maintainer will otherwise get wrong, both recorded
in `.claude/CLAUDE.md` with the full derivation:

- **`electric_field` is exact as written.** Two plausible "simplifications" are
  already shipped elsewhere in the group and both change the numbers — one by 0.6%,
  one by 1.8×. Do not touch it.
- **`gate_capacitance` uses only that gate's hBN.** The TMDC is the
  *counter-electrode*, not a slab inside the capacitor, so neither its thickness nor
  `eps_stack` enters. Reaching for `eps_stack` here is the sign you have the wrong
  tool. A test pins that a 5-layer stack returns the same number.

`StackLayer` is the per-material slab (`material`, `n_layers`, `d_monolayer`, `eps`),
with thickness and ε looked up from `constants` when not given. Note `_slabs()`
returns bare `(d, ε)` tuples rather than `StackLayer`s — wrapping an hBN flake in one
would assert a false `n_layers`.

---

# 5. Life of a load — `AttoCubeSpectralSweep`, start to finish

```python
scan = AttoCubeSpectralSweep(
    "PL-dual-gate-sweep_iter_0.csv",
    spectra_type = "PL",
    sweep        = "electric_field",
    geometry     = geom,
    gates        = {"top": "V_A", "bottom": "V_B"},
    bg_region_nm = (700, 710),
)
```

## The base class contract

`_AttoCubeSweep` holds everything independent of *what the measured axis is*. A
subclass declares four class attributes and then drives construction itself:

```python
_LAYOUT_KIND = "spectral"   # which export layout it accepts
_AXIS_ATTR   = "wavelength" # attribute holding the (n_points,) axis
_SIGNAL_ATTR = "spectra"    # attribute holding the (n_points, n_sweeps) signal
_POINT_NOUN  = "pixels"     # what to call n_points in __repr__
```

`AttoCubeTRPLSweep` sets the same four to `"temporal"`, `"time"`, `"decays"`,
`"time bins"`. Those four strings are the *entire* difference the base class sees —
`n_sweeps` is `getattr(self, self._SIGNAL_ATTR).shape[1]`, and everything else
follows.

Construction is spelled out in each subclass rather than hidden in a template method,
**because the ordering is load-bearing and a reader should be able to see it.**

## The order, and why each step is where it is

### Step 0 — argument checks, before the read

Mutually-exclusive `bg_region_nm`/`bg_region_eV`, and unknown `cosmic_rays` keys.
Both precede the decode because an export is large enough that a typo should not cost
you the read first.

`_COSMIC_RAY_KEYS` is derived with `inspect.signature(processing.remove_cosmic_rays)`
so the accepted keys cannot drift from the function they are forwarded to.

### Step 1 — `_decode_and_describe`: everything independent of the axis

```python
self.path            = str(path)
payload              = self._decode(path)      # ← dispatch: dir / .h5 / .csv
self.source_metadata = dict(payload["metadata"])
self.parameters      = payload["parameters"]   # (a)
self._n_declared     = payload.get("n_declared")
self.spectra_type    = self._resolve_spectra_type(spectra_type, meta)
self.geometry        = geometry if geometry is not None else meta.get("geometry")
self._gates          = self._resolve_gates(gates or meta.get("gates"))   # (b)
# ... then the curated registry ...                                      # (c)
```

Three ordering constraints inside this one method:

**(a) `parameters` first.** The gate machinery needs to be able to *name the file's
actual rows* in its error messages, and that is only possible once they are loaded.

**(b) gates before the curated registry**, because the declaration supplies two of
that registry's labels.

**(c) the registry, built then frozen:**

```python
self._curated = {name: list(cfg) for name, cfg in self._CURATED.items()}  # mutable

# HDF5 dumps the resolved label of EVERY curated entry, gates included.  That is
# the writer's bookkeeping, not a wiring claim — strip it silently, or a
# CSV→HDF5→reload round trip would launder a defaulted mapping into a stated one.
meta_labels = {k: v for k, v in (meta.get("curated_labels") or {}).items()
               if k not in _GATE_CURATED}

for store, idx in ((meta_labels, 0), (curated_labels, 0),
                   (meta.get("curated_scales"), 1), (curated_scales, 1)):
    ...   # unknown name -> raise;  a gate name as a *label* key -> raise, pointing
          # at gates=.  idx==0 restricts that to labels: curated_scales still
          # reaches gate entries, since a unit conversion claims nothing about wiring.

for role, attr in _GATE_ROLE_CURATED.items():        # declared labels win last
    if (self._gates or {}).get(role) is not None:
        self._curated[attr][0] = self._gates[role]

self._curated = {name: tuple(cfg) for name, cfg in self._curated.items()}  # frozen
```

**The core of the gate design is in that last block.** When a role is undeclared, the
`_CURATED` default label *survives* — and that is fine, because nothing will read it.
The refusal lives on `self._gates`, never on the label. So a defaulted label can
never be mistaken for a stated one.

The **asymmetry** — silent strip for HDF5, raise for a caller — is deliberate: an
archive's own dump is bookkeeping the writer produced automatically, and raising on
it would make every existing archive unreadable. A caller passing
`curated_labels={"v_top": …}` is making a wiring claim through the wrong door.

### Step 2 — the arrays

```python
self.wavelength   = payload["wavelength"]      # nm, ascending
self.spectra_roi1 = payload["roi1"]            # (n_pixels, n_sweeps)
self.spectra_roi2 = payload["roi2"]
self._validate_payload()                       # shapes agree, axis is 1-D non-empty

self.spectra = self.spectra_roi1 if roi == 1 else self.spectra_roi2
```

**Both ROIs are always loaded.** `ExpROI1` is the excitation spot and `ExpROI2` a
remote, spatially-filtered spot, for two-spot galvo scans. `ExpROI2` is identically
zero in every other measurement, which is why the loader **warns when the selected
ROI is all zeros** — a flat zero array is otherwise indistinguishable from a valid
dark measurement. `roi=` only chooses which one `spectra` points at.

### Step 3 — `_bind_sweep_axis`, and why it is here

**`n_sweeps` now exists.** It is derived from the signal array
(`getattr(self, self._SIGNAL_ATTR).shape[1]`), so the sweep axis cannot be resolved
before Step 2. That is the whole reason construction is explicit rather than
templated.

### Step 4 — the correction chain

See §7. Everything from here builds *new* arrays.

## The decode dispatch

`_decode` is three lines of routing, and it is where the "one class, two formats"
promise is kept:

```python
if path.is_dir():                return self._decode_dir(path)    # TRPL only
if suffix in _HDF5_SUFFIXES:     return hdf5.read_sweep(path)     # + axis-kind check
if suffix in _CSV_SUFFIXES:      return self._decode_csv(path)
raise ValueError(...)
```

`_decode_dir` on the base class raises with a message naming the one class that reads
directories. `h5py` is imported *inside* the HDF5 branch, so it is only a dependency
if you use it.

## The CSV decode, in six lines

```python
blocks = _read_block_layout(path)              # header line ALONE — cheap, and
                                               # usable as a file classifier
raw    = pd.read_csv(path, header=0, index_col=0, low_memory=False)
d      = raw.to_numpy(float)[:, :n_declared * width]     # drop the empty pad

cols   = {role: np.arange(offset, d.shape[1], width)     # one column index per
          for offset, role in enumerate(blocks["roles"])} # sweep point, per field

keep, n_declared, axis_block = _drop_unwritten_blocks(d[:, cols["Wavelength"]], path)

parameters = {label: d[i, cols["Par"]][keep] for i, label in enumerate(row_labels)}
```

Two things worth understanding here:

- **The stride trick.** `cols[role]` is one column index per sweep point for that
  field — `arange(offset, n_cols, block_width)`. Fancy-indexing with it lifts a whole
  field out of the interleaved layout in one operation, no loop.
- **The labelled rows are overlaid on the leading pixel rows.** Row *i*'s value for
  sweep *j* sits in block *j*'s `Par` column. That is why `parameters` is built by
  enumerating the CSV index against `cols["Par"]`.
- **`axis_block` is returned separately from `keep`** because in the interleaved
  (warned, nothing-dropped) case block 0 may itself be zero-filled — taking the axis
  from it would give an all-zero wavelength axis and infinite energies.

---

# 6. Reading a scan afterwards

## What is on the object

```python
# — identity —
scan.path, scan.spectra_type, scan.spectroscopy, scan.source_metadata

# — shapes —
scan.n_points / n_pixels, scan.n_sweeps, scan.n_declared_sweeps

# — the measured axis —
scan.wavelength         # nm, ascending, as the file wrote it
scan.energy             # eV, ascending  (hc/λ, then argsorted)

# — signals, wavelength space —
scan.spectra            # THE FILE'S OWN COUNTS.  never mutated.
scan.spectra_roi1/roi2  # both ROIs, always present
scan.spectra_cr         # cosmic-ray repaired, or None
scan.cosmic_ray_mask    # which pixels moved, or None
scan.contrast           # (S−R)/R, or None

# — signals, energy space (all ascending in energy) —
scan.energy_spectra                  # Jacobian per apply_jacobian, no background
scan.energy_spectra_pre_jacobian     # never Jacobian; same object when it is off
scan.energy_spectra_bg               # background-corrected, or None
scan.energy_contrast                 # Jacobian NEVER applied — it cancels in a ratio
scan.best_energy_spectra             # _bg if available, else energy_spectra

# — instrument state —
scan.parameters, scan["V_A"], scan.get_parameter("V_A", scale)
scan.varying_parameters()
scan.power, scan.scanner_x, scan.scanner_y

# — device —
scan.gates, scan.is_dual_gated, scan.gate_mode
scan.v_top, scan.v_bot, scan.v_channel, scan.ef, scan.carrier_density
scan.i_top, scan.i_bot, scan.i_channel        # role-backed, need gates=

# — axis for plotting —
scan.sweep_type, scan.sweep_axis, scan.sweep_axis_label, scan.signal_label
scan.sweep_grid()
```

## `best_energy_spectra` — the "just give me the right array" accessor

Returns `energy_spectra_bg` if a background was supplied, else `energy_spectra`, so
downstream code need not know which. It **never returns the contrast**, even when a
reference was given: contrast is a *different quantity*, not a better-corrected one,
and it is negative-going — a peak fit whose model decays to zero in the wings would
give quietly meaningless numbers, and a PL colour bar would silently start meaning
ΔR/R₀. Ask for `energy_contrast` explicitly.

`loaders._resolve_spectra` mirrors this with `spectra_source=` — the `source=` of
`get_spectrum_at`, and what `plotting` imports — and `"best"` there
follows the same rule.

## The three properties that must never raise

A diagnostic that dies is no diagnostic. These stay permissive on purpose:

| Property | Behaviour without a declaration |
|---|---|
| `gate_mode` | works — the correlation between two rows is symmetric, so the field-like/doping-like verdict is unchanged by transposing them. Only the *wording* changes: `"single gate driven ('V_B')"` instead of `"bottom-gate only"`. |
| `__repr__` | prints, and says the wiring is undeclared plus the call shape to fix it |
| `ef` | returns `None` when no geometry was given (saying "no field was computed" needs no knowledge of which electrode is which); raises only when a geometry *was* given and the roles are missing |

`__repr__` is the discoverability half of the gate design: the raises catch you when
you ask for something specific, the repr tells you before you ask. Given that the
failure mode is a plot that looks entirely normal, the intervention has to happen
where you are already looking.

## `sweep_grid()` and the declared nest

Two things, and the split between them is the same one `gate_mode` has with `gates=`:
**detection reports, declaration decides.**

`sweep_grid()` detects a 2-D raster flattened into one file (`n_fast × n_slow ==
n_sweeps` exactly) and reports it as a `SweepGrid` named tuple. It searches the **raw
parameter rows**, so a nest whose axis is a *derived* quantity is reported through the
channels that carry it — and its candidate ordering comes from `varying_parameters()`,
so where two pairs verify equally well it returns whichever `varying_parameters()`
ranked first — on a raster taken during an anti-symmetric gate sweep, that may be
the gates rather than the scanners. It is a diagnostic that says what to declare,
and nothing keys off it.

The nest itself is declared at load with `fast_sweep=` / `slow_sweep=`, inner axis
first *by name* rather than by tuple position:

```python
scan = AttoCubeSpectralSweep(path, spectra_type="RC",
                             fast_sweep="Scanner X", slow_sweep="Scanner Y")
scan.nesting          # SweepNesting: both coordinate axes, labels, units
scan.is_nested        # the predicate as_grid() and the accessors need
scan.as_grid(scan.spectra)     # (n_points, n_slow, n_fast) — a view
```

Both go through `_resolve_sweep`, the same resolver `sweep=` uses, so a
`_SWEEP_TYPES` key works as well as a raw row: `fast_sweep="electric_field",
slow_sweep="power"` is the case the design exists for, where both gates move together
and no single row is the axis.

**A nest is not a `_SWEEP_TYPES` entry, and `fast_sweep` is not an alias of `sweep`.**
That registry answers *"which 1-D array of length `n_sweeps` is the sweep axis"*, and
a raster has two. And "sweep" means the *flattened point* everywhere else in the
package (`n_sweeps`, `sweep_index`, `sweep_axis`), so `sweep=` keeps answering "what
labels each flat point" while the nest is a separate statement about structure. On a
nested scan `sweep=` is normally omitted and `sweep_axis` is the flat index. Full
argument in **E14**.

`spectra` keeps shape `(n_points, n_sweeps)` either way — a declaration must not
change the rank of an attribute.

`axis=` is the fourth entry point onto the same resolver: it says which quantity a
positional value is read against, so a sweep declared in one coordinate can be
searched in another.

```python
scan.get_spectrum_at(15.0, axis="top_voltage")   # flat sweeps only
scan.nearest_index(15.0, axis="top_voltage")
```

Flat sweeps only, because on a nest an arbitrary quantity matches `n_slow` points or
one depending on how the scan was driven — the return rank would follow the data
rather than the call.

Such a quantity need not label its points individually, so a coordinate can name
several of them. `nearest_index` **warns** and returns the first, a single `int`
being its whole contract; `get_spectrum_at` **raises**, because it returns data and
the API already holds the complete answer — a declared nest gives every match at
once through `fast=`/`slow=`. The same fact is checked once at load, against
whatever `sweep=` names:

> **Does the sweep axis give each point its own value?** A map positions its
> spectra along it, so points sharing a value land on top of each other and only
> one is drawn. Both quantities of a nest repeat, and so do deliberate repeat
> measurements — all of them warn, with no grid-detection guard, since
> `sweep_grid()` sees nothing in a field × power nest.

---

# 7. The correction chain

This is the part most likely to be broken by a well-meaning edit, because the order
is physics.

```
   file counts  ──►  spectra                          (never mutated, ever)
        │
        ▼  cosmic_rays=          FIRST — a spike biases everything downstream
   spectra_cr / cosmic_ray_mask
        │
        │   signal = spectra_cr if repaired else spectra
        ├──────────────────────────────────────────────┐
        ▼  (wavelength space)                          ▼  (no bg)
   bg_region_nm  → subtract_background            energy_spectra
   bg_spectrum   → subtract_spectrum         = jacobian?(signal), argsorted
        │
        ├──► energy_spectra_bg  = jacobian?(corrected), argsorted
        │
        └──► reference= → spectral_contrast(corrected, reference)
                 └──► contrast, energy_contrast   (Jacobian NEVER applied)
```

The four rules encoded in that diagram:

1. **Cosmic rays first.** Both corrections below read the counts as signal: a spike
   inside the `bg_region` window pulls the pedestal estimate up, and a spike in
   either array of a contrast biases the ratio non-linearly. Also in *wavelength*
   space, because the 3-point Laplacian the detection is built on assumes uniform
   sample spacing — which the detector axis has and the energy axis does not.

2. **Background before the Jacobian.** The Jacobian multiplies by λ², so it turns a
   flat dark pedestal `B` into `B·λ²/hc` — a baseline curving towards the red rather
   than an offset a fit can absorb. Construction *warns* if `apply_jacobian=True`
   with no background supplied.

3. **Background off both arrays before the ratio.** A pedestal in either biases a
   contrast non-linearly.

4. **The Jacobian is never applied to the contrast**, whatever `apply_jacobian` says.
   `(S·λ²/hc)/(R·λ²/hc) = S/R` exactly — it cancels identically, and applying it to
   the numerator alone would be an error.

**Sentinels.** `energy_spectra_bg` is `None` when no background was supplied, and the
test for that is `corrected is not signal` — comparing against `signal`, not
`spectra`, so a cosmic-ray repair on its own does not masquerade as a background
subtraction. `energy_spectra_pre_jacobian` is the *same object* as `energy_spectra`
when the Jacobian is off, and a separate array when it is on, so both representations
are always reachable.

**Grid mismatch on an auxiliary spectrum raises rather than interpolating.**
Resampling changes the numbers and smooths the data, so it is a correction and cannot
be a default. `_resolve_aux_spectrum` accepts a bare array precisely so a caller who
has aligned the axes themselves has a route in, with no extra API.

---

# 8. The other loaders

| Class | Reads | Notes |
|---|---|---|
| `AttoCubeSpectralSweep` | one spectral CSV, or an `.h5` | the main one |
| `AttoCubeTRPLSweep` | a **directory** of temporal CSVs, or an `.h5` | separate class, no `spectra` attribute |
| `AttoCubePLVabScan` | — | compatibility shim over the above; raises `FutureWarning` |
| `SingleSpectrum` | a 2-row CSV (row 0 = λ/nm, row 1 = counts) | mirrors the sweep's attribute names so plotting works unchanged |
| `AttoCubePLScanRealSpace` | a directory of numeric-grid CSVs | image sequence for diffusion work |
| `SingleImage`, `AttoCubeSampleImage`, `AttoCubePLImage` | one numeric-grid CSV | share `_AttoCubeImage` |
| `AttoCubeLaserReferenceImage` | one numeric-grid CSV | fits the laser spot centre and 1/e² radius on construction |

## `AttoCubeTRPLSweep` — why a directory

A TRPL sweep arrives as **one file per sweep point**, each carrying its own full
57-row parameter snapshot, plus a **metadata companion**: a *spectral*-layout file
whose `Par_i` columns hold one snapshot per point and whose Wavelength/ROI columns
are identically zero.

Two consequences the assembly code exists for:

- **The companion collides on `iter_0` with the first data file and is written last,
  so classification is by content, not filename.** `_read_block_layout` reads one
  header line per file and sorts them.
- **Order by the integer in `_iter_N`**, via `_order_by_iter`. Lexicographic order
  puts `iter_10` before `iter_2`. The helper is module-level and shared with
  `AttoCubePLScanRealSpace`, which has the same problem over image frames; it takes a
  plain `list[Path]`, so this call site re-attaches each file's layout by dict lookup
  after the sort. It warns on a missing suffix, a gap, and an index claimed by more
  than one file, and repairs none of them.

The per-file time axes are not bit-identical (bin width varies in its seventh
figure), so `_assemble` compares them with `time_rtol`, never for equality.

It is a **separate class rather than a mode of `AttoCubeSpectralSweep`**, and has no
`spectra` attribute — the signal is `decays`, the axis is `time`, and there are no
ROIs. Everything it shares lives in `_AttoCubeSweep`.

`_TRPL_TIME_UNIT` is the single place the ns/4-ps-bin assumption is written down. Any
fitted lifetime inherits it, and it is consistent with the Picoharp rows and a
~78 MHz rep rate but **not independently confirmed**.

---

# 9. HDF5 round trip

`scan.to_hdf5(path)` → `hdf5.write_sweep`; reading is automatic when you hand a
loader an `.h5`.

The layout is in the `hdf5.py` module docstring and is the authority; the short
version:

```
/                    format, format_version, created, toolkit_version
├── metadata/        spectra_type, axis_kind, sweep_*, curated_labels,
│   └── geometry/    curated_scales, gates, cosmic_rays, + provenance
├── auxiliary/       bg_spectrum, reference   (arrays, not paths)
├── axes/            wavelength | time        ← named for the quantity it holds
├── parameters/      one dataset per instrument row, raw file units
└── spectra/ | decays/
```

Four design points a maintainer needs:

- **Nothing derivable is stored.** No energy axis (`hc/λ`), no energy-space spectra,
  no repaired spectra, no sweep axis. Storing them invites two copies to disagree,
  and would freeze one session's loading choices into the archive where a later
  reader could not tell them from raw data.
- **Scalars live in group attributes**, not 0-d datasets — the idiomatic HDF5 home,
  and it keeps `h5ls -v` readable.
- **Dicts go through `_JSON_ATTRS`** as JSON strings, because HDF5 attributes have no
  mapping type. `gates` is one of them.
- **`gates` is written as its own attribute, conditionally.** This is the
  *laundering* fix: `curated_labels` always contains `v_top`/`v_bot` (they have
  defaults), so it alone cannot distinguish a declared mapping from a defaulted one.
  Without the separate record, CSV (undeclared) → HDF5 → reload would come back
  looking like a stated fact. **Presence is the record of declared-ness.** The test
  that pins this is the one least worth losing, because a regression is invisible —
  everything still works, it just quietly starts lying again.

`FORMAT_VERSION` is gated on the **major** on read. Bump the minor when adding a
field; bump the major only for a change an older reader would mis-read (2.0 moved the
auxiliary spectra out of `/metadata`, which an old reader would silently drop).

---

# 10. `processing`, `fitting`, `plotting`

## `processing` — arrays in, arrays out

No objects, no files. Everything takes an `axis=` and respects the
`(n_points, n_sweeps)` convention. The one exception to "no matplotlib" is
`_draw_region_box`, which lives here so that `loaders` and `diffusion` share one
drawer instead of forking it (D1).

```
normalise_peak / normalise_area      subtract_background / subtract_spectrum
smooth_median / smooth_savgol        crop / _window_slice
wavelength_to_energy / energy_to_wavelength / jacobian_correction_wvl2E
spectral_contrast                    remove_cosmic_rays
```

`_window_slice` turns a `(lo, hi)` window on a measured axis into a `slice`, so the
axis and the signal are cut by one object and both stay views. It is what
`AttoCubeSpectralSweep.pixel_slice` and `fitting.fit_scan_peak` are both built on, and
it refuses an empty window and warns on a clipped bound — which is why it sits next to
`crop`, the remaining spelling of the same window that does neither. (The third is
`plot_power_series`'s inline mask.) Unifying `crop` on it would make it raise where it
now returns an empty array, so that is its own change.

`spectral_contrast` returns `(contrast, reference_guarded)` — the second is the
reference actually used, so the caller can see what it divided by. That is the
*return the evidence* rule: masks, flags, and diagnostics come back so the researcher
can make the decision.

`remove_cosmic_rays` is the worked example of the whole corrections philosophy: it
returns `(repaired, mask)`, keeps `cross_sweep_veto=False` (conservative,
assumption-free, shape-invariant), and **warns when a pixel is flagged in most
sweeps**, because that is precisely the case where the conservative default is the
damaging one.

## `fitting` — arrays in, dataclasses out

`FitResult` and `DipoleResult` carry parameters, uncertainties, R², and the model
label. Peak models all take `baseline={"constant"|"linear"|"none"}`.

**`baseline="constant"` defaults *on*, and that is a stated exception to
corrections-are-opt-in.** It is a *model term*, not a modification of the data:
omitting it does not preserve anything, it silently migrates the dark-count pedestal
into fitted amplitude and FWHM. Do not "fix" it to `"none"`.

Note `voigt_approx` is implemented and reachable from no `fit_*`. Per *delete before
documenting*, that is deletion material, not documentation material.

## `plotting` — returns handles, takes few parameters

Every function returns `(fig, ax, <artist>)` and never calls `plt.show()`.

**The return contract *is* the styling API.** `line.set_color("k")`,
`mesh.set_clim(0, 1)`, `ax.set_xlim(...)` are one line each at the call site, which
is why there is no `color=` parameter. The corollary matters as much: **a function
that draws several artists must return them all**, or callers have no route to
restyle and the parameters grow back.

The test for whether something earns a parameter: **does it change the numbers, or
only the pixels?**

| Earns a parameter | Does not |
|---|---|
| corrections (`median_kernel`, `smooth_sigma`, `bg_stat`) | colours, line widths, fonts |
| which data is shown (`x_axis`, `spectra_source`, `normalize`) | anything whose body is `artist.set_<thing>(value)` |
| context the function cannot infer (`pixel_scale`, `origin`, `laser_ref`) | anything `set_style()`/rcParams already owns |
| structure (`panels`, `ax`, `save`) | |

`plot_diffusion_cloud` is the standing counter-example — ~30 parameters, half of them
enumerated styling, and it returns `result` instead of its artists. It predates the
rule. Don't copy it.

The animation system is the shape to aim for: `animate_panels(panels, …)` takes a
list of `AnimationPanel` objects, so any subset, order, or combination works with no
special-casing. `animate_wl_pl_spectra` is a public function that builds its panels
and *delegates* — a new entry point, no new engine.

---

# 11. Invariants

A change that breaks one of these is a bug even if the tests pass.

1. **`scan.spectra` is the file's own counts.** Corrections add attributes; they
   never overwrite. Same for `wavelength`, `parameters`, `decays`.
2. **Signals are `(n_points, n_sweeps)`; parameters are `(n_sweeps,)`.**
3. **Loading is not deciding.** No correction may move into a loader's default path,
   however obviously right it looks.
4. **A destructive default is forbidden; a *permitted* default that can still destroy
   a feature must warn**, naming what was affected.
5. **The gate mapping is declared through `gates=` and nowhere else.** `v_top` /
   `v_bot` are rejected as `curated_labels` keys. One fact, one spelling.
6. **`gate_mode` and `__repr__` never raise; `ef` returns `None` without a geometry.**
7. **HDF5 stores nothing derivable and never replays a correction on read.**
8. **An undeclared `sweep=` is the index**, never an auto-detected parameter. Nor is
   a nest ever inferred: `sweep_grid()` reports, `fast_sweep=`/`slow_sweep=` decides.
9. **A declaration never changes the rank of an attribute.** `spectra` is
   `(n_points, n_sweeps)` nested or flat; the grid is a view from `as_grid()`.
10. **Vectorise, but name the shapes.** Broadcasting and fancy indexing are wanted;
   the condition is that a reader never has to re-derive the trick from the
   expression. Say what the operation does and over what:

   ```python
   # (n_pixels, 1) broadcast against (n_pixels, n_sweeps): one baseline per
   # pixel, subtracted from every sweep.
   corrected = spectra - baseline[:, None]
   ```

11. **New module → `docs/api/<module>.md` with a `:::` directive *and* a `nav` entry
    in `mkdocs.yml`.** `python -m mkdocs build --strict` must stay green.

---

# 12. Where to look when…

| You want to | Go to |
|---|---|
| add a new sweep axis | write the property, add a `_SWEEP_TYPES` row (+ `_SWEEP_REQUIRES` if it needs a specific row) — usable as `sweep=`, `fast_sweep=` and `slow_sweep=` at once, since all three share `_resolve_sweep` |
| reshape a raster, or pick a spectrum out of one | declare `fast_sweep=`/`slow_sweep=`, then `as_grid()` / `get_spectrum_at()`; §`sweep_grid()` above |
| add a new input format | write a decoder returning the §2.1 payload; add a suffix to the dispatch in `_decode` |
| add a curated parameter | one row in `_AttoCubeSweep._CURATED`, plus a property |
| understand a gate refusal | `dev/E7b-E7c-gates.md`, Part 8 (the refusal matrix) |
| know why a number is what it is | `.claude/CLAUDE.md`, *Physics conventions* |
| know whether a bug is known | `.claude/CLAUDE.md`, *Known issues* → `dev/audit-2026-07.md` |
| add a plotting option | first ask whether the returned artist already does it (§10) |
| run the tests | `conda run --no-capture-output -n viz-sci-plot python -m pytest -q` |

**One environment note that will otherwise cost you an afternoon.** Activate
`viz-sci-plot` before anything that imports numpy, or use `conda run`. Calling the
env's `python.exe` directly imports numpy fine and then dies at the **first BLAS
call** — matmul, `np.corrcoef` (so `gate_mode`), skimage `regionprops` (so
`_fit_laser_spot`) — with `Windows fatal exception: code 0xc06d007f`. It is a
delay-load failure, so there is no traceback, and it lands on a different test each
run. Nothing is wrong with the environment when that happens.

---

# 13. Reading the constants tables

Every module-level `_TABLE` in `loaders.py`, in one place, since they are the
skeleton of the whole design:

| Table | Maps | Adding an entry means |
|---|---|---|
| `_SWEEP_TYPES` | sweep type → (property name, label, unit) | a new declarable sweep axis |
| `_SWEEP_REQUIRES` | sweep type → curated rows it needs | load-time validation for it |
| `_GATE_ROLE_CURATED` | role → curated attribute | (fixed — the one gate fact) |
| `_GATE_CURATED` | *(derived)* the gate curated names | — |
| `_ROLE_FOR_CURATED` | *(derived)* curated name → role | — |
| `_GATE_ELECTRODES` | *(derived)* roles that are gates | — |
| `_GATE_ROLES` | *(derived)* + `"channel"` | — |
| `_BLOCK_LAYOUTS` | block field names → layout kind | a new export layout |
| `_CLASS_FOR_KIND` | layout kind → class name | (for the "wrong class" error) |
| `_CURATED` | curated attribute → (row label, scale, unit) | a new promoted parameter |
| `_COSMIC_RAY_KEYS` | *(derived from a signature)* | — |
| `constants.SPECTROSCOPY_TYPES` | code → full name | a new measurement type |
| `constants.SIGNAL_LABELS` | code → (axis name, unit) | its axis label |
| `hdf5._AXIS_KIND_FOR_LAYOUT` | layout kind → dataset name/units/group | a new axis kind |
| `hdf5._JSON_ATTRS` | which metadata keys are JSON-encoded | a new dict-valued metadata key |
| `loaders._SPECTRA_SOURCES` | `spectra_source=` / `source=` → attribute name | a new plottable array |

The four derived gate names all come from `_GATE_ROLE_CURATED` on the lines below it,
so they cannot drift. (`_GATE_CURATED` and `_ROLE_FOR_CURATED` currently hold the
identical collection of strings — the tuple and the dict's keys — which is one name
more than the design needs. Noted, not yet collapsed.)
