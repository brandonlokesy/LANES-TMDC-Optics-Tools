# E7b / E7c — the `gates` declaration

Implementation record for two changes made 2026-08-05, both to the same mechanism.
Status entries live in `dev/audit-2026-07.md` under **E7b** and **E7c**; standing
conventions are in `.claude/CLAUDE.md`. This file is the *how and why* in full: what
each piece does, and how control flows from `__init__` through to a plot.

| | |
|---|---|
| **E7b** | The channel-to-gate mapping was silently defaulted. Now required. |
| **E7c** | `gates` could not describe a single-gated device. Now it declares topology, and carries a carrier-density path. |

Line references are to `src/tmdc_optics_tools/loaders.py` unless stated.

---

# Part 1 — E7b: the mapping was a default

## The incident

A voltage was applied to a device's **bottom** gate. The code read it as the top
gate. Nothing complained.

The cause was three lines of table:

```python
_CURATED = {
    "v_top": ("V_A", 1.0, "V"),      # class-level on _AttoCubeSweep
    "v_bot": ("V_B", 1.0, "V"),
    ...
}
```

`V_A` and `V_B` are acquisition channel names — what the instrument wrote into the
export. `v_top` and `v_bot` are *physical roles*. The table asserted a correspondence
between them that is set by where someone plugged a wire, and which appears in no
file. Re-plug the wire and the table is wrong, with no signal that anything changed.

Three consequences, worst last:

1. `sweep="bottom_voltage"` passed validation and returned the **top** channel. The
   resulting plot looks completely normal.
2. `ef` was mirrored, so any dipole length or Stark coefficient extracted from it had
   the wrong sign.
3. `gate_mode` returned `"top-gate only"`. The repr did not merely assume wrong — it
   **asserted** wrong, in the one place a researcher looks to check.

## Why renaming to `V_A`/`V_B` was rejected

The obvious reaction is: the roles are ambiguous, so stop using role names. Call
everything `v_a`/`v_b` and let the caller sort it out.

That makes things worse. `DeviceGeometry.electric_field` computes `v_bot - v_top`
(`:452`). The *sign* of that difference is the sign of the field, hence of every
dipole extracted from it. Written as `v_a - v_b` the expression becomes
unambiguous-looking and still wrong half the time: the ambiguity has not been
removed, it has been moved out of a named, recordable, repr-printed,
HDF5-round-tripped mapping and into the researcher's head — which is exactly where it
resolved wrongly in the first place.

So the role layer keeps role names, because that is the layer where the sign
convention lives. The channel layer was already fully reachable and needed no new
API:

```python
scan["V_A"]                  # sugar for get_parameter -> raw row
scan.parameters["V_A"]       # the same array
sweep="V_A"                  # raw-row sweep axis, no role implied
```

## Why the fix is a refusal, not a warning

The package cannot know the wiring. But it was not the *not knowing* that caused the
error — it was proceeding anyway. Two precedents already in the codebase say what to
do with a fact no file records:

- `spectra_type=` is keyword-only with **no default**, because the value is written
  into exported metadata and trusted thereafter, so a guess would outlive the session
  that made it.
- An undeclared `sweep=` resolves to the sweep **index**, never to an auto-detected
  parameter.

The channel-to-electrode mapping is the same kind of fact, and was the only one of
the three still guessing. A warning was considered and rejected: this failure
produces a plausible, publishable-looking plot, and a warning above such a plot in a
notebook is scrolled past.

---

# Part 2 — E7c: one gate and a grounded channel

## The second incident

Immediately after E7b landed, a real device did not fit: **one electrode drives the
bottom gate, the other contacts the TMDC to ground it.** There is no top gate. No
assignment of `{"top", "bottom"}` is correct — including the transposition that had
just been the worry.

## Why this is not a wiring question at all

`electric_field` is exact *within its assumptions*, and this geometry breaks two of
them.

```
eps_2D * E_2D = eps_stack * E_stack ,   E_stack = (V_BG - V_TG) / d_TOT
```

**There is no second boundary.** `E_stack` is a potential *difference* across
`d_TOT`, which requires an equipotential at each end of the stack. With no top gate
the upper surface has no defined potential. There is nothing to subtract.

**The TMDC is now an electrode, not a slab inside the capacitor.** The derivation
assumes no free charge between the gates; grounding the TMDC through a contact
introduces exactly that. Field lines from the bottom gate terminate on the induced
sheet charge, so `V_BG` drops across `d_hBN,bottom` — not across `d_TOT`, and not
with `eps_stack` in it.

Had the grounded row been passed as `v_top` (it sits at 0, so `v_bot - 0 = v_bot`),
the result would have been `V_BG/d_TOT * eps_stack/eps_2D`: wrong denominator by
about 2x for a 53/46 nm stack, reporting a field in a slab that is now the
terminating electrode. Not a small error — a quantity that does not exist in this
device.

**Underneath both:** one gate is one degree of freedom. Field and density are locked
together. Independent control of the two is precisely what a dual-gate
anti-symmetric sweep buys, which is why `gate_mode`'s field-like/doping-like verdict
matters. A single-gate sweep is a doping sweep by construction.

## What the device actually controls

```
C_BG = eps_0 * eps_hBN / d_hBN,bottom       [F/m^2]
dn   = C_BG * (V_BG - V_ref) / e            [cm^-2]
```

`dn/dV = C/e` is **purely geometric** — it needs `d_hbn_bottom` and `eps_hbn`, both
already on `DeviceGeometry`. Note the TMDC thickness does *not* enter, because the
TMDC is the counter-electrode: another sign that `eps_stack`/`d_stack` are the wrong
tools here. For 46 nm hBN at eps = 3.9 this is 4.685e11 cm^-2 V^-1.

Absolute `n` needs the threshold at which the channel populates — a transfer curve
or the PL charging step. Per-device, in no file. So it is not defaulted; `v_ref` is
exposed and documented as a *gate voltage*, making the return a density difference
unless the caller supplies a measured threshold.

---

# Part 3 — the data model

Five module-level names, `:620-635`, all derived from one dict so nothing can drift:

```python
_GATE_ROLE_CURATED = {"top": "v_top", "bottom": "v_bot"}   # role -> curated attr
_GATE_CURATED      = tuple(_GATE_ROLE_CURATED.values())    # ("v_top", "v_bot")
_ROLE_FOR_CURATED  = {attr: role for role, attr in _GATE_ROLE_CURATED.items()}
_GATE_ELECTRODES   = tuple(_GATE_ROLE_CURATED)             # ("top", "bottom")
_GATE_ROLES        = _GATE_ELECTRODES + ("channel",)       # + the TMDC contact
```

Each exists for a specific lookup direction:

| Name | Answers | Used by |
|---|---|---|
| `_GATE_ROLE_CURATED` | "which curated entry backs role X" | label application `:996`, kwarg names `:1550` |
| `_GATE_CURATED` | "is this curated name a gate row" | `curated_labels` rejection `:979`, HDF5 strip `:963` |
| `_ROLE_FOR_CURATED` | "which role does this curated name need" | sweep gating `:1214` |
| `_GATE_ELECTRODES` | "which roles are gates across a dielectric" | `is_dual_gated`, `ef`, `gate_mode` |
| `_GATE_ROLES` | "what may appear in a `gates` dict" | validation `:1007` |

**The list that deliberately does not exist** is `_GATE_SWEEPS = ("electric_field",
"top_voltage", "bottom_voltage")`. That information is already in `_SWEEP_REQUIRES`,
and a second copy could drift: add a gate-backed sweep type next year, forget the
list, and the requirement silently does not apply. Instead `_resolve_sweep` derives
it by mapping `_SWEEP_REQUIRES[sweep]` through `_ROLE_FOR_CURATED` (`:1213-1219`), so
a new sweep type inherits the requirement from the declaration it already has to
make.

## What a `gates` value means

```python
gates = {"top": "V_A", "bottom": "V_B"}       # dual-gated
gates = {"bottom": "V_A", "channel": None}    # bottom-gated, TMDC hard-grounded
gates = {"bottom": "V_A", "channel": "V_B"}   # ... contact on a recorded channel
```

- **The keys present describe the device.** A missing `"top"` is not an omission, it
  is a statement that the device has no top gate. This is why topology needed no
  second parameter and no flag: what is computable is *derived* from which roles are
  there.
- **`"channel"` is not a gate.** It contacts the TMDC itself — inside the stack, not
  across a dielectric from it. It carries no thickness, enters no field, and is
  excluded from `gate_mode`. Its job is to record that the device is contacted, which
  is what a density is referenced to and what makes a single-gate declaration
  unambiguous.
- **`None` as a value** means that electrode is tied to ground and no acquisition row
  records it. Its voltage is zero at every sweep point — a consequence of the
  declaration, not a guess about a missing row.

---

# Part 4 — call trace: construction

Take the single-gate case end to end.

```python
scan = AttoCubeSpectralSweep(
    "doping.csv", spectra_type="PL", sweep="carrier_density",
    geometry=geom, gates={"bottom": "V_A", "channel": None},
)
```

### Step 1 — `__init__` (`:2290`)

`gates` is keyword-only (everything after the bare `*` is). It does nothing here but
forward. The mutual-exclusion check on `bg_region_nm`/`bg_region_eV` runs, then:

```python
payload = self._decode_and_describe(
    path, spectra_type=spectra_type, geometry=geometry, gates=gates,
    curated_labels=curated_labels, curated_scales=curated_scales,
)
```

### Step 2 — `_decode_and_describe` (`:911`), everything independent of the spectral axis

Order inside this method is load-bearing. It runs:

**2a.** `self.path`, then `payload = self._decode(path)` — dispatches on suffix to the
CSV parser or the HDF5 reader. `meta = payload["metadata"]` is essentially empty for
CSV and fully populated for HDF5.

**2b.** `self.parameters = payload["parameters"]` (`:936`). **Every labelled row is
now available.** This matters: the gate machinery needs to be able to name the file's
actual rows in its errors, and that is only possible after this line.

**2c.** `self.geometry` resolved from the argument, else from `meta` (`:942`).

**2d.** The mapping (`:950`):

```python
self._gates = self._resolve_gates(
    gates if gates is not None else meta.get("gates")
)
```

Argument wins over file, matching how `sweep` and `curated_labels` already behave.
Resolved **before** the curated registry because it supplies two of that registry's
labels.

Inside `_resolve_gates` (`:1007`, a `@staticmethod` — pure validation, no instance
state):

```
gates is None                        -> return None      (undeclared is legal)
not a dict                           -> ValueError
keys - _GATE_ROLES non-empty         -> ValueError, lists valid roles
keys & _GATE_ELECTRODES empty        -> ValueError  "at least one gate electrode"
exactly one gate and no "channel"    -> ValueError  "ambiguous"
otherwise -> {role: gates[role] for role in _GATE_ROLES if role in gates}
```

Two of those deserve explanation.

*The ambiguity rule.* `{"bottom": "V_A"}` alone is refused. It cannot be told from a
two-gate device whose top gate was forgotten — and treating it as single-gated would
reintroduce, one level down, exactly the silent assumption E7b removed. Requiring the
`"channel"` makes the single-gate case a **statement**. Physically it is also the
honest requirement: a lone gate with a floating TMDC defines neither a field nor a
density.

*The return is a copy*, rebuilt in canonical role order. Without it,
`scan.gates["top"] = "V_B"` after construction would silently rewrite what the scan
claims its wiring was while `_curated` still pointed at the original row — the object
would disagree with itself. This matters more than it looks, because the mapping is
written into exported HDF5: a post-hoc-editable scan can export a false provenance
record. The `gates` property (`:1341`) returns `dict(self._gates)` for the same
reason, and `test_gates_are_recorded_on_the_scan` pins it.

**2e.** The curated registry, `:957-1002`. Built as a dict of **lists** so labels
(index 0) and scales (index 1) can be mutated, then frozen to **tuples** at the end so
nothing downstream can alter a scan's parameter mapping after load.

```python
self._curated = {name: list(cfg) for name, cfg in self._CURATED.items()}
```

Then `meta_labels` (`:963`) — the HDF5-sourced `curated_labels` with the gate entries
**stripped**:

```python
meta_labels = {name: value
               for name, value in (meta.get("curated_labels") or {}).items()
               if name not in _GATE_CURATED}
```

This is the subtlest line in the change; Part 6 explains why it must be silent while
the caller-supplied case must raise.

Then the merge loop over four stores in precedence order:

```python
for store, idx in ((meta_labels, 0), (curated_labels, 0),
                   (meta.get("curated_scales"), 1), (curated_scales, 1)):
```

with two guards per entry: unknown curated name -> `ValueError` (`:974`), and a gate
name used as a **label** key -> `ValueError` pointing at `gates=` (`:979`). The
`idx == 0` condition restricts that second guard to labels: `curated_scales` still
reaches the gate entries, because a scale is a unit conversion and says nothing about
which electrode a channel reached.

Then the declared labels are applied (`:996`):

```python
for role, attr in _GATE_ROLE_CURATED.items():
    label = (self._gates or {}).get(role)
    if label is not None:
        self._curated[attr][0] = label
```

`.get(role)` handles both a role left out (single-gate) and a role declared `None`
(grounded) with the same expression, and `(self._gates or {})` handles the undeclared
case. In all three of those the `_CURATED` **default label survives** — and that is
fine, because nothing will read it. The refusal lives on `self._gates`, never on the
label, so a defaulted label can never be mistaken for a stated one. This separation
is the core of the whole design.

Finally the freeze to tuples (`:1002`), which is why the label application had to come
before it.

### Step 3 — back in `__init__`: arrays

`self.wavelength`, `spectra_roi1`, `spectra_roi2`, `_validate_payload()`, the `roi`
selection and its all-zeros warning, `apply_jacobian`, `contrast_mode`,
`reference_scale`, and the auxiliary spectra. **Nothing gate-related.** The point of
this step for our purposes is that `n_sweeps` now exists — which is why the sweep axis
cannot be resolved before it.

### Step 4 — `_bind_sweep_axis(sweep, sweep_label, sweep_unit)` (`:1055`)

Falls back to `meta` for each of the three, then calls `_resolve_sweep`, unpacking
into `self.sweep_type`, `self._sweep_source`, `self._sweep_label`,
`self._sweep_unit`.

### Step 5 — `_resolve_sweep` (`:1194`), the load-time validation

For `sweep="carrier_density"`, in order:

**5a.** `sweep is None` -> `"index"`. Not our path.

**5b.** `sweep in _SWEEP_TYPES` -> yes. `source = "carrier_density"`,
`default_label = r"$\Delta n$"`, `default_unit = r"cm$^{-2}$"`.

**5c.** The gate-role loop (`:1212-1219`). `_SWEEP_REQUIRES` has **no**
`"carrier_density"` entry — deliberately, because its requirement depends on which
roles were declared and a static table cannot express that. So `.get(sweep, ())`
yields `()` and this loop is a no-op here.

For contrast, on `sweep="electric_field"` the same loop iterates
`("v_top", "v_bot")`, maps each through `_ROLE_FOR_CURATED` to `"top"`/`"bottom"`,
and calls `_require_role` on each. On our single-gate device that raises at `"top"` —
at **load time**, before any array is read.

**5d.** The grounded check (`:1220`). If a required role was declared `None`, its axis
would be all zeros, so it is refused as an axis rather than silently plotted flat.

**5e.** The row-presence check (`:1230`). Also empty for `carrier_density`. Its error
message branches on `name in _GATE_CURATED` to point at `gates=` or at
`curated_labels`, since the old text unconditionally suggested `curated_labels` —
advice that now raises.

**5f.** The `electric_field` geometry check (`:1245`). Not our sweep.

**5g.** The `carrier_density` block (`:1255-1268`), placed here beside it for the same
reason — a requirement too dynamic for the table:

```python
if self.geometry is None:            raise ValueError(...)   # need a capacitance
self._require_role("channel", "sweep='carrier_density'")     # need a reference
for role in _GATE_ELECTRODES:
    if role in self._gates:
        self.geometry.gate_capacitance(role)                 # need that hBN
```

The third line is a **discard** — called only for its exception. It makes a missing
`d_hbn_bottom` fail at load with a clear message rather than at first plot.

**5h.** Returns `(sweep, ("curated", "carrier_density"), label, unit)`. The
`("curated", name)` pair is the *only* thing connecting a sweep type to the property
that computes it, and the next section is where it is used.

### Step 6 — the rest of `__init__`

Background window resolved to nm, corrections applied, contrast built if a reference
was given. Gate-independent.

---

# Part 5 — call trace: `_require_role`, the single refusal point

Five call sites funnel here, so the message is written once.

```python
def _require_role(self, role, what):        # :1384
    self._require_gates(what)               # (a) was anything declared at all?
    if role in self._gates:
        return                              # (b) yes, and this device has it
    extra = (" A single-gated device has no gate-to-gate potential difference "
             "and so no displacement field: carrier density is the quantity "
             "it controls."
             if role in _GATE_ELECTRODES and not self.is_dual_gated else "")
    raise ValueError(
        f"{what} needs the {role!r} electrode, which '{self.path}' does not "
        f"have: gates declared {sorted(self._gates)}.{extra}")
```

Two distinct failures, two distinct messages:

**(a) `_require_gates` (`:1368`)** — nothing was declared. The message states the
mechanism (either wiring is possible, no export records which, transposing flips a
dipole sign), gives both call shapes, lists the file's candidate rows, **and** names
the escape hatch. That last part is not politeness: many people hitting this error do
not need roles at all — they want to plot against a channel. Telling them only
"declare your wiring" pushes them into inventing a mapping to get past the error,
which is the worst available outcome.

The candidate list comes from `_gate_candidates` (`:1418`): the conventional gate rows
first *if this file has them*, then anything else that varied. A row constant at 0 V
is excluded — it is indistinguishable from an unused channel, so proposing it would be
a guess of the kind being removed. On the committed stark-shift export this prints
`['V_A', 'V_B', 'I_B', 'I_A', 'Excitation Power']`, enough to decide on sight.

**(b) the role is absent** — the device does not have this electrode. When the missing
role is a *gate* and the device is not dual-gated, the physics sentence is appended,
because "does not have a top electrode" alone invites the reader to add one rather
than to reach for density.

---

# Part 6 — call trace: downstream access

## `sweep_axis` — the funnel

```python
@property
def sweep_axis(self):                       # :1596
    kind, name = self._sweep_source
    if kind == "index": return np.arange(self.n_sweeps, dtype=float)
    if kind == "row":   return self.get_parameter(name)
    return getattr(self, name)               # <- "curated": the property
```

That last line is why the sweep registry needs no special-casing per type: the
`_SWEEP_TYPES` entry names a property, and `getattr` calls it. Every guard below is
therefore reached through the sweep axis too, which is what makes plotting safe
without plotting knowing anything about gates.

## `v_top` / `v_bot` (`:1432`, `:1443`)

```python
self._require_role("top", "v_top")
return self._gate_value("top")
```

Raise-on-access, not raise-at-load, matching this class's existing idiom: a curated
row a file does not contain is not an error, and the property raises only if
accessed. Consistency matters — a file loads, you inspect it, and you meet the
refusal only when you ask for something that depends on the missing fact.

## `_gate_value(role)` (`:1401`)

```python
label = self._gates[role]
if label is None:
    return np.zeros(self.n_sweeps)          # grounded, by declaration
attr = _GATE_ROLE_CURATED.get(role)
return (self._curated_value(attr) if attr is not None
        else self.get_parameter(label))
```

Gate electrodes route through `_curated_value` -> `get_parameter(label, scale)` ->
`self.parameters[label] * scale`, so a `curated_scales` override still applies. The
channel has no curated entry, so it reads the row directly. Safe to index
`self._gates[role]` without a guard because every caller has been through
`_require_role`.

## `v_channel` (`:1454`)

`_require_role("channel", "v_channel")` then `_gate_value("channel")`. A dual-gate
declaration says nothing about a contact, so this raises there — correct: the scan was
never told the device has one.

## `ef` (`:1490`)

```python
if self.geometry is None:
    return None                              # (1)
for role in _GATE_ELECTRODES:
    self._require_role(role, "ef")           # (2)
return self.geometry.electric_field(self.v_top, self.v_bot)
```

**(1) is an asymmetry, and deliberate.** Saying "no field was computed" requires no
knowledge of which electrode is which. Making this raise would be over-strict and
would break the `scan.ef is not None` idiom `plotting` already used correctly.

**(2) is redundant for correctness** — `self.v_top` would raise anyway — and present
for *message quality*. Without it the error reads `v_top needs the 'top'
electrode...` when the caller asked for `ef`. The caller asked for a field, and the
reason they cannot have one is about the device, so `ef` names itself.

## `carrier_density` (`:1510`)

```python
if self.geometry is None: return None                    # same asymmetry as ef
self._require_role("channel", "carrier_density")         # charge needs a source

channel_row = self._gates["channel"]
if channel_row is not None and channel_row in self.varying_parameters():
    warnings.warn(... f"varies by {span:.4g} V" ...)      # the reference moves

volts = {_GATE_ROLE_CURATED[role]: self._gate_value(role)
         for role in _GATE_ELECTRODES if role in self._gates}
return self.geometry.carrier_density(**volts)
```

The `volts` comprehension (`:1550`) is the topology paying off:
`_GATE_ROLE_CURATED[role]` is `"v_top"`/`"v_bot"`, which are exactly
`DeviceGeometry.carrier_density`'s parameter names, so declared roles become kwargs
directly. A gate the device lacks is **left out of the sum**, not passed as zero —
omitting it and passing `0.0` differ, because `C_i * (0 - v_ref)` is not zero for
non-zero `v_ref`.

**The warning.** A density is referenced to the contact that supplies the charge; a
contact that is itself being driven moves the reference under the axis. Legitimate for
a source-drain bias measurement, wrong for a doping sweep, and the file cannot say
which — so it reports the span it saw rather than choosing. This is the *"where a
permitted default can still destroy a feature, it must say so"* rule; it fires from
one place, and `sweep_axis` funnels through it, so a plot against the density axis
warns too.

## `DeviceGeometry.gate_capacitance` (`:454`) and `carrier_density` (`:500`)

```python
thickness = {"top": self.d_hbn_top, "bottom": self.d_hbn_bottom}
...
return EPS_0 * self.eps_hbn / (d_nm * 1e-9)
```

Only that gate's hBN. The TMDC is the counter-electrode, not a slab inside the
capacitor, so neither its thickness nor `eps_stack` appears —
`test_gate_capacitance_is_geometric_and_uses_only_that_gates_hbn` pins that a 5-layer
stack returns the same number, because that is the assertion most likely to be
"helpfully" broken later.

```python
sigma = sum(self.gate_capacitance(gate) * (np.asarray(v, float) - v_ref)
            for gate, v in supplied.items())
return sigma / E_CHARGE * 1e-4                # m^-2 -> cm^-2
```

Sums over supplied gates only; each injects charge through its own capacitance.
Signed, electrons positive. `EPS_0` and `E_CHARGE` were added to `constants.py`
rather than imported from scipy here, so physical constants keep one home.

## `gate_mode` (`:1707`) — the one that must not raise

This was the property that lied (`"top-gate only"` for a bottom-gate sweep), so the
tempting fix is to make it strict. That would be wrong, because **most of what it
reports needs no mapping at all.** Whether two channels moved anti-correlated or
correlated — field-like vs doping-like — is a property of the data, and
`np.corrcoef(a, b)[0, 1]` is *symmetric*, so the verdict is identical under
transposition. Making it raise would discard the one diagnostic that catches a doping
sweep mistaken for a field sweep, to protect against an error it cannot make.

So only the branch that names an electrode changed:

```python
if self._gates is not None:
    rows = {role: self._gates[role] for role in _GATE_ELECTRODES
            if self._gates.get(role) is not None}      # gates only, non-grounded
else:
    rows = {role: self._CURATED[attr][0]
            for role, attr in _GATE_ROLE_CURATED.items()}
rows = {role: lbl for role, lbl in rows.items() if lbl in self.parameters}
if not rows: return None
```

| Situation | declared | undeclared |
|---|---|---|
| nothing varies | `"gates static"` | same |
| one gate driven | `"bottom-gate only"` | `"single gate driven ('V_B')"` |
| both driven | `"dual-gate, anti-correlated (field-like)"` etc. | same |

The undeclared string is true whatever the wiring; `"top-gate only"` was not. Rows are
read via `self.parameters`, never `self.v_top`, precisely so this property keeps
working without a declaration. `"channel"` is excluded — it is not a gate. And a
partially-missing row now degrades to describing the other gate rather than returning
`None`, since reporting what it can beats saying nothing.

## `__repr__` (`:1858`) — also must not raise

A repr that throws breaks `print(scan)`, Jupyter cell output, pytest assertion
rendering, and the error messages of unrelated failures. Two changes:

```python
wiring = "(" + ", ".join(
    f"{role} <- " + ("grounded" if lbl is None else f"'{lbl}'")
    for role, lbl in self._gates.items()) + ")"
```

Generic over roles, so the *topology* is visible — a missing `top` is what makes it a
single-gated device — and `<- grounded` rather than an invented row name. Undeclared,
it says so and gives the call shape.

```python
if self.sweep_type != "electric_field" and self.is_dual_gated:
    extra.append(("E_F", self.ef, "mV/nm"))
```

`is_dual_gated` (`:1358`) covers both reasons a field is unavailable in one predicate:
undeclared wiring (sign undefined) and single gate (no field at all).

This line is the *discoverability* half of the whole change. The raises catch you when
you ask for something specific; the repr tells you before you ask. Given that the
failure mode is a plot that looks entirely normal, the intervention has to happen
where you are already looking.

---

# Part 7 — HDF5: the laundering problem

`write_sweep` dumps the **resolved** label of every curated entry
(`hdf5.py:287-289`). That dict *always* contains `v_top` and `v_bot`, because
`_CURATED` has defaults for them. **So `curated_labels` alone cannot distinguish a
declared mapping from a defaulted one.**

Without a separate record, CSV (undeclared) -> HDF5 -> reload would come back with
`V_A`->top / `V_B`->bottom looking like a stated fact, and the refusal would never
fire again. Worse than the original default, because now it has a paper trail.

**Write side** (`hdf5.py:296-298`):

```python
if scan.gates is not None:
    meta.attrs["gates"] = json.dumps(scan.gates)
```

Presence is the record of declared-ness; contents are the mapping.

**Read side** — `"gates"` was appended to `_JSON_ATTRS` (`hdf5.py:141`), and that
existing loop decodes each key to a dict or `None`. `_decode_and_describe` then picks
it up via `meta.get("gates")` on the same line pattern as `curated_labels`. No new
read code.

**The asymmetry, explained.** A *caller* passing `curated_labels={"v_top": ...}` is
making a wiring claim through the wrong door and is told. An *HDF5 file's* dump of its
own resolved labels is bookkeeping the writer produced automatically — raising on it
would make every existing archive unreadable. Hence: strip silently on read (`:963`),
raise on the caller (`:979`).

`test_undeclared_wiring_stays_undeclared_on_read` pins this. It is the test least
worth losing, because a regression is completely invisible: everything still works,
it just quietly starts lying again.

---

# Part 8 — the refusal matrix

| Access | undeclared | `{top, bottom}` | `{bottom, channel}` |
|---|---|---|---|
| `scan["V_A"]`, `sweep="V_A"` | ok | ok | ok |
| `gate_mode`, `repr()` | ok | ok | ok |
| `v_bot` | **raise** | ok | ok |
| `v_top` | **raise** | ok | **raise** (no such gate) |
| `v_channel` | **raise** | **raise** (not declared) | ok |
| `ef`, no geometry | `None` | `None` | `None` |
| `ef`, with geometry | **raise** | ok | **raise** (no field) |
| `carrier_density`, no geometry | `None` | `None` | `None` |
| `carrier_density`, with geometry | **raise** | **raise** (no channel) | ok |
| `sweep="top_voltage"` | **raise** | ok | **raise** |
| `sweep="electric_field"` | **raise** | ok (needs geometry) | **raise** |
| `sweep="carrier_density"` | **raise** | **raise** (no channel) | ok (needs geometry) |
| `sweep="power"`, `"piezo_x"`, ... | ok | ok | ok |

Load-time vs access-time: everything in a `sweep=` row raises during construction;
everything in a property row raises on first access.

---

# Part 9 — compatibility

**`AttoCubePLVabScan`** (`:2717`) hardcodes `sweep="electric_field"` or
`"top_voltage"`, both now requiring a declaration. It passes the historical mapping
**explicitly** (`:2776`):

```python
gates = {"top":    top_gate_label if top_gate_label is not None else "V_A",
         "bottom": bot_gate_label if bot_gate_label is not None else "V_B"}
```

Its old `top_gate_label`/`bot_gate_label` now feed `gates` instead of
`curated_labels` — same information, new door.

Is this reintroducing the bug? No. The shim's job is that scripts written against it
keep producing the numbers they always produced; silently changing old results would
be a different and worse failure. Verified against `examples/data/stark-shift/`:
`ef[:3] = [-171.2652, -165.5573, -159.8484]` mV/nm, unchanged, with the transpose
giving the exact negation. What moves people off the assumption is the
`FutureWarning`, which fires on every construction; its docstring now states the
assumption outright so it renders on the docs site.

**`plotting`** — two call sites (`plotting.py:293`, `:405`) previously fell back to
`scan.v_top` and labelled the axis `$V_\mathrm{top}$`. On a rewired device that
caption named the wrong electrode. Both now use `scan.sweep_axis` /
`sweep_axis_label`, which asserts only what the scan was told, and both guard on
`scan.is_dual_gated` **before** reading `scan.ef` — short-circuit order matters, since
reading `ef` without it raises. Otherwise `plot_spectrum` would raise while computing
a default *label*, a baffling failure for a plotting call. This is also the direction
E12 wants plotting to go, so it is not a stopgap.

---

# Part 10 — verification

224 tests pass; `mkdocs build --strict` green. One unrelated pre-existing failure,
`test_contrast.py:251`, from the uncommitted `SIGNAL_LABELS` rename in
`constants.py` (`"Reflectance"` -> `"Reflected intensity"`).

End-to-end against `examples/data/stark-shift/PL-dual-gate-sweep_..._iter_0.csv`:

| Check | Result |
|---|---|
| undeclared + `sweep="electric_field"` | raises, names `V_A` `V_B` `I_B` `I_A` `Excitation Power` |
| `gates={"top": "V_A", "bottom": "V_B"}` | `ef[:3] = [-171.2652, -165.5573, -159.8484]` mV/nm |
| transposed mapping | exact negation (`np.array_equal`) |
| declared as `{bottom, channel}` | `is_dual_gated False`; `v_top`/`ef` raise with the physics sentence |
| `C_bottom`, 46 nm hBN | `7.5068e-4` F/m^2 |
| `dn/dV` | `4.6854e+11` cm^-2 V^-1 |
| 61-point +/-17.283 V sweep | `+/-8.098e+12` cm^-2 |
| `v_ref` shift | same span, offset to `[0, 1.6195e+13]` |
| HDF5 round trip, declared | mapping restored, `ef` matches |
| HDF5 round trip, undeclared | still `None`, `v_top` still raises |
| shim on the same file | `FutureWarning`, old numbers reproduced |

The `4.685e11 cm^-2 V^-1` figure is the useful sanity check — it is the expected order
for ~50 nm hBN-gated TMDCs, and
`test_carrier_density_per_volt_matches_the_capacitance` asserts it to 2% against
`4.7e11` so a unit slip in the `1e-9`/`1e-4` conversions cannot pass silently.

## Tests worth knowing about

Beyond the mechanical `gates=` additions:

- `test_transposing_the_wiring_negates_the_field` — `forward.ef == -swapped.ef`
  exactly. The reason the change exists, as an executable fact.
- `test_ef_refuses_..._only_when_a_geometry_was_given` — pins the `None`-vs-raise
  asymmetry, the nuance a later refactor would flatten to "always raise".
- `test_gate_mode_needs_no_declared_wiring`,
  `test_gate_mode_names_the_channel_when_wiring_is_undeclared` — pin that `gate_mode`
  stays permissive.
- `test_undeclared_wiring_stays_undeclared_on_read` — the laundering path.
- `test_gates_rejects_an_ambiguous_declaration` — parametrised over the five bad
  shapes, including both lone-gate cases.
- `test_gate_capacitance_is_geometric_and_uses_only_that_gates_hbn` — the 5-layer
  invariance.
- `test_carrier_density_warns_when_the_channel_is_driven` — and asserts the
  hard-grounded case stays **silent**, under `simplefilter("error")`.

---

# Part 11 — files touched

| File | What |
|---|---|
| `loaders.py` | the whole mechanism: constants `:620-635`, `gate_capacitance` `:454`, `DeviceGeometry.carrier_density` `:500`, `_resolve_gates` `:1007`, resolution in `_decode_and_describe` `:950-1002`, `_resolve_sweep` gating `:1212-1268`, the property block `:1341-1553`, `gate_mode` `:1707`, `__repr__` `:1858`, shim `:2717` |
| `hdf5.py` | `_JSON_ATTRS` `:141`, conditional `gates` write `:296`, layout diagram, `read_sweep` returns block |
| `constants.py` | `EPS_0`, `E_CHARGE` |
| `plotting.py` | two `is_dual_gated` guards + axis fallbacks (`:293`, `:405`) |
| `__init__.py` | quick-start gains `gates=` |
| `tests/` | `test_loaders.py` (+~250 lines), `test_hdf5_roundtrip.py`, `test_loaders_trpl.py` |
| `README.md` | one paragraph in section 2 flagging the requirement |
| `dev/audit-2026-07.md` | E7b closed, E7c added |
| `.claude/CLAUDE.md` | topology table, settled decisions |

Nothing was deleted except the `v_top`/`v_bot` entries in the shim's `labels` dict and
the `curated_labels` gate-wiring example in the class docstring.

---

# Part 12 — adjacent problems found, not fixed

1. **`electric_field` never checks the device is dual-gated.**
   `DeviceGeometry.from_single("WSe2", d_hbn_top=None, d_hbn_bottom=46)` still returns
   a field, using `d_stack = d_2d + d_hbn_bottom`. Same class of defect as E7b/E7c —
   a plausible number for a quantity the geometry cannot support. The loaders now
   refuse this route via `is_dual_gated`, but `DeviceGeometry` called directly does
   not. Deserves its own entry.

2. **`signal_label` is defined twice.** A stub returning `None` at `:1332` is shadowed
   by a working implementation at `:1637`. Python takes the later one, so nothing is
   broken and the property works — the stub is dead code that should be deleted.

3. **`_validate_axis_and_signals` has an unbound local.** The error at `:1094-1098`
   interpolates `n_sweeps` into its f-string, but `n_sweeps` is not assigned until
   `:1104`. That path raises `UnboundLocalError` instead of the intended message.

4. **README sections 5 and 6** still reference APIs that do not exist
   (`AttoCubePLScan`, `plot_pl_map`, `bg_region=` on `fit_scan_peak`) — pre-existing,
   listed in CLAUDE.md's open issues. Section 2 was corrected only where this change
   made it actively misleading.
