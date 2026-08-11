# 0003 — `gates` declares device topology, and carries the carrier-density path

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-05 |
| **Audit** | E7c |
| **Extends** | [0002](0002-gate-wiring-must-be-declared.md) |

## Context

Raised immediately by [0002](0002-gate-wiring-must-be-declared.md)'s refusal, on a real
device: one electrode drives the bottom gate, the other contacts the TMDC to ground it.
There is no top gate, so **no** `{"top", "bottom"}` assignment was correct — including
the transposition the caller had been worried about.

For that geometry `electric_field` does not merely get a sign wrong; its premise fails
twice. `E_stack = (V_BG − V_TG)/d_tot` needs an equipotential at each end of the stack,
and with no top gate the upper surface has no defined potential — there is nothing to
subtract. And the derivation assumes no free charge between the gates, which is exactly
what grounding the TMDC through a contact introduces: field lines terminate on the
induced sheet charge, so `V_BG` drops across the bottom hBN rather than across `d_tot`
with `ε_stack` in it. Passing the grounded row as `v_top` (it sits at 0, so `v_bot − 0`)
would have returned `V_BG/d_tot · ε_stack/ε_2D` — wrong denominator by ≈2× for a
53/46 nm stack, and a field in a slab that is now the terminating electrode.

Underneath both: one gate is one degree of freedom, so field and density are locked
together. Independent control of the two is what a dual-gate anti-symmetric sweep buys.

## Decision

### 1. The role vocabulary describes topology, not only wiring

Which quantities exist is **derived from which roles are present** rather than flagged
separately.

| Declaration | Device | Available |
|---|---|---|
| `{"top": "V_A", "bottom": "V_B"}` | dual-gated | `ef`, `v_top`, `v_bot` |
| `{"bottom": "V_A", "channel": None}` | bottom-gated, TMDC grounded | `carrier_density`, `v_bot`; `ef` raises |

`"top"` / `"bottom"` are gate electrodes. **`"channel"` is a contact to the TMDC itself
and is not a gate** — it sits inside the stack, carries no thickness, enters no field,
and is excluded from `gate_mode`. A value of `None` means the electrode is tied to
ground with no row recording it, giving zeros; `__repr__` prints `← grounded` rather
than inventing a row name.

`is_dual_gated` is the single predicate for "a field is defined", and is what `plotting`
branches on.

**Validation keeps 0002's no-half-declaration guarantee:** at least one gate is
required, and a lone gate must also name its `"channel"` — otherwise a single-gated
device could not be told from a two-gate device whose second gate was forgotten.
Sweeping an electrode declared as grounded is refused: its voltage is zero at every
point, so it is not an axis. `gate_mode` still never raises, and now reports on
whichever gates it can see, so a partially-missing row degrades to describing the other
gate rather than returning `None`.

### 2. A carrier-density path, geometric only

`gate_capacitance(gate)` is `ε₀ε_hBN/d_hBN`. The TMDC is the **counter-electrode**, not
a slab inside the capacitor, so neither its thickness nor `eps_stack` appears — a test
pins that a 5-layer stack gives the same number. `carrier_density` sums
`C_i(V_i − V_ref)/e` over the gates supplied, signed with electrons positive, in cm⁻².

`sweep="carrier_density"` requires a geometry, a declared `"channel"` (charge comes from
the contact that supplies it), and a computable capacitance for every declared gate.
Its requirement depends on which roles were declared, so it cannot live in the static
sweep-requirements table and is checked explicitly beside the
`electric_field`-needs-a-geometry case.

The property **warns** when the channel's own row varies: the density is referenced to
that contact, so a driven contact moves the reference under the axis. Expected for a
source-drain bias, wrong for a doping sweep, and no file distinguishes them — so it
reports the span it saw instead of choosing.

## Rejected

**Making `ef` work for a single-gated device.** The derivation fails twice there, as
above. Do not reinstate it by passing the grounded contact as the missing gate.

**Treating `"channel"` as a third gate.** It records the contact that grounds the TMDC,
which is what makes a single-gate declaration unambiguous and what a density is
referenced to. It is excluded from `gate_mode` and from every field.

**An absolute-`n` API.** Only the geometric part is free of unrecorded facts. `v_ref` is
a *gate voltage*, not a threshold, so the result is a density **difference**. Absolute
density needs the voltage at which the channel populates — a transfer curve or the PL
charging step — which is in no file. Pass it as `v_ref` if measured; do not default a
guess.

**Reporting the density as exact.** It is geometric only: quantum and interface-trap
capacitance are in series and make the effective value smaller, so it is an **upper
bound**.

## Consequences

- What is computable now follows the declaration, so adding a device topology is a new
  role combination rather than a new flag.
- For 46 nm hBN at ε = 3.9 the capacitance gives 4.685 × 10¹¹ cm⁻² V⁻¹, the right order
  for hBN-gated TMDCs; the committed 61-point ±17.3 V sweep spans ±8.10 × 10¹² cm⁻².
- Verified by running against `examples/data/stark-shift/`.
