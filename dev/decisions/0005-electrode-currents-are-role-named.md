# 0005 — Electrode currents are role-named and resolved from `gates=`

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-07 |
| **Audit** | E15 |
| **Extends** | [0002](0002-gate-wiring-must-be-declared.md), [0003](0003-gates-declares-device-topology.md) |

## Context

The curated registry held `"Ich1": ("I_A", …)` and `"Ich2": ("I_B", …)` — three faults
in one entry: the only camel-case keys in a registry of snake_case attribute names; a
`1`/`2` indexing that the file (`I_A` / `I_B`) does not use; and a **channel** named in
the one place everything else names a **role**.

Channels A and B are source-meter channels: `V_A` is the bias applied and `I_A` the
current sourced at the *same terminal*. So `I_A` belongs to whichever electrode channel A
reached — which is exactly what `gates=` records and nothing else does. Channel-named
keys hid that. The two currents got one blanket description ("leakage currents") when on
a contacted device they are two different quantities: leakage across a dielectric on a
gate, transport into the flake on the channel contact. Reaching the bottom gate's
leakage meant knowing both that `1↔A` and that `A↔bottom`, and only the second was
written down anywhere.

`examples/old_example_stark_shift.ipynb` shows the cost directly:
`ich2_label="I_A"`, `ich1_label="I_B"` — someone using label overrides to repair a
mapping the index names got wrong.

## Decision

`i_top` / `i_bot` / `i_channel`, resolved from `gates=` exactly as `v_top` / `v_bot`
are, via a channel-sibling table recording that `V_A`↔`I_A` and `V_B`↔`I_B` are one
terminal each.

The division of labour: the **sibling table is a format fact** — fixed, verified across
PL/R/TRPL — while **`gates=` keeps naming rows**, which is the per-session fact.
`Ich1` / `Ich2` are deleted, along with the `ich1_label` / `ich2_label` shim parameters
that could no longer reach anything.

**Two deliberate asymmetries.** The current registry covers all three roles where the
gate registry covers two — a current flows at the channel contact just as at a gate;
only the *field* is restricted to the two gates. And a role declared `None` gives
`v_channel` zeros but makes `i_channel` **raise**: grounding an electrode is what holds
its potential at zero, but it says nothing about the current, which still flows and
simply was not recorded. Returning zeros there would fabricate a measurement.

## Rejected

**`gates={"bottom": "A"}`** — declaring channel letters and deriving both rows. The
tempting spelling, and the one first proposed. Against it: *"row `V_A` and row `I_A` are
the same terminal"* is a fact about the **export format**, whereas *"channel A reached
the bottom gate"* is a fact about the **session**. Different lifetimes and different
sources of truth, so they belong in different places — fusing them would mean a row
rename in a future acquisition version reaches into every notebook. Channel letters
would also have moved `gates` into channel-space while `sweep=`, `scan[...]`,
`varying_parameters()` and `sweep_grid()` all stay in row-space, including the
undeclared-gates error, which prints candidate *rows* and would then have asked for a
*letter*. And `gates` accepts any row, which letters could not express.

Accepted cost: `{"bottom": "V_A"}` implies `I_A` without saying so, which the `gates`
docstring states.

**Keeping a channel-level `i_a` / `i_b` pair.** Which electrode `I_A` flows into is
precisely the undeclared fact, so the currents refuse without `gates=` just as the
voltages do. `scan["I_A"]` still works with no declaration.

**A bare rename to `i_a` / `i_b`.** It differs from the row label `I_A` **only by case** —
too weak a place to put the boundary between the two vocabularies, and the confusion
that opened this thread was a row label passed where an attribute name was wanted.

## Consequences

- **`plot_current` now requires `gates=`.** It plots whichever roles the scan declared
  and have a recorded current, so a dual-gated device shows top+bottom and a contacted
  one bottom+channel. Its `color_ich1` / `color_ich2` parameters went with the rename —
  the roles present vary, so two enumerated colours no longer map onto the traces — and
  it returns its lines instead.
- **Found in passing, not fixed:** `curated_scales` *replaces* a scale rather than
  multiplying it, so flipping a current's polarity is `{"i_bot": -1e9}` and
  `{"i_bot": -1.0}` silently returns amps. Harmless for the voltages, whose scale is
  `1.0`.
- Tested for currents transposing with the wiring, the channel contact's own row, the
  grounded-electrode raise, the non-source-meter-row raise, and the scale override.
