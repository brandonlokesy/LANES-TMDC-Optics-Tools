# 0011 — Axis and colour-bar labels: `None` derives, a string is verbatim, nothing is appended

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-06 |
| **Audit** | E12 (label half) |

## Context

`"PL intensity"` was hardcoded in roughly six places across `plotting` — a map's colour
bar, two spectrum plots, a power series' y-label default, a panel's y-label default. A
reflectance sweep was therefore labelled `"PL intensity (counts)"`.

The scan already knew better: a signal label derived from `spectra_type`, and a contrast
label for `ΔR/R₀`, both unused.

There was also a live composition defect. Where a label *was* built, it was built by
appending — and `"PL intensity (norm.) (norm.)"` had shipped.

## Decision

**One contract, everywhere in `plotting`:**

- `None` derives the label from the scan.
- A string is used **verbatim**.
- **Nothing is ever appended to a caller's string.**

Derivation reads the signal name, signal unit and contrast label through one helper, so
it follows `spectra_type` and no plot hardcodes a measurement. A `normalized` flag
**substitutes** the unit rather than adding one — a ratio such as `ΔR/R₀` already reads as
normalised, so `$\Delta R/R_0$ (norm.)` states it twice.

A label is **semantics, not styling**: it states what the numbers *are*, so a wrong one is
a misread rather than an ugly figure. That is why it earns a parameter at all, when
`artist.set_<thing>(value)` styling does not.

## Rejected

**Composing `caller_string + " (counts)"`.** It forces a "pass it without a unit"
convention that is undocumentable at the call site, and it is what produced
`"(norm.) (norm.)"`. The defect came from the append, not from the parameter.

**Deleting `colorbar_label` as a styling parameter**, which was the original plan for this
pass. Reversed: it is a semantic label, and the generalised contract above is what it
needed instead.

**A loader-level override on the signal side.** `spectra_type` is a closed, validated
vocabulary, so the derived signal label has one correct value per measurement. The *sweep*
side does get overrides, because a raw row's meaning is not in any vocabulary — declared
once through `sweep=` / `sweep_label=` / `sweep_unit=`.

## Consequences

- A reflectance or contrast plot labels itself correctly with no argument.
- Two defects fixed in passing, neither previously recorded:
  - **`plot_image` discarded `colorbar_label` whenever `rescale_img=True`**, always
    overwriting it with a hardcoded `"Intensity (norm.)"`. Silent — the caller passed a
    label and got a different one.
  - **A second, dead signal-label property** whose whole body was a bare `return`, sitting
    ~300 lines above the real one. Later definition wins in a class body, so nothing
    misbehaved, but any reordering would have turned every signal label into `None`.
    Deleted.
- One correction had shipped without its test, leaving a contrast test failing on `main`;
  updated with a comment on why the wording is not that test's subject.
- The renames that make up the rest of E12 are still owed, and `plot_current` goes first —
  see `dev/plan-E12.md`.
