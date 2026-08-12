# 0013 — `plot_power_series` thins by slice step and stacks by an absolute offset

| | |
|---|---|
| **Status** | Accepted · label handling generalised by [0011](0011-label-contract-derive-or-verbatim.md) |
| **Date** | 2026-08-05 |

## Context

A dense power series drawn one line per sweep is unreadable, so two additions were
wanted: draw a subset, and stack the drawn spectra vertically. The function's y-label was
also one of the hardcoded `"PL intensity (counts)"` sites, so a reflectance sweep through
it came out labelled as photoluminescence.

## Decision

| Parameter | Meaning |
|---|---|
| `sweep_step: int = 1` | draw every *n*-th sweep |
| `spectrum_offset: float = 0.0` | stack by a cumulative shift, **absolute**, in the units of the plotted array |
| `ylabel: str = None` | derived from the scan |

## Rejected

**`skip_idx` as the name**, which is what was first proposed. `skip_idx=1` meaning *no
skip* is exactly a NumPy slice step, but the name reads as *"the index of a sweep to
skip"* — one sweep dropped, close to the opposite of the behaviour. Under that name the
docstring has to spend two sentences arguing against the name. `sweep_step` names the
mechanism, inherits the meaning of `1` from a convention every NumPy reader already has,
and matches the existing `sweep_index` in the module.

**A fractional offset** — `0.2` meaning 20% of the plotted data's full span. Easier to
pick blind, because it scales itself to whatever the counts happen to be. Rejected
because the self-scaling reads off the **plotted subset**: the divisor would depend on the
x-range, on the background region, and on `sweep_step`, so the *same scan cropped
differently would stack differently*. A figure whose spacing silently tracks the crop
cannot be reproduced from the call alone.

Absolute is shape-invariant — the same value gives the same figure however the data was
cropped or thinned. The cost is real and is stated rather than engineered away: you have
to read a peak height off an unstacked plot before you can choose a sensible number.

**Keeping the literal y-label.** With a stack applied it prints a unit that is wrong, with
no signal to the reader.

**`None` selecting between two hardcoded PL strings.** Solves the unit and leaves
reflectance scans labelled as PL.

## Consequences

- The draw loop carries **two counters**, and they differ once `sweep_step > 1`: one
  indexes the scan, so colour and alpha keep tracking each line's own power, while the
  other counts drawn lines, so offsets stack contiguously instead of leaving gaps where a
  skipped sweep would have been.
- `ylabel=None` meaning *derive it* was not a new convention — one other plot already did
  it, with no shared helper, so this followed the local pattern rather than adding an
  abstraction. [0011](0011-label-contract-derive-or-verbatim.md) later made it a
  module-wide contract.
