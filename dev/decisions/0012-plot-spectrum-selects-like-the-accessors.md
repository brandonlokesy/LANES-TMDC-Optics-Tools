# 0012 — `plot_spectrum` selects a point by coordinate, keyword-only, through the accessors

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-10 |
| **Extends** | [0004](0004-nested-sweeps-fast-and-slow.md) |

## Context

`plot_spectrum` took a sweep **position** only. Researchers name a setting — 2.5 V,
50 µW — not a column, so every value-based plot meant finding the index first. The
accessors from [0004](0004-nested-sweeps-fast-and-slow.md) already did coordinate lookup,
with settled policies about ambiguity and distance.

## Decision

A coordinate is the primary spelling: `plot_spectrum(scan, value=2.5)`.

**Six selectors in two exclusive spellings** — `value` / `fast` / `slow` are coordinates,
`index` / `index_fast` / `index_slow` are positions. The two never share a keyword, so a
request cannot be half of each. Naming both ways, or neither, raises. `axis=` applies to
coordinates only, and `axis=None` means *the default* rather than reaching the scan, which
would read an undeclared axis as the flat index.

**Every selector is keyword-only, and `value` has no positional slot.** The accessors can
afford `get_spectrum_at(2.5)` because the *method name* says which kind of selection it
is; `plot_spectrum` merges both methods into one function, so that disambiguation has
nowhere to live but the keyword.

**Positions must be whole numbers.** A fractional one is refused rather than truncated.
`np.integer` passes, so an index out of `argmin` is unaffected.

**Selection is not re-implemented here.** It forwards to the accessors' selector, so the
settled policies reach the figure unchanged: an ambiguous coordinate is refused rather
than drawn, and a distant one warns.

## Rejected

**A bare positional number.** On a sweep whose coordinates span the same range as its
positions — a power sweep in µW — `plot_spectrum(scan, 50)` would take 50 µW with no
warning, because 50 is a real coordinate. Keyword-only is what turns a silent misread into
a `TypeError`.

**Truncating a fractional position.** `int(1.9)` plotting point 1 is silent, and a
fractional position is far likelier a coordinate that reached the wrong keyword.

**Composing `nearest_index` at the call site as the documented idiom.** It *warns* where
the accessors *refuse*, so recommending it would route every value-based plot around the
refusal.

**Drawing a free nest axis as N lines.** It selects a spectrum per point, and the return
contract is one artist. Refused instead.

**Forwarding the loader's error messages for a wrong-shape position request.** The scan's
messages are written for its accessors, where `fast=` / `slow=` are whichever spelling
that method takes; here they are always coordinates, so its advice names the wrong
keyword — and following it *succeeds*, selecting a different point in silence. Those two
cases are pre-empted in `plotting`'s own vocabulary.

## Consequences

- The legend names the coordinate addressed — **both** coordinates for a nest, where the
  declared sweep axis is the flat index and says nothing. It reuses the sweep-label
  composition, so existing legends are unchanged and a raw-row axis correctly shows no
  unit.
- The no-point error reports **what arrived**, because with every selector keyword-only a
  renamed or misspelt one is absorbed by the `**line_kwargs` style passthrough instead of
  raising. It is keyed to nothing; it is not a list of old parameter names.
- The function inherits the existing `stacklevel` miscount on this path rather than
  fixing it, so that the existing tests stay honest. Anything routing through the
  accessors' selector needs one more level than it passes — the uncounted frame is a
  lambda, which has no `def` to scroll past.
