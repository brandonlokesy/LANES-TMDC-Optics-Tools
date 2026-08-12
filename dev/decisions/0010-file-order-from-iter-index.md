# 0010 — Directory order comes from the `_iter_N` integer, and anomalies warn rather than repair

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-06 |
| **Audit** | A7, A12 |

## Context

Two loaders read a directory of per-point files: the TRPL sweep and the real-space frame
sequence. Both originally ordered them with `sorted()` on filenames.

Exports *are* zero-padded, so that looks safe — but the **padding width varies between
exports**, so alphabetical order is right only by luck. Where it is wrong, frames pair
with the wrong parameter snapshot and nothing announces it.

## Decision

One module-level helper, shared by both directory loaders, ordering files by the
**integer** in the `_iter_N` suffix.

Three conditions warn, and **none is repaired**:

| Condition | Warning |
|---|---|
| a file with no `_iter_N` suffix | reported, not silently sorted in |
| a gap in the sequence | reported, never closed up |
| one index claimed by more than one file | reported, **naming the colliding files** |

The duplicate message names the files because two acquisitions sharing a directory is the
usual cause and a narrower prefix the usual fix.

`stacklevel` is a **required** argument of the helper, because its two callers sit at
different depths.

## Rejected

**`sorted()` on filenames.** Right only when the padding width happens to be constant,
and silent when it is not.

**Closing up a gap.** Renumbering the remaining files to be contiguous would silently
restore exactly the mispairing this helper exists to catch.

**Resolving a duplicate by picking a winner** — newest mtime, longest name, first
alphabetically. Same objection: it produces a plausible-looking sweep built on a guess.
Duplicates are reachable with every file legitimate, so this is not a
malformed-input case with an obviously discardable side.

**A default `stacklevel`.** The two call sites are at different depths, so any single
default is wrong for one of them, and a wrong `stacklevel` blames library code for the
researcher's line.

## Consequences

- A directory with anomalous naming still loads, in the order the integers give, with the
  anomaly reported.
- The warnings are diagnostics about the acquisition, not about the code, so the fix is
  usually to split a directory or narrow a prefix.
