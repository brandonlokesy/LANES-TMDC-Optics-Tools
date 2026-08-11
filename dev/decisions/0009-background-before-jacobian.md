# 0009 — Background is subtracted in wavelength space, before the Jacobian

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-04 |
| **Audit** | G5 |

## Context

Converting a spectrum to an energy axis multiplies by `λ²/hc`. A flat pedestal `B`
therefore becomes `B·λ²/hc` — **curved, not flat** — so subtracting a background after
the Jacobian removes a constant from something that is no longer constant.

The loader already did this in the right order. What was missing was any test that would
fail if the two were swapped.

## Decision

Background comes off in wavelength space, before the `λ²/hc` multiply, and the ordering
is **pinned by a test**. Requesting the Jacobian with no background at all warns.

## Rejected

**Leaving the ordering as an implementation detail.** It was already correct, and
therefore silently revertible by anyone reorganising the correction chain.

**A test that only asserts the correct order produces the expected numbers.** Each
ordering case also asserts that the right and wrong orders **differ**, so the comparison
cannot be satisfied by both. The two cases were confirmed to fail against a deliberately
reversed loader before being kept.

## Consequences

- The correction chain's order is now a tested contract rather than a convention.
- This retired the *Jacobian* paragraph from the standing physics conventions, one
  sentence of which had already gone stale — it claimed the docstrings and README
  documented a `True` default, which they no longer did.
