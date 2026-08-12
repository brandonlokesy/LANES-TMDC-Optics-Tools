# 0007 — One loader class reads both formats, and HDF5 stores no derived arrays

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-07-30 |
| **Audit** | G2 |

## Context

CSV exports are large and slow to read, so an archive format was wanted. The two
questions it raised: does HDF5 input get its own loader class, and does the archive hold
the corrected arrays a session produced?

## Decision

`scan.to_hdf5(path)` writes; the loader accepts `.h5` / `.hdf5` and dispatches on
suffix, so **one class serves both formats**. Both paths produce the same decoder
payload, and nothing after that point can tell which ran.

The file stores raw signal arrays (both ROIs for a spectral sweep), every parameter row
verbatim, and the measurement metadata.

It does **not** store the energy axis, the energy-space spectra, or the sweep axis — all
derivable. Corrections (`apply_jacobian`, the background regions, the cosmic-ray
declaration, and the `bg_spectrum` / `reference` arrays) are recorded as **provenance**
and are **never replayed on read**. Loading is not deciding.

The auxiliary spectra are stored as *arrays, not paths*, so a contrast can still be
rebuilt from the archive alone once the original CSVs have moved.

## Rejected

**A second loader class for HDF5.** It would double the API for one file format.

**Storing the derived arrays.** Freezing them would put one session's corrections into
the archive, where the next reader inherits them without having chosen them — and would
contradict *corrections are opt-in* one layer down.

**Replaying the recorded corrections on read.** Same reason. The declaration is
provenance so a reader knows what was done; re-applying it would make loading a decision.

**Storing paths to the reference and background spectra.** They would break as soon as
the source files moved, which is the situation an archive exists for.

## Consequences

- An archived, corrected scan **reloads uncorrected**. The declaration is visible in the
  metadata, so the caller can restate it, but a round trip does not carry it.
- The format version bumps additively, so older files keep reading.
- Sizes from the committed examples: a 4.59 MB PL CSV writes as 0.14 MB; the 4-file
  11.57 MB TRPL sweep as 0.069 MB.
- Because the wiring declaration cannot be inferred from the blanket curated-label dump,
  it is stored separately — see [0002](0002-gate-wiring-must-be-declared.md).
