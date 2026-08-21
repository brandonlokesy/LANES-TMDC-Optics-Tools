# 0034 — Converted files land in a `converted/` folder, not `processed/`

| | |
|---|---|
| **Status** | Accepted · supersedes 0032 |
| **Date** | 2026-08-21 |
| **Audit** | — |

## Context

`dev/decisions/0032-converted-files-land-in-a-processed-folder.md` chose
`processed/` as the folder every conversion writes into. The rule it settled — one
unconditional destination folder, fixed name, `out=` as the single override — was
right and is kept. The **name** was wrong.

`processed/` already means something in the group's layout, and it does not mean
this. A measurement folder is kept as:

```
EXP-2026-08-05-PL-benchmarking/
├── figures/
├── processed/      ← what an analysis pulled out of the raw data
└── raw/            ← what the instrument wrote
```

`processed/` is where extracted quantities live: fitted peak positions, integrated
intensities, diffusion lengths — the output of deciding something about the
measurement. A conversion decides nothing. The `.h5` holds the same counts the CSV
held, in the same units, with the same parameter rows; `converters.py` exists
precisely because it does *not* interpret anything.

Writing both into one folder erases a distinction the layout depends on. Given a
file in `processed/`, you could no longer tell whether it is raw data in a better
container — reproducible at any time by re-running the converter, and safe to
delete — or an analysis result that took a decision to produce and may not be
reproducible at all.

## Decision

1. The output folder is named **`converted/`**. `_OUT_DIR = "converted"` in
   `converters.py`.
2. Everything else from 0032 stands unchanged: the folder is unconditional, it is
   the sibling of `raw/` when the source sits in one and is created beside the
   source otherwise, both names are module constants rather than parameters, and
   `out=` is the single override.
3. The folder is created when it does not exist, at every depth. This was already
   true — `_claim_target` and `hdf5.write_sweep` each call
   `mkdir(parents=True, exist_ok=True)` — and is now stated where a caller reads it.
4. The reason for the name is recorded at `_OUT_DIR` itself, because a future
   reader's obvious "improvement" is to unify the two folders.

## Rejected

**Keep `processed/`.** Rejected on the argument above: it collides with an
established meaning, and the collision costs exactly the information the folder
name is there to carry.

**`h5/`, or a name per format.** Would separate `.h5` from `.tif`. Rejected because
the folder answers *how did this file get here*, not *what type is it* — the
suffix already answers the second, and a per-format folder would scatter one
measurement's outputs across two places.

**`raw-converted/`, or a sibling inside `raw/`.** Keeps the association with the
source explicit. Rejected because output inside `raw/` is what the default rule
already avoids: `raw/` is what the instrument wrote, and nothing else should
appear in it.

**Make the name configurable now that there are two plausible ones.** Rejected for
the same reason 0032 rejected it — a setting is a second place the answer lives.
Two candidate names is an argument for choosing correctly once, not for deferring
the choice to every caller.

## Consequences

- `dev/decisions/0032-converted-files-land-in-a-processed-folder.md` is superseded.
  Its rejected alternatives — beside the source, require `out=` every time, a
  configuration file, a `processed_dir=` parameter — all still stand and are still
  the record for them; only the folder's name changed.
- No migration is owed. The package is pre-adoption and nothing has been converted
  in anger yet.
- `processed/` and `converted/` now sit side by side in a measurement folder, and
  which one a file is in says whether it is reproducible by re-running a converter.
- The distinction is worth holding to elsewhere: anything this package writes that
  *is* an analysis result should not go to `converted/`.

## Load-bearing choices

That conversion and analysis are different kinds of output. If `converters.py` ever
grew a step that decided something about the data — a correction, a fit, a derived
axis — it would no longer belong in `converted/`, and the right response would be
to move that step out of the module rather than to rename the folder.
