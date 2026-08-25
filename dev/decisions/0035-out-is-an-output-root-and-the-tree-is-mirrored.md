# 0034 — `out=` is an output root, and the tree is mirrored beneath it

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-21 |
| **Audit** | — |

## Context

`dev/decisions/0034-converted-files-land-in-a-converted-folder.md` settled where
output goes when the caller says nothing: a `converted/` folder, resolved per
source folder. It left `out=` as "the single override" without saying what an
override means for a **tree**, because at that point `convert_path` only walked one
level in practice.

A real measurement tree made the gap concrete:

```
EXP-2026-08-05-PL-benchmarking/
├── converted/                     ← where converted data should go
├── processed/                     ← analysis output; not this module's business
└── raw/
    ├── spot01/{01-PL-Vbot-sweep, 02-PL-power-sweep, 03-R-Vbot-sweep, ref}/
    ├── spot02/…
    └── … seven spots
```

Neither behaviour available served it:

- **Omitting `out=`** created `raw/spot01/01-PL-Vbot-sweep/converted/` — a folder
  nested inside `raw/`, once per leaf, when the experiment keeps its converted
  data in one place at the top.
- **Passing `out=EXP-…/converted`** handed the *same* directory to every file, so
  the whole tree flattened into one folder. Measured on a replica: `spot02/ref` and
  `spot03/ref` both hold `laser_ref_26_08_07_10_35_44.csv`, and the run reported
  `9 file(s) written, 1 error(s)` — the second was refused by the overwrite guard
  rather than lost, but the tree could not be converted in one command.

## Decision

1. For a directory run, `out=` is an output **root**. Each source folder's path
   *relative to the directory that was named* is appended to it, so
   `<path>/spot01/01-PL/sweep.csv` becomes `<out>/spot01/01-PL/sweep.h5`.
2. No `converted/` level is added under `out=`. The root the caller named already
   is that folder.
3. Mirroring lives entirely in `convert_path`, which resolves a destination per
   folder and hands it down as an ordinary `out=`. The single-file converters keep
   their existing contract — `out=` is a directory or a filename — and are
   unchanged.
4. A folder at the top of the walk has a relative path of `.`, which `joinpath`
   drops. So a non-recursive run addresses `out=` directly and behaves exactly as
   before; the mirror only appears where there is a tree to mirror.
5. An `out=` carrying a suffix is still one filename, used verbatim. That is
   meaningful with `stack=True`, which is one output for a whole folder.

## Rejected

**Flatten, as before.** Rejected because it cannot express the layout above, and
because two folders holding an identically named reference frame is not an edge
case — it is what happens when the same alignment reference is recorded at each
spot. The overwrite guard turns that into an error rather than data loss, so the
old behaviour was safe but unusable.

**A separate `--mirror` flag alongside `--out`.** Keeps both behaviours. Rejected
because the two differ only when recursing, and flattening while recursing has no
use that survives the collision it causes. A flag whose off position is a footgun
is not a choice worth offering, and *parameters earn their place*.

**Mirror from the filesystem root, or from the nearest `raw/`.** Would let
`out=` be given once for several unrelated inputs. Rejected as unpredictable: the
depth of the mirrored tree would depend on where the source happens to sit, and
anchoring on `raw/` would silently do something different for a tree that does not
use that name. Anchoring on the directory the caller named is the one reference
point the caller can see.

**Add a `converted/` level under `out=`.** Symmetry with the default rule.
Rejected because it would produce `EXP/converted/converted/spot01/…` for the
obvious invocation, and the caller naming a destination has already said where
output goes.

## Consequences

- The tree above converts in one command:
  `tmdc-convert …/raw --recursive --spectra-type PL --out …/converted` writes
  `converted/spot01/01-PL-Vbot/sweep.h5`, `converted/spot02/ref/laser_ref.tif`,
  and so on. Verified on a replica: 10 files, 0 errors, both `laser_ref` files
  kept.
- **Behaviour change:** a recursive run with `out=` no longer flattens. Nothing
  depended on that, since it collided; the package is pre-adoption and carries no
  shim.
- `out=` now has one meaning across the default and the override: a source's
  position is preserved either way. That is what makes the identically-named-frame
  guarantee in 0032 hold for a tree and not only for a folder.
- `--spectra-type` remains one value per run, so a tree mixing PL and reflectance
  sweeps still needs one pass per measurement type. Mirroring does not address
  that and is not meant to.

## Load-bearing choices

Anchoring the relative path on the named directory. It is what makes the result
predictable from the command alone, and it is why the non-recursive case needs no
special handling. Any future "convert several roots in one call" would have to
decide this again, and should not reach for the filesystem to do it.
