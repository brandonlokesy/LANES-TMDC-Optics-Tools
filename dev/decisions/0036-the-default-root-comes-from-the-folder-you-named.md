# 0036 — The default output root comes from the folder you named, and `from_raw=` opts into searching

| | |
|---|---|
| **Status** | Accepted · extends 0034 §2, reuses 0035's anchor |
| **Date** | 2026-08-21 |
| **Audit** | — |

## Context

`dev/decisions/0034-converted-files-land-in-a-converted-folder.md` rejected putting
output inside `raw/`, on the grounds that *"`raw/` is what the instrument wrote, and
nothing else should appear in it."* Its rule did not achieve that.

`_default_output` reads the **immediate parent** of each file. A measurement tree
does not put its exports one level under `raw/`:

```
EXP/raw/spot01/01-PL-Vbot-sweep/PL_10uW_…_iter_0.csv
```

The parent is `01-PL-Vbot-sweep`, not `raw`, so the rule fell through to "a
`converted/` folder beside the source" — *inside* `raw/`. Measured on a replica of
a real benchmarking run, one recursive pass scattered eight of them:

```
raw\spot01\01-PL-Vbot\converted     raw\spot01\ref\converted
raw\spot01\02-PL-power\converted    raw\spot02\01-PL-Vbot\converted
raw\spot01\03-R-Vbot\converted      …
```

`out=` already fixed this, but only if the caller remembered it every time. The
default contradicted the principle the folder name exists to express.

Two anchors were available for a better default. **The folder named in the call**,
which is what `dev/decisions/0035-out-is-an-output-root-and-the-tree-is-mirrored.md`
already uses for `out=` and which is visible in the command. Or **the nearest
ancestor called `raw`**, found by searching upward, which places output correctly
even when the call points inside `raw/` — but which 0035 rejected for `out=` on the
grounds that it reads folders the caller did not name.

## Decision

1. **Default:** when `out=` is omitted, `convert_path` derives a root from the
   folder the call names. If that folder is called `raw` (case-insensitively), the
   root is its sibling `converted/` and the tree is mirrored beneath it, by 0035's
   existing machinery. Otherwise there is no root and each source folder keeps its
   own `converted/`, exactly as before.
2. **`from_raw=True`** searches upward from the named path for the nearest folder
   called `raw` and uses its sibling `converted/` as the root. This covers the two
   cases the default deliberately does not: a call naming one measurement folder
   inside `raw/`, and a call naming a single file.
3. `out=` and `from_raw=` together **raise `ValueError`**. Both answer where output
   goes, and `out=` already mirrors beneath the folder named.
4. `from_raw=True` with no `raw` found **warns and falls back** to the default. It
   is an instruction that could not be carried out, not a failure of the run.
5. Root and anchor are resolved **once** per call, before the walk, in
   `convert_path`. `_default_output` and the single-file converters are unchanged:
   called directly they still read the immediate parent only.

Where each case lands, all verified end to end:

| Call | Default | `from_raw=True` |
|---|---|---|
| `convert_path("EXP/raw", recursive=True)` | `EXP/converted/spot01/01-PL/…` | same |
| `convert_path("EXP/raw/spot01/01-PL")` | `EXP/raw/spot01/01-PL/converted/…` | `EXP/converted/spot01/01-PL/…` |
| `convert_path("EXP/raw/…/sweep.csv")` | `EXP/raw/…/converted/sweep.h5` | `EXP/converted/spot01/01-PL/sweep.h5` |
| no `raw` anywhere | `<folder>/converted/…` | same, with a warning |

## Rejected

**Search upward by default.** Fixes every case with no flag, which is its whole
appeal. Rejected because the destination stops being readable from the call: a
folder called `raw` anywhere above — `X:\Brandon\raw\01_Projects\…` — would anchor
everything beneath it, producing a mirror as deep as the path happens to be, and
nothing in the command would hint at it. That is the objection 0035 raised, and it
is not weakened by moving to the default path. Available as `from_raw=` for a
caller who has looked at their path.

**Leave the default alone and rely on `out=`.** No behaviour change, and `out=`
already works. Rejected because the failure is silent and the default is what an
unattended or first-time run uses: the files convert, nothing warns, and eight
`converted/` folders appear inside `raw/`. A default that quietly violates the
package's own stated rule is worth changing even though a workaround exists.

**Refuse when output would land inside `raw/`.** Strictest reading of 0034 —
never write there, make the caller pass `out=`. Rejected because it turns a working
call into an error for someone who does not care where the file goes, and because
the same rule would have to fire for a single-file conversion, which is the most
casual use there is.

**Drop the `raw/` special case entirely**, so output is always beside the source
and `out=` is the only placement tool. Simplest rule to state. Rejected because
`raw/` + `converted/` is the layout this package is trying to make ordinary, and
the default is the strongest way to spread it.

## Consequences

- `tmdc-convert EXP/raw --recursive --spectra-type PL` now places the whole
  experiment under `EXP/converted`, mirroring the spot and measurement folders,
  with no arguments about paths at all.
- **Behaviour change:** a recursive run with no `out=` that names a `raw` folder
  used to scatter `converted/` folders inside it and now does not. Nothing
  depended on the old placement.
- Naming a folder *inside* `raw/` still writes inside `raw/`. That is the price of
  a readable default, and `--from-raw` is the documented answer. It is stated in
  the guide rather than left to be discovered.
- `convert_image_csv_to_tiff` and friends called directly are unaffected: one file,
  immediate parent, `converted/` beside it. Only `convert_path` grew a layer.

## Load-bearing choices

That a default should be predictable from the call. Both this decision and 0035
rest on it, which is why the searching version is a flag rather than the default in
each. If that principle were ever traded away, the two should be revisited together
rather than one at a time.
