# 0037 — `beside=` is shorthand for one `out=`, not a third placement rule

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-22 |
| **Audit** | — |

## Context

For a targeted conversion — one file you want an archive of, sitting next to the
CSV — the `converted/` folder is in the way. The committed `examples/data` archives
are exactly this shape: `PL-dual-gate-sweep_26_05_15_14_03_18_iter_0.h5` sits beside
its `.csv`, not in a subfolder.

This was **already possible**, and that is the argument this record exists to
settle. Verified before writing any code:

```
tmdc-convert <dir>/sweep.csv --spectra-type PL --out <dir>
  -> <dir>/sweep.h5                       # no converted/ level

tmdc-convert <raw> --recursive --out <raw>
  -> every output beside its own source   # anchor == root, so relative paths are "."
```

So a new argument buys no capability. `dev/design-principles.md` asks whether an
argument does something the caller cannot readily supply themselves, and three
records in a row have refused second spellings on exactly this basis: a
configuration file for the folder names (0032, 0034), a `--mirror` flag (0035), a
`processed_dir=` parameter (0032).

What survives that test is a narrower claim. `out=` takes a **path**, and a
mistyped path does not fail — it is still a valid destination, so the output
silently lands somewhere else. For a one-off conversion the caller is retyping a
path they have already typed once in the same command. A boolean cannot be
mistyped into a wrong-but-valid destination.

## Decision

1. `convert_path(beside=True)` / `--beside` writes each output into the folder its
   source came from, with no `converted/` level.
2. It is implemented as **root and anchor being the same folder**, so every
   relative path is `.` and the existing machinery from 0035 does the work. No new
   placement concept is introduced, and no code path is added below
   `convert_path`.
3. It is defined as, and documented as, *identical to `out=` naming that folder*.
   A test asserts the two produce the same output path, so a divergence is a
   failure rather than a discovery.
4. At most one of `out=`, `from_raw=` and `beside=`. The refusal names which ones
   were given.
5. It does **not** warn when it writes inside `raw/`. 0034 says nothing but
   instrument output belongs there, but that governs the default; an explicit
   argument is the researcher deciding, which is the package's stated position.
6. It lives on `convert_path` only. The single-file converters already take
   `out=`, and `out=path.parent` from Python is not worth a second spelling.

## Rejected

**Do not add it; document `--out <folder>`.** The position this record argued from,
and the right default answer for a request like this. Rejected on the typo
argument alone: the flag's whole value is that it removes a path from a command
whose destination is already determined, and the failure it prevents is silent.

**Make it a placement rule of its own** — its own branch in `_default_output`, or a
`no_subfolder=` on every converter. Rejected because it would then be a second
thing that can drift from `out=`, which is the cost the design has been avoiding.
Shorthand that is literally implemented as the long form cannot drift.

**Warn when it writes inside `raw/`.** Protects 0034's principle. Rejected as
patronising: the caller named the flag, and warning about an explicit instruction
trains people to ignore warnings.

**Name it `--in-place`.** Rejected as misleading — it suggests the source file is
modified. `--beside` matches the vocabulary already used for where output goes, and
the neighbouring prose was reworded from "a folder beside the source" to "one
inside the source's own folder" so the two senses do not collide.

## Consequences

- `tmdc-convert examples/data/stark-shift/sweep.csv --spectra-type PL --beside`
  reproduces the committed archive layout in one command.
- Over a folder or a tree it puts every output with its own source and creates no
  `converted/` anywhere, which is a reasonable way to convert a directory in place.
- The mutual-exclusion guard now covers three arguments rather than two, and its
  message lists the offenders. An older test pinning the two-argument wording was
  updated in the same change.

## Load-bearing choices

That it stays *definitionally* equal to an `out=`. The test asserting the two agree
is what keeps this a convenience rather than a second answer to "where does output
go". If a future change gives `beside=` a behaviour `out=` cannot express, it has
stopped being shorthand and needs its own argument made.
