# 0032 — Converted files land in a `processed/` folder

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-21 |
| **Audit** | — |

## Context

`converters.py` writes a new file for every one it reads, so it needs a rule for
where that file goes when the caller does not say. The rule is used by every
entry point in the module and by the `tmdc-convert` command, where it is the
whole ergonomics of the tool: a default that matches how the researcher already
keeps files means the common case is typing nothing.

The version on the unmerged `dev/hdf5` branch had a partial rule: a source in a
folder named `raw` wrote to a sibling `processed/`, and anything else wrote
beside the source. So the same command produced two different layouts depending
on a folder name, and the tidy behaviour only appeared for people who already had
the tidy layout.

The maintainer's own measurements are kept as `measurement/raw/` and
`measurement/processed/`. Whether anyone else in the group does is unknown.

## Decision

1. Output **always** goes into a folder named `processed/`.
   - Source in a `raw/` folder → `processed/` is the sibling of `raw/`.
   - Source anywhere else → `processed/` is created beside the source.
2. The two folder names are module constants, `_RAW_DIR` and `_OUT_DIR`. They are
   not parameters, not settings, and not read from the environment.
3. `out=` is the single override, and takes either a directory or a file path. A
   path carrying a suffix is a file; one without is a directory.
4. A directory run resolves the rule **per source folder**, not once for the run.
5. `_default_output` resolves a path and creates nothing. Creating the directory
   and refusing an existing file are `_claim_target`'s, and the existence check
   runs first.

## Rejected

**Write beside the source.** The simplest rule, and no folder to learn. Rejected
because the output of a conversion is derived data and mixing it in with the raw
export is what makes a measurement folder unreadable a year later. It also gives
the tool no way to be run twice without the `.tif` files becoming part of what
the next run globs over.

**Require `out=` every time.** Most predictable, and no hidden behaviour at all.
Rejected because it makes the command line verbose exactly where it is used most
— a bulk conversion over many measurement folders would need a destination per
folder, which is the thing the per-folder rule computes correctly and for free.

**A configuration file or an environment variable for the folder names.** The
obvious way to "encode my workflow". Rejected because it creates a second place
where the answer lives. The first time the configured name disagrees with the
folder on screen, the researcher is debugging their tooling rather than their
measurement, and the failure gives no hint that a setting is involved. Fixed
names can be read straight off the code, and the workflow is encoded by making it
the default rather than by making it configurable.

**A `processed_dir=` parameter.** *Parameters earn their place* asks whether an
argument changes the numbers or only where they land. This one changes neither —
`out=` already reaches every destination it could, and more directly. Two
overrides for one thing is how they drift apart.

**Keep the branch's partial rule.** Rejected because the behaviour it gives a
folder not named `raw` is the "write beside the source" case above, so it carries
that option's cost while also making the tool's behaviour depend on something the
caller did not mention.

## Consequences

- The `raw/` + `processed/` layout is what happens when nothing is specified, so
  it spreads by being the path of least resistance rather than by instruction.
- Someone who does not use that layout still gets a folder rather than files
  scattered among their exports, and the folder tells them what the convention is.
- Two sweeps holding an identically named frame cannot overwrite one another in a
  recursive run, because each folder resolves its own destination. Handing one
  `out` to the whole run — what the branch did — flattened them together
  silently. `tests/test_converters.py::test_recursive_run_keeps_folders_apart`
  pins this.
- A refused conversion leaves no empty `processed/` behind, because nothing is
  created until the target is claimed.
- Any future converter in this module gets the rule by calling `_default_output`,
  rather than deciding again.

## Load-bearing choices

The folder names being constants rather than settings. The argument is that a
second source of truth costs more than it saves *at this scale* — one group,
one convention. If the package were ever installed somewhere with an established
and different layout that its users could not change, the trade would be worth
re-examining, and the place to do it is `_default_output` alone.
