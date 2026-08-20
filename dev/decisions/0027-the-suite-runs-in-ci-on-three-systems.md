# 0027 — The test suite runs in CI on three operating systems, and a red run blocks a merge

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-20 |
| **Audit** | F1 |

## Context

Tests were local-only and `pytest` was undeclared in `pyproject.toml`, so the dependency
existed in exactly one environment. E4 records what that cost: the item drifted twice in
two days — present, absent, then present again after a manual install — because an
undeclared test dependency has no way to announce itself when an environment is rebuilt.

CI existed, and built and deployed documentation only. Nothing had ever installed this
package from clean, on any machine but the maintainer's, and the maintainer's is conda on
Windows. That left three questions unanswerable rather than merely unanswered:

- **Whether the package installs at all from a plain `pip install`.** `dependencies`
  lists `ffmpeg`, which E5 identifies as an unmaintained PyPI wrapper rather than the
  binary matplotlib wants. Whether it even resolves was untested.
- **Whether the tests' repo-relative data paths survive a case-sensitive filesystem.**
  Eight test files open committed data by paths like `examples/data/Raman/map2.txt`.
  Windows and macOS match filenames case-insensitively; Linux does not. A wrong capital
  would have been invisible locally and fatal elsewhere. Checked by hand against the git
  index before the first run: all nine referenced paths match exactly.
- **Whether `requires-python = ">=3.9"` is honest.** `constants.py` and `__init__.py`
  carry no `from __future__ import annotations`, so 3.9 is plausible and unverified.

The forcing constraint is adoption. The package is becoming the group's standard analysis
workflow, the group runs Windows with some macOS, and "works on my machine" stops being a
private problem the moment someone else commits. `origin` already required pull requests
into `main`, but with nothing gating them, so the rule governed the route and not the
content.

## Decision

1. `.github/workflows/tests.yml` runs the suite on every pull request and every push to
   `main`.
2. Python 3.12 only.
3. Three runners — `ubuntu-latest`, `windows-latest`, `macos-latest` — with
   `fail-fast: false` and `timeout-minutes: 30`.
4. `pytest` is declared as a `test` extra. CI installs `".[test,colormaps]"`.
5. The three resulting checks are **required** on `main`. `strict` — branches must be up
   to date before merging — is off. The existing admin bypass is kept.
6. Both workflows pin actions to their current majors, which are the Node 24 releases.

## Rejected

**A 3.9 / 3.12 matrix.** Two lines of YAML, and it would have answered the
`requires-python` question outright. Rejected because numpy and scikit-image no longer
publish wheels for 3.9, so pip resolves years-old versions of both and a red run reports
the state of the ecosystem rather than the state of this package — the worst possible
first experience of CI for someone learning to read a failing run. The question is
**deferred, not settled**: a 3.9 job may be added deliberately, as its own experiment,
and this record does not forbid it.

**Installing only `".[test]"`.** The minimal extra, and it has a real virtue: it proves
the package works with nothing but its required dependencies, which is what a group
member gets from a plain `pip install`. Rejected because it silently skipped eight tests.
`tests/test_plotting_cmap.py` gates five tests on `cmcrameri` and three on `cmocean`
through `pytest.mark.skipif`, and those are exactly the tests that check third-party
colormaps register under the `cmc.` and `cmo.` prefixes and do not shadow matplotlib's
own `berlin`, `managua` and `vanimo`. Under `".[test]"` that behaviour was covered on no
machine anywhere. The virtue is also mostly retained without it: `test_defaults_need_no_optional_package`
checks the no-extras promise directly and runs either way.

**Ubuntu only.** Fastest and cheapest, and it matches what `docs.yml` already used.
Rejected because it exercises nothing anyone in the group installs on.

**Windows and macOS only.** What was first asked for, and it covers the group exactly.
Rejected because Linux is the only one of the three with a case-sensitive filesystem, so
it is what checks those repo-relative data paths on every run rather than once by hand.
It is also the cheapest of the three, so the coverage is close to free.

**`fail-fast: true`, the default.** Rejected because the first machine to fail cancels
the other two, so a Windows-only problem hides whatever macOS was about to report —
which defeats the only reason a matrix exists. Three independent answers per run is the
point.

**`strict` required status checks.** Correct in principle: it guarantees the tests that
passed ran against the combination about to land, not against a stale branch. Rejected
for now because with one committer it guards against a collision that does not occur, at
the cost of a re-push and a re-run every time `main` moves. Worth turning on when several
branches are open at once.

**Removing the admin bypass.** Would make the gate absolute, including for the
maintainer. Rejected because the bypass is the way back in if `main` breaks badly enough
that the normal route is blocked. The rule still does its substantive work — it governs
everyone else's changes, and it converts an unnoticed red into a deliberate override.

**Bumping only to the minimum Node 24 majors.** `actions/checkout@v5` and
`actions/setup-python@v6` are the first releases on Node 24, so they clear the
deprecation with the smallest possible move. Rejected in favour of the current majors —
`checkout@v7`, `setup-python@v7`, and in `docs.yml` `upload-pages-artifact@v5` with
`deploy-pages@v5` — because the smallest move buys another bump in a few months. Nothing
in the intervening changelogs touches this use: `checkout` v6 changed where it stores
credentials and v7 blocks checking out forked pull requests for `pull_request_target`,
which neither workflow uses; `setup-python` v7 removed a `pip-install` input, also unused.

**Fixing the tests' data paths first.** The paths depend on pytest's working directory
rather than on where the test file sits, so `cd tests && pytest` fails. Rejected as the
opening move because it touches eight test files, and a red first CI run would then have
had two candidate causes. Doing it second is strictly safer: CI on three systems is the
thing that checks the refactor.

## Consequences

- `pip install -e ".[test]"` provisions pytest in any environment. That closes the signal
  failure E4 described; the dependency can now announce itself.
- CI is no longer documentation-only. Two workflows, and a merge to `main` runs both.
- 862 tests run on each of three machines: roughly 1 min on Linux and macOS, 2 min on
  Windows, against about 175 s locally.
- **Required checks are matched by name** — `pytest (ubuntu-latest)`,
  `pytest (windows-latest)`, `pytest (macos-latest)`. Renaming a matrix entry, or adding
  a `paths:` filter so the job skips for some changes, leaves every pull request waiting
  on a check that can never report. The ruleset must be updated in the same change.
  `.claude/CLAUDE.md` carries this as a rule.
- **Every run downloads the committed example data.** `examples/data` is 573 MB in the
  working tree; the eight test files above need five of its folders. The repository is
  public, so runner minutes are free, and this is a wall-clock cost rather than a money
  one.
- **CI runs newer libraries than the conda environment** — matplotlib 3.11.1 against
  3.10.9 at the time of writing. That is a feature, and it immediately surfaced F2: 88
  deprecation warnings per run from inside `cmocean`, invisible locally because the older
  matplotlib does not emit them.
- **E5 is not settled and CI cannot settle it.** `ffmpeg` 1.4 resolves and installs
  cleanly on all three systems. A clean-room install has no way to tell that it is the
  wrong package, so E5 stands as written and needs a human decision.
- **`requires-python` remains unverified.** Both workflows pin 3.12 and nothing has ever
  run 3.9.
- **A change to `docs.yml` still cannot be validated before merging.** It triggers only
  on pushes to `main`, so a pull request touching it reports the three pytest checks,
  which know nothing about it. A failed docs build is not destructive — `deploy` has
  `needs: build`, and Pages keeps serving the last successful deployment — but the gap is
  real and open. Giving the build job a `pull_request` trigger while keeping `deploy` on
  `main` would close it.
- The tests' data paths stay working-directory dependent. A `working-directory:` key
  added to a workflow step would break them; every step starts in the checkout root by
  default, which is why they work.

## Load-bearing choices

**Python 3.12 only.** If a group member reports a failure on an older interpreter, the
matrix is the first thing to revisit, and the 3.9 question above is the one to reopen.

**Installing `colormaps` in CI.** This trades away the property that CI proves the
package works on required dependencies alone. `test_defaults_need_no_optional_package` is
now the only thing carrying that promise; if it is ever weakened or removed, this trade
should be re-examined rather than left implicit.

**Keeping the admin bypass.** The gate is advisory for one person and binding for
everyone else. That is deliberate while the maintainer is sole committer, and it is the
first thing to change when they are not.
