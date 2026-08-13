# 0015 — A reversed coordinate window is refused, not reversed for you

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-13 |
| **Audit** | — (completes the coordinate selection `0014` left owed) |

## Context

`0014` made an animation take frame *indices*, and left selecting them by physical
coordinate as owed work. `plotting.frame_window(scan, start, end, axis=)` is that piece: it
resolves two coordinates to a range of frames, so a researcher can say "the interval between
these two positions" without converting to indices by hand.

That raises a case with no obvious answer. What should `frame_window(scan, 4.8, 3.2)` do,
when the second coordinate lands before the first?

The original PR #16, which resolved a pair of positions to a pair of indices, **silently
swapped them**. It is the sort of convenience that looks harmless and removes the only
evidence that something was mistyped.

## Decision

Refuse. `frame_window` raises when *end* resolves before *start*, naming both frames and the
idiom for reverse playback.

The refusal applies **only to `frame_window`**, not to `frames=` on `animate_panels`, which
continues to accept any sequence in any order — including descending and arbitrary.

The asymmetry is deliberate, and it is about how much intent each spelling carries:

| | evidence of intent |
|---|---|
| `frame_window(scan, 4.8, 3.2)` | two coordinates whose order came out backwards — indistinguishable from a typo |
| `frames=[5, 4, 3, 2]` | the sequence written out — unambiguously deliberate |

So reverse playback is reached by asking for it: `frame_window(scan, 3.2, 4.8)[::-1]`.
A `range` slices to a `range`, and `animate_panels` honours the order, both verified.

## Rejected

**Silently swapping the endpoints** (what PR #16 did). Produces a working animation from a
mistyped call, so nothing ever surfaces the mistake. It also spends the one spelling that
could have meant reverse playback on meaning nothing.

**Returning a descending range.** Defensible — watching a diffusion cloud collapse is a real
thing to want — but it makes a transposed pair silently produce a backwards animation, and a
backwards animation of a physical process is exactly the kind of figure that gets
misinterpreted rather than noticed. Reverse playback stays available through an explicit
slice, where a reader can see it at the call site.

**Reversing only when the axis itself descends.** Some scans sweep a coordinate downward, so
"start" and "end" could be interpreted along the axis' own direction rather than by index. It
would make the rule depend on data rather than on what the caller wrote, and two scans of the
same region acquired in opposite directions would then read the same call differently.

**Refusing inside `animate_panels` too.** The engine receives indices, where a descending
sequence is a statement rather than a possible typo. Refusing there would remove reverse
playback altogether and leave the refusal in `frame_window` with nothing to offer as the
alternative.

## Consequences

`frame_window` is a thin resolver: both endpoints go through `_select_sweep_point`, so the
scan's own policies arrive unchanged — an ambiguous coordinate is refused, a distant one
warns, a nest is refused, and a role-named axis still requires `gates=`. Nothing about
coordinate lookup is reimplemented, which is why those behaviours cannot drift apart from
`plot_spectrum`'s.

Both endpoints are inclusive: a caller who names two points is asking to see both. Omitting
one runs to that edge of the scan; omitting both gives every frame.

A nest is refused rather than served, because a contiguous index range across a raster is a
snake through acquisition order, not a region. Selecting a rectangle of a raster remains
unsolved, and `frames=` is the escape hatch for anyone who knows what they want.

## Load-bearing choices

**The error message names `[::-1]`.** The refusal is only reasonable because the alternative
is one slice away; a message that refused without saying so would read as a missing feature.
