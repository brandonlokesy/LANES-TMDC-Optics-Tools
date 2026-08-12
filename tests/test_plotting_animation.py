"""
Tests for the animation engine — ``animate_panels`` and the ``AnimationPanel`` protocol.

This surface had no tests at all (defect E18). Nothing called ``animate_panels``, so the
frame-count minimum, the shared-title composition and the ``frame_label`` hook were
entirely unpinned, and the one thing that class of code gets wrong is *late* failure: an
animation builds cleanly and then dies on render, or renders the wrong frames while the
caption claims otherwise. Constructing the figure cannot catch either.

So the rule here is the same one ``test_plotting_laser_circle.py`` follows: anything that
only manifests during rendering is tested by **actually rendering**, via
``anim.save(..., PillowWriter)`` into ``tmp_path``. ``_ProbePanel`` records every frame
index it is handed, which is what makes "did it animate the frames it said it would?"
an assertion rather than an inspection.
"""

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import PillowWriter
from matplotlib.text import Text

from tmdc_optics_tools import plotting


SHAPE = (12, 16)      # (ny, nx) — non-square, so a transposed frame would show


@pytest.fixture(autouse=True)
def _close_figures():
    """Plotting functions never close their figures; don't leak them across tests."""
    yield
    plt.close("all")


class _ProbePanel(plotting.AnimationPanel):
    """
    A minimal AnimationPanel that records what the engine asked of it.

    ``seen`` is the frame index passed to every ``update`` call, in order, which is how
    the tests below check *which* frames were animated rather than merely how many.
    ``init_frames`` records the frame count handed to ``init_artists``.
    """

    def __init__(self, n_frames=4, title="", label=None, top_axis=False):
        self._n_frames   = n_frames
        self.title       = title
        self.label       = label
        self.top_axis    = top_axis
        self.seen        = []
        self.init_frames = None
        self._line       = None

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def init_artists(self, ax, n_frames: int) -> None:
        self.init_frames = n_frames
        x = np.linspace(1.5, 2.5, 32)
        (self._line,) = ax.plot(x, np.sin(x))
        ax.set_xlabel("Energy (eV)")
        if self.title:
            ax.set_title(self.title)
        if self.top_axis:
            # A conjugate wavelength axis, drawn the way a spectrum panel would.
            # Only its vertical footprint matters here.
            def _conv(e):
                e = np.asarray(e, float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    return 1239.84193 / e
            secax = ax.secondary_xaxis("top", functions=(_conv, _conv))
            secax.set_xlabel("Wavelength (nm)")

    def update(self, frame: int) -> tuple:
        self.seen.append(frame)
        return (self._line,)

    def frame_label(self, frame: int):
        if self.label is None:
            return None
        return f"{self.label} {frame}"


class _LateLabelPanel(_ProbePanel):
    """
    A panel whose ``frame_label`` only works once ``init_artists`` has run.

    ``DiffusionCloudPanel`` behaves this way — it resolves its swept-variable array in
    ``init_artists`` — and ``animate_panels`` carries a load-bearing comment about
    evaluating the suptitle afterwards for exactly this reason. Without a panel that
    actually depends on the ordering, that comment is unenforced.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ready = False

    def init_artists(self, ax, n_frames: int) -> None:
        super().init_artists(ax, n_frames)
        self._ready = True

    def frame_label(self, frame: int):
        if not self._ready:
            return None
        return f"resolved {frame}"


def _shared_title(fig) -> Text:
    """
    The engine's shared title, or ``None``.

    It is an *axes* text artist rather than ``fig.suptitle`` (blit never repaints
    figure-level artists), so it is found on an axes, not on the figure.
    """
    for ax in fig.axes:
        for text in ax.texts:
            return text
    return None


def _render(fig, anim, tmp_path, name="anim.gif"):
    """Drive every frame through a real writer and return the output path."""
    out = tmp_path / name
    anim.save(str(out), writer=PillowWriter(fps=4))
    return out


def _advanced_through(seen):
    """
    The frame sequence with consecutive repeats collapsed.

    Building the animation and priming the writer both draw frame 0 — measured as
    ``[0, 0, 0, 0, 1, 2, 3]`` for a four-frame animation — so a raw comparison against
    ``[0, 1, 2, 3]`` pins matplotlib's priming behaviour rather than ours. Collapsing
    runs keeps the assertion on the thing we control: which frames were visited, in
    what order, exactly once each.
    """
    return [f for i, f in enumerate(seen) if i == 0 or f != seen[i - 1]]


# ---------------------------------------------------------------------------
# Frame count
# ---------------------------------------------------------------------------


def test_no_panels_is_refused():
    with pytest.raises(ValueError, match="at least one panel"):
        plotting.animate_panels([])


def test_frame_count_is_the_minimum_across_panels():
    """
    The shortest panel governs. This is the behaviour the AttoCube's extra white-light
    frame currently lands on: a longer sequence is silently truncated to the shorter.
    """
    short, long_ = _ProbePanel(n_frames=4), _ProbePanel(n_frames=10)
    fig, anim = plotting.animate_panels([short, long_])

    assert short.init_frames == 4
    assert long_.init_frames == 4
    assert list(anim.new_frame_seq()) == [0, 1, 2, 3]


def test_an_explicit_frame_count_truncates():
    panel = _ProbePanel(n_frames=10)
    fig, anim = plotting.animate_panels([panel], n_frames=3)

    assert panel.init_frames == 3
    assert list(anim.new_frame_seq()) == [0, 1, 2]


def test_every_frame_is_animated_in_order(tmp_path):
    """
    Rendered, not inspected: ``update`` is only driven by the writer, so a frame-sequence
    bug is invisible until something actually saves.
    """
    panel = _ProbePanel(n_frames=4)
    fig, anim = plotting.animate_panels([panel])
    out = _render(fig, anim, tmp_path)

    assert out.stat().st_size > 0
    assert _advanced_through(panel.seen) == [0, 1, 2, 3]


def test_panels_advance_in_lock_step(tmp_path):
    a, b = _ProbePanel(n_frames=4), _ProbePanel(n_frames=4)
    fig, anim = plotting.animate_panels([a, b])
    _render(fig, anim, tmp_path)

    assert a.seen == b.seen


# ---------------------------------------------------------------------------
# The shared title
# ---------------------------------------------------------------------------


def test_the_frame_counter_is_shown_by_default():
    fig, anim = plotting.animate_panels([_ProbePanel(n_frames=4)])
    assert _shared_title(fig).get_text() == "Frame 0/4"


def test_the_frame_counter_format_is_overridable():
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4)], frame_count_fmt="{frame} of {n_frames}",
    )
    assert _shared_title(fig).get_text() == "0 of 4"


def test_panel_labels_join_the_counter():
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4, label="V ="), _ProbePanel(n_frames=4, label="P =")],
        suptitle_sep=" | ",
    )
    assert _shared_title(fig).get_text() == "Frame 0/4 | V = 0 | P = 0"


def test_a_panel_without_a_label_contributes_nothing():
    """A ``None`` label is dropped, so the separator never dangles."""
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4, label=None), _ProbePanel(n_frames=4, label="P =")],
        suptitle_sep=" | ",
    )
    assert _shared_title(fig).get_text() == "Frame 0/4 | P = 0"


def test_no_title_artist_when_there_is_nothing_to_say():
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4, label=None)], show_frame_count=False,
    )
    assert _shared_title(fig) is None


def test_the_shared_title_tracks_the_frame(tmp_path):
    """
    The reason the title is an axes artist at all: ``fig.suptitle`` is a figure-level
    artist and ``blit=True`` never repaints those, so it would freeze on frame 0.
    """
    fig, anim = plotting.animate_panels([_ProbePanel(n_frames=4)])
    title = _shared_title(fig)
    _render(fig, anim, tmp_path)

    assert title.get_text() == "Frame 3/4"
    assert title in title.axes.texts      # an axes artist, so blit repaints it


def test_a_label_resolved_in_init_artists_still_reaches_the_title():
    """
    Guards the ordering: the suptitle is built after every panel is initialised, so a
    panel that resolves its label in ``init_artists`` is not silently dropped.
    """
    fig, anim = plotting.animate_panels(
        [_LateLabelPanel(n_frames=4)], show_frame_count=False,
    )
    assert _shared_title(fig).get_text() == "resolved 0"


# ---------------------------------------------------------------------------
# Layout — the shared title against the panels' own decorations
# ---------------------------------------------------------------------------


def test_the_shared_title_clears_a_panel_title():
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4, title="Panel A")], figsize=(10, 4),
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title_top = fig.axes[0].title.get_window_extent(renderer).y1
    shared_bottom = _shared_title(fig).get_window_extent(renderer).y0
    assert shared_bottom >= title_top


@pytest.mark.xfail(
    reason="B5: the shared title's y is a hardcoded axes fraction (1.12) that knows "
           "nothing about the panels' decorations, so a secondary top axis lifts the "
           "axes title straight through it. Measured 20.1 px of overlap at "
           "figsize=(10, 4). Fixed by placing the title from a measured extent.",
    strict=True,
)
def test_the_shared_title_clears_a_secondary_top_axis():
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4, title="Panel A", top_axis=True)], figsize=(10, 4),
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title_top = fig.axes[0].title.get_window_extent(renderer).y1
    shared_bottom = _shared_title(fig).get_window_extent(renderer).y0
    assert shared_bottom >= title_top
