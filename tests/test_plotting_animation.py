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
from matplotlib.animation import HTMLWriter, PillowWriter
from matplotlib.text import Text
from PIL import Image

from tmdc_optics_tools import plotting


SHAPE = (12, 16)      # (ny, nx) — non-square, so a transposed frame would show

# Matplotlib warns from Animation.__del__ when an animation is collected having never
# drawn. Many tests here deliberately build a figure and assert on it without
# rendering, so the warning is expected noise for them rather than a signal — and a
# noisy suite hides real warnings. It appears at all only because the engine does not
# blit: setting up blitting used to force an init draw, which marked the animation as
# started as a side effect. Scoped to this one message so anything else still surfaces.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning"
)


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

    A real ``fig.suptitle``, so it lives on the figure. That is what makes the layout
    engine reserve room for it, which is what keeps it clear of whatever the panels
    draw on top (E20).
    """
    return fig._suptitle if fig.texts else None


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
    Blitting only repaints axes artists, so a figure-level title freezes on frame 0
    under ``blit=True`` — verified frozen in both the notebook slider and MP4. The
    engine therefore redraws in full, and this renders to prove the title moves.
    """
    fig, anim = plotting.animate_panels([_ProbePanel(n_frames=4)])
    title = _shared_title(fig)
    _render(fig, anim, tmp_path)

    assert title.get_text() == "Frame 3/4"
    assert title in fig.texts


def test_the_engine_does_not_blit(tmp_path):
    """
    Pins the reason, not just the effect. If this is ever flipped back to ``True``,
    the shared title silently freezes in the notebook slider and in MP4 — the two
    paths that matter most here, and the two a GIF-only check would not catch.
    """
    fig, anim = plotting.animate_panels([_ProbePanel(n_frames=4)])
    assert anim._blit is False


def test_the_header_updates_in_the_notebook_player(tmp_path):
    """
    The workflow this package is actually used through: ``to_jshtml``'s player, whose
    slider steps frame by frame.

    Tested behaviourally rather than by checking the blit flag, because this is the
    path that silently broke — with blitting on, every slider frame carried the
    frame-0 header while the GIF looked fine. ``HTMLWriter`` is what ``to_jshtml``
    uses; ``embed_frames=False`` makes it write the frames as real PNGs, which are
    the pixels the slider shows.
    """
    fig, anim = plotting.animate_panels([_ProbePanel(n_frames=3)], figsize=(6, 3))
    anim.save(str(tmp_path / "player.html"),
              writer=HTMLWriter(fps=4, embed_frames=False))

    headers = []
    for png in sorted(tmp_path.rglob("frame*.png")):
        with Image.open(png) as img:
            img.load()
            headers.append(np.asarray(img.convert("L"))[:40, :].copy())

    assert len(headers) == 3
    # Every frame's header strip must differ from the first: the counter changed.
    assert all(not np.array_equal(h, headers[0]) for h in headers[1:])


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


@pytest.mark.parametrize("n_panels", [1, 2, 3, 4])
@pytest.mark.parametrize("figsize", [(10, 4), (5, 4), (20, 4), (10, 2.5)])
def test_the_shared_title_clears_a_secondary_top_axis(n_panels, figsize):
    """
    E20's regression guard, parametrised because the old bug was scale-dependent.

    A panel that draws a secondary top axis is the case that broke: the layout engine
    shrinks the panel rather than growing the figure, so a title positioned as a
    fraction of the panel slid down into the panel's own title — measured 20.1 px of
    overlap. A real suptitle is reserved space instead, and clears by ~8 px at every
    panel count and figure size tried.
    """
    panels = [
        _ProbePanel(n_frames=4, title=f"Panel {i}", top_axis=True)
        for i in range(n_panels)
    ]
    fig, anim = plotting.animate_panels(panels, figsize=figsize)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title_top = max(ax.title.get_window_extent(renderer).y1 for ax in fig.axes[:n_panels])
    shared_bottom = _shared_title(fig).get_window_extent(renderer).y0
    assert shared_bottom >= title_top
