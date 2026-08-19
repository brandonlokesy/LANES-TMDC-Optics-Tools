"""
Tests for the diffusion-cloud analysis/plotting double-subtraction fix (A5).

``analyse_diffusion_cloud`` and ``plot_diffusion_cloud`` accept both a bare
2-D array and an ``_AttoCubeImage``-family object (which may already be
background-subtracted, from a ``bg_region`` passed at *construction* time).
The bug: handing the plotting function's already-corrected ``.img`` array
to the analysis function, plus a *second* ``bg_region``, subtracted twice.
The fix relies on passing the image *object* through so ``_load_image`` can
apply its own "use img_raw for an _AttoCubeImage" rule, and on
``DiffusionResult.image`` carrying the array that was actually analysed
(what a caller should display), rather than every caller re-deriving it.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np

from tmdc_optics_tools import plotting
from tmdc_optics_tools.diffusion import analyse_diffusion_cloud
from tmdc_optics_tools.loaders import AttoCubePLImage

BG_LEVEL  = 50.0
BG_REGION = (slice(0, 3), slice(0, 3))  # flat corner patch, away from the cloud


def _make_image(tmp_path):
    img = np.full((10, 10), BG_LEVEL)
    img[5:8, 5:8] += 500.0  # bright "cloud", well above background + threshold
    path = tmp_path / "frame.csv"
    np.savetxt(path, img, delimiter=",")
    return path


def test_loader_bg_region_is_a_single_subtraction(tmp_path):
    # Sanity check on the loader itself, independent of the diffusion code:
    # bg_region at construction leaves that patch at ~0 exactly once.
    image = AttoCubePLImage(_make_image(tmp_path), bg_region=BG_REGION)
    assert np.allclose(image.img[BG_REGION], 0.0, atol=1e-9)
    assert np.allclose(image.img_raw[BG_REGION], BG_LEVEL, atol=1e-9)


def test_analyse_diffusion_cloud_does_not_double_subtract(tmp_path):
    image = AttoCubePLImage(_make_image(tmp_path), bg_region=BG_REGION)

    result = analyse_diffusion_cloud(image, bg_region=BG_REGION, smooth_sigma=0.0)

    # A second subtraction of the same already-corrected region would drive
    # it to -BG_LEVEL instead of leaving it at ~0.
    assert np.allclose(result.image[BG_REGION], 0.0, atol=1e-9)
    assert not np.allclose(result.image[BG_REGION], -BG_LEVEL, atol=1e-3)
    # The cloud itself is unaffected either way.
    assert result.area_px2 > 0


def test_analyse_diffusion_cloud_on_a_bare_array_still_subtracts_once(tmp_path):
    # A caller who was never handed an _AttoCubeImage (a raw array) must
    # still get exactly one subtraction, not zero.
    img = np.full((10, 10), BG_LEVEL)
    img[5:8, 5:8] += 500.0

    result = analyse_diffusion_cloud(img, bg_region=BG_REGION, smooth_sigma=0.0)
    assert np.allclose(result.image[BG_REGION], 0.0, atol=1e-9)


def test_plot_diffusion_cloud_displays_the_same_array_it_analysed(tmp_path):
    image = AttoCubePLImage(_make_image(tmp_path), bg_region=BG_REGION)

    fig, ax, result = plotting.plot_diffusion_cloud(
        image, bg_region=BG_REGION, smooth_sigma=0.0, laser_annotation=False,
    )
    assert np.allclose(result.image[BG_REGION], 0.0, atol=1e-9)

    displayed = np.asarray(ax.images[0].get_array())
    assert np.array_equal(displayed, result.image)
    matplotlib.pyplot.close(fig)


def test_plot_diffusion_cloud_with_precomputed_result_uses_its_image(tmp_path):
    image = AttoCubePLImage(_make_image(tmp_path), bg_region=BG_REGION)
    result = analyse_diffusion_cloud(image, bg_region=BG_REGION, smooth_sigma=0.0)

    fig, ax, result_out = plotting.plot_diffusion_cloud(
        image, result=result, laser_annotation=False,
    )
    assert result_out is result
    displayed = np.asarray(ax.images[0].get_array())
    assert np.array_equal(displayed, result.image)
    matplotlib.pyplot.close(fig)
