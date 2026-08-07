"""
Tests for plot_image's extent/origin parameters, added for RamanMap's
physical-coordinate maps.

Forces the Agg backend and does a real draw, per the convention in
test_plotting_laser_circle.py -- constructing a figure without rendering it
would not catch an imshow() call with a bad kwarg.
"""

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from tmdc_optics_tools import plotting

DATA = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]])


def test_plot_image_default_behaviour_is_unchanged():
    # extent=None, origin="upper" (the new parameters' defaults) must
    # reproduce the pre-existing pixel-index behaviour exactly.
    fig, ax, im = plotting.plot_image(DATA)
    fig.canvas.draw()
    assert im.get_extent() == [-0.5, 2.5, 1.5, -0.5]  # matplotlib's own pixel-index default
    assert im.origin == "upper"
    plt.close(fig)


def test_plot_image_extent_and_origin_are_applied():
    fig, ax, im = plotting.plot_image(
        DATA, extent=(0.0, 3.0, 0.0, 2.0), origin="lower",
        xlabel="X (um)", ylabel="Y (um)",
    )
    fig.canvas.draw()
    assert im.get_extent() == [0.0, 3.0, 0.0, 2.0]
    assert im.origin == "lower"
    assert ax.get_xlabel() == "X (um)"
    plt.close(fig)


def test_plot_image_masks_nan_instead_of_coloring_it():
    fig, ax, im = plotting.plot_image(DATA)
    fig.canvas.draw()
    arr = im.get_array()
    assert np.ma.is_masked(arr[0, 2])
    assert not np.ma.is_masked(arr[0, 0])
    plt.close(fig)


def test_plot_image_all_nan_still_renders_via_clim():
    # plot_image itself with an explicit clim must not raise on all-NaN data.
    fig, ax, im = plotting.plot_image(np.full((2, 2), np.nan), clim=(0.0, 1.0))
    fig.canvas.draw()
    plt.close(fig)
