"""
One spectral-axis vocabulary, one refusal, shared by every ``x_axis=``.

``constants.X_AXES`` names the two orderings a spectrum can be served on, and
``_x_axis_name_unit`` is the only thing that decides whether a value is one of
them. Each entry point still picks its own arrays — what is shared is the
refusal, and the reason to share it is that a value which is neither key is
otherwise read as the wavelength one by the branch below it, and returns or fits
the wrong array in silence.

The five parametrised entry points are the whole public surface of that
vocabulary: two accessors, the spectral-window helper, the peak fitter and a
plotting function. The last test pins what the refusal deliberately does *not*
cover — asking for a wavelength-space source on the energy axis still only
warns, because an explicitly named source is served on its own axis.
"""

import warnings

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tmdc_optics_tools import fitting, plotting
from tmdc_optics_tools.constants import X_AXES, _x_axis_name_unit
from tmdc_optics_tools.loaders import AttoCubeSpectralSweep

from test_loaders import make_spectral_csv


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "sweep.csv"
    make_spectral_csv(path)
    return AttoCubeSpectralSweep(str(path), spectra_type="PL")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x_axis", sorted(X_AXES))
def test_every_row_unpacks_to_a_name_and_a_unit(x_axis):
    """A row is what an axis label is composed from, so both fields must be there."""
    name, unit = X_AXES[x_axis]
    assert isinstance(name, str) and name
    assert isinstance(unit, str) and unit


def test_the_composed_labels_are_the_ones_the_plots_carry(scan):
    labels = {axis: plotting._resolve_x_axis(scan, axis)[1] for axis in X_AXES}
    assert labels == {"energy": "Energy (eV)", "wavelength": "Wavelength (nm)"}


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


def test_the_refusal_names_every_key():
    # Derived from the table rather than written out, so a key added to X_AXES
    # cannot go unmentioned in the message.
    with pytest.raises(ValueError) as excinfo:
        _x_axis_name_unit("eV")
    message = str(excinfo.value)
    assert all(repr(key) in message for key in X_AXES)
    assert "'eV'" in message


def test_the_refusal_names_the_calling_function():
    with pytest.raises(ValueError, match=r"^pixel_slice\(\): x_axis must be"):
        _x_axis_name_unit("eV", what="pixel_slice()")


def test_an_unhashable_axis_is_a_value_error_not_a_type_error():
    with pytest.raises(ValueError, match="must be 'energy' or 'wavelength'"):
        _x_axis_name_unit(["energy"])


@pytest.mark.parametrize("call", [
    pytest.param(lambda s: s.pixel_slice((1.6, 1.7), x_axis="eV"),
                 id="pixel_slice"),
    pytest.param(lambda s: s.get_spectrum_at(value=0.0, x_axis="eV"),
                 id="get_spectrum_at"),
    pytest.param(lambda s: s.get_spectrum_by_index(0, x_axis="eV"),
                 id="get_spectrum_by_index"),
    pytest.param(lambda s: fitting.fit_scan_peak(s, x_axis="eV"),
                 id="fit_scan_peak"),
    pytest.param(lambda s: plotting.plot_spectral_map(s, x_axis="eV"),
                 id="plot_spectral_map"),
])
def test_every_entry_point_refuses_an_unknown_axis(scan, call):
    # Each of these used to serve the wavelength array instead. The accessors
    # resolve the sweep point first, so their coordinates are ones the fixture
    # really holds — the axis is what is being refused, not the point.
    with pytest.raises(ValueError, match="must be 'energy' or 'wavelength'"):
        call(scan)


def test_a_valid_axis_still_reaches_the_data(scan):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert scan.get_spectrum_at(value=0.0, x_axis="wavelength").shape == (
            scan.n_pixels,)
        assert np.array_equal(
            scan.get_spectrum_at(value=0.0, x_axis="wavelength"),
            scan.spectra[:, 0],
        )


# ---------------------------------------------------------------------------
# The axis and the source cannot disagree
# ---------------------------------------------------------------------------


def test_a_named_source_is_served_on_the_axis_asked_for(scan):
    # A source names a correction state and the axis names the representation, so
    # every state exists on both and no combination can be a mismatch.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        on_wavelength = scan.get_spectrum_at(value=0.0, source="raw",
                                             x_axis="wavelength")
        on_energy     = scan.get_spectrum_at(value=0.0, source="raw",
                                             x_axis="energy")
    assert np.array_equal(on_wavelength, scan.spectra[:, 0])
    assert np.array_equal(on_energy, scan.energy_spectra[:, 0])
