"""
Tests for the declared 2-D sweep nest — ``fast_sweep=`` / ``slow_sweep=``.

Covers the declaration and its refusals, the ``as_grid`` reshape, the by-value
and by-index accessors and the shapes they return, the out-of-range warning, the
sawtooth warning a 1-D sweep on a raster now raises, and the HDF5 round trip.

The raster fixtures are synthetic: the committed reflectance export is the first
50 points of a 41 × 51 scan, so it exercises the aborted-raster refusal and
nothing else.  A small ``n_fast × n_slow`` grid is written with the same builder
the rest of the suite uses, which pins the contract without pretending to
reproduce one file's dimensions.
"""

import warnings

import numpy as np
import pytest

from tmdc_optics_tools.loaders import (
    AttoCubeSpectralSweep,
    DeviceGeometry,
    StackLayer,
    _axis_atol,
)

from test_loaders import make_spectral_csv

# 4 fast inside 3 slow.  Small and round: the arithmetic is meant to be obvious
# at a glance, not to match any particular export.
N_FAST, N_SLOW = 4, 3
N_SWEEPS       = N_FAST * N_SLOW
GATES          = {"top": "V_A", "bottom": "V_B"}

# A spatial raster.  Everything but the two scanners is held still, so exactly
# one pair of rows can form a grid and a detection test has a single answer.
RASTER = {
    "V_A":              np.full(N_SWEEPS, 1.0),
    "V_B":              np.full(N_SWEEPS, -1.0),
    "Excitation Power": np.full(N_SWEEPS, 2e-6),
    "Scanner X":        np.tile(np.arange(N_FAST, dtype=float) * 2.0, N_SLOW),
    "Scanner Y":        np.repeat(np.arange(N_SLOW, dtype=float) * 5.0, N_FAST),
}

FAST_VALUES = RASTER["Scanner X"][:N_FAST]          # 0, 2, 4, 6
SLOW_VALUES = RASTER["Scanner Y"][::N_FAST]         # 0, 5, 10

# A field sweep nested inside a power sweep.  Both gates move together by a
# per-row common-mode offset that leaves the field alone, so the *field* takes
# n_fast values while each gate row takes all n_fast x n_slow — which is what
# makes the derived axis the only one that can be declared here.
_FIELD  = np.tile(np.array([4.0, 3.0, 2.0, 1.0]), N_SLOW)      # descending
_COMMON = np.repeat(np.array([10.0, 20.0, 30.0]), N_FAST)

FIELD_SWEEP = {
    "V_A":              _COMMON - _FIELD / 2,       # top
    "V_B":              _COMMON + _FIELD / 2,       # bottom; V_B - V_A = field
    "Excitation Power": np.repeat(np.linspace(1e-6, 3e-6, N_SLOW), N_FAST),
    "Scanner X":        np.full(N_SWEEPS, 5.0),
    "Scanner Y":        np.full(N_SWEEPS, 7.0),
}

# A gate sweep nested inside a power sweep whose slow axis is *read back* at every
# point rather than commanded, so each level has a width of its own.  The wobble
# is a fixed fraction of the reading and the levels are decade-spaced, which
# makes the top level wider than a tolerance scaled to the whole axis while every
# step inside it stays far smaller than the distance between levels.
_WOBBLE   = np.array([-9e-4, -3e-4, 3e-4, 9e-4])       # fraction of the reading
_LEVELS   = np.array([1.0, 10.0, 100.0])               # µW
_READBACK = np.repeat(_LEVELS, N_FAST) * (1 + np.tile(_WOBBLE, N_SLOW))

POWER_SWEEP = {
    "V_A":              np.full(N_SWEEPS, 1.0),
    "V_B":              np.tile(np.array([-2.0, -1.0, 1.0, 2.0]), N_SLOW),
    "Excitation Power": _READBACK / 0.303e6,           # undo the loader's µW scale
    "Scanner X":        np.full(N_SWEEPS, 5.0),
    "Scanner Y":        np.full(N_SWEEPS, 7.0),
}


@pytest.fixture
def raster_csv(tmp_path):
    path = tmp_path / "raster.csv"
    make_spectral_csv(path, params=RASTER)
    return path


@pytest.fixture
def field_csv(tmp_path):
    path = tmp_path / "field.csv"
    make_spectral_csv(path, params=FIELD_SWEEP)
    return path


@pytest.fixture
def nested(raster_csv):
    return AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL",
                                 fast_sweep="Scanner X",
                                 slow_sweep="Scanner Y")


@pytest.fixture
def flat(raster_csv):
    """The same file with no nest declared — every grid entry point then refuses."""
    return AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL")


@pytest.fixture
def geometry():
    return DeviceGeometry(tmdc_stack=[StackLayer("MoSe2"), StackLayer("WSe2")],
                          d_hbn_top=50.0, d_hbn_bottom=50.0)


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_declared_nest_reports_its_shape_and_coordinates(nested):
    nest = nested.nesting
    assert nested.is_nested
    assert (nest.n_fast, nest.n_slow) == (N_FAST, N_SLOW)
    assert nest.shape == (N_SLOW, N_FAST)
    assert np.allclose(nest.fast_axis, FAST_VALUES)
    assert np.allclose(nest.slow_axis, SLOW_VALUES)
    assert nest.fast_type == "Scanner X" and nest.slow_type == "Scanner Y"


def test_undeclared_sweep_is_not_nested(flat):
    assert not flat.is_nested
    assert flat.nesting is None


def test_coordinates_keep_acquisition_order(field_csv, geometry):
    """A descending axis stays descending: the coordinates are not sorted."""
    scan = AttoCubeSpectralSweep(str(field_csv), spectra_type="PL",
                                 gates=GATES, geometry=geometry,
                                 fast_sweep="electric_field", slow_sweep="power")
    fast = scan.nesting.fast_axis
    assert np.all(np.diff(fast) < 0), fast


def test_derived_axis_nests_where_the_raw_rows_cannot(field_csv, geometry):
    """
    The motivating case: both gates moved together to sweep the field.

    Each gate row takes a different value at every one of the ``n_fast x
    n_slow`` points, so no row-level reading can express the nest.  The field
    they encode takes exactly ``n_fast``, and declares it.
    """
    scan = AttoCubeSpectralSweep(str(field_csv), spectra_type="PL",
                                 gates=GATES, geometry=geometry,
                                 fast_sweep="electric_field", slow_sweep="power")
    assert (scan.nesting.n_fast, scan.nesting.n_slow) == (N_FAST, N_SLOW)
    assert scan.nesting.fast_unit == "mV/nm"

    # The rows the field is derived from vary at every point, so neither is an
    # axis, and neither detection nor declaration can make one of them into one.
    assert np.unique(scan["V_A"]).size == N_SWEEPS
    assert scan.sweep_grid() is None
    with pytest.raises(ValueError, match="does not describe"):
        AttoCubeSpectralSweep(str(field_csv), spectra_type="PL",
                              fast_sweep="V_A", slow_sweep="power")


def test_a_level_wider_than_the_axis_tolerance_still_nests(tmp_path):
    """
    A slow axis read back per point rather than commanded: levels have a width.

    That width grows with the reading while a tolerance scaled to the axis's span
    is fixed by the largest reading, so on a decade-spaced power sweep the top
    level is wider than the tolerance while the levels stay cleanly apart.  What
    the nest turns on is which level each point sits on, so it verifies.
    """
    path = tmp_path / "power.csv"
    make_spectral_csv(path, params=POWER_SWEEP)
    scan = AttoCubeSpectralSweep(str(path), spectra_type="PL",
                                 fast_sweep="V_B", slow_sweep="power")

    assert (scan.nesting.n_fast, scan.nesting.n_slow) == (N_FAST, N_SLOW)
    np.testing.assert_allclose(scan.nesting.slow_axis, _LEVELS, rtol=1e-3)

    # Pin that the fixture reaches the case: comparing readings against one of
    # their own would reject this nest, so passing it is not free.
    top = scan.power[-N_FAST:]
    assert np.ptp(top) > _axis_atol(scan.power)


def test_a_wobbling_level_still_names_the_swap(tmp_path):
    """The inverted declaration must still be told apart from a genuine mismatch."""
    path = tmp_path / "power.csv"
    make_spectral_csv(path, params=POWER_SWEEP)
    with pytest.raises(ValueError) as exc:
        AttoCubeSpectralSweep(str(path), spectra_type="PL",
                              fast_sweep="power", slow_sweep="V_B")
    assert "Swapping them does" in str(exc.value)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_swapped_declaration_raises_and_names_the_swap(raster_csv):
    with pytest.raises(ValueError) as exc:
        AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL",
                              fast_sweep="Scanner Y", slow_sweep="Scanner X")
    msg = str(exc.value)
    assert "Swapping them does" in msg
    assert "fast_sweep='Scanner X'" in msg
    assert f"({N_SLOW} distinct)" in msg and f"({N_FAST} distinct)" in msg


@pytest.mark.parametrize("kwargs, missing", [
    ({"fast_sweep": "Scanner X"}, "slow_sweep"),
    ({"slow_sweep": "Scanner Y"}, "fast_sweep"),
])
def test_one_axis_alone_raises(raster_csv, kwargs, missing):
    with pytest.raises(ValueError, match=missing):
        AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL", **kwargs)


def test_same_axis_twice_raises(raster_csv):
    with pytest.raises(ValueError, match="two different axes"):
        AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL",
                              fast_sweep="Scanner X", slow_sweep="Scanner X")


def test_aborted_raster_raises_naming_both_counts(tmp_path):
    """A partial final row cannot be reshaped — the committed export's own case."""
    partial = {k: v[:-1] for k, v in RASTER.items()}      # 11 of 12 points
    path = tmp_path / "aborted.csv"
    make_spectral_csv(path, params=partial)
    with pytest.raises(ValueError) as exc:
        AttoCubeSpectralSweep(str(path), spectra_type="PL",
                              fast_sweep="Scanner X", slow_sweep="Scanner Y")
    assert "aborted" in str(exc.value)
    assert f"not {N_SWEEPS - 1}" in str(exc.value)


def test_the_committed_truncated_raster_refuses():
    """The real export is 50 of 41 × 51 points, so it must not reshape."""
    path = ("examples/data/reflectance-contrast/"
            "sample_truncated_26_07_24_17_55_47_iter_0.csv")
    with pytest.raises(ValueError):
        AttoCubeSpectralSweep(path, spectra_type="R",
                              fast_sweep="Scanner X", slow_sweep="Scanner Y")


# ---------------------------------------------------------------------------
# as_grid
# ---------------------------------------------------------------------------


def test_as_grid_is_a_view_that_round_trips(nested):
    cube = nested.as_grid(nested.spectra)
    assert cube.shape == (nested.n_pixels, N_SLOW, N_FAST)
    assert np.shares_memory(cube, nested.spectra)
    assert np.array_equal(cube.reshape(nested.n_pixels, -1), nested.spectra)


def test_as_grid_reshapes_a_parameter_row(nested):
    grid = nested.as_grid(nested["Scanner X"])
    assert grid.shape == (N_SLOW, N_FAST)
    # Every row of the raster ran the same fast values.
    assert np.allclose(grid, FAST_VALUES)


def test_as_grid_refuses_without_a_nest_and_points_at_the_declaration(flat):
    with pytest.raises(ValueError) as exc:
        flat.as_grid(flat.spectra)
    assert "fast_sweep=" in str(exc.value)


def test_as_grid_refuses_a_mismatched_trailing_axis(nested):
    with pytest.raises(ValueError, match="last axis"):
        nested.as_grid(nested.wavelength)


def test_declaring_a_nest_does_not_reshape_the_stored_arrays(nested, flat):
    """The shape-polymorphism guard: `spectra` means the same thing either way."""
    assert nested.spectra.shape == flat.spectra.shape == (nested.n_pixels,
                                                          N_SWEEPS)
    assert nested.n_sweeps == flat.n_sweeps == N_SWEEPS
    assert np.array_equal(nested.spectra, flat.spectra)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def test_return_shapes_follow_how_much_is_specified(nested, flat):
    n_px = nested.n_pixels
    assert nested.get_spectrum_at(fast=2.0, slow=5.0).shape == (n_px,)
    assert nested.get_spectrum_at(fast=2.0).shape == (n_px, N_SLOW)
    assert nested.get_spectrum_at(slow=5.0).shape == (n_px, N_FAST)
    assert flat.get_spectrum_by_index(2).shape == (n_px,)


def test_accessor_results_are_views(nested):
    source = nested.best_energy_spectra
    for arr in (nested.get_spectrum_at(fast=2.0, slow=5.0),
                nested.get_spectrum_at(fast=2.0),
                nested.get_spectrum_at(slow=5.0)):
        assert np.shares_memory(arr, source)


def test_by_value_and_by_index_agree_with_the_flat_column(nested):
    i_fast, i_slow = 1, 2                                  # x = 2.0, y = 10.0
    expected = nested.best_energy_spectra[:, i_slow * N_FAST + i_fast]
    assert np.array_equal(nested.get_spectrum_at(fast=2.0, slow=10.0), expected)
    assert np.array_equal(
        nested.get_spectrum_by_index(fast=i_fast, slow=i_slow), expected)


def test_pinning_one_axis_takes_the_right_stride(nested):
    be = nested.best_energy_spectra
    assert np.array_equal(nested.get_spectrum_at(fast=2.0), be[:, 1::N_FAST])
    assert np.array_equal(nested.get_spectrum_at(slow=10.0),
                          be[:, 2 * N_FAST:3 * N_FAST])


def test_negative_indices_count_from_the_end(nested):
    assert np.array_equal(nested.get_spectrum_by_index(fast=-1, slow=-1),
                          nested.best_energy_spectra[:, -1])


def test_source_selects_the_array(raster_csv):
    scan = AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL",
                                 fast_sweep="Scanner X", slow_sweep="Scanner Y")
    assert np.array_equal(scan.get_spectrum_at(fast=2.0, slow=5.0, source="raw",
                                               x_axis="wavelength"),
                          scan.spectra[:, 1 * N_FAST + 1])
    with pytest.raises(ValueError, match="not available"):
        scan.get_spectrum_at(fast=2.0, slow=5.0, source="contrast")


@pytest.mark.parametrize("call, match", [
    (lambda s: s.get_spectrum_at(2.0),           "does not locate a point"),
    (lambda s: s.get_spectrum_at(),              "name fast= and/or slow="),
    (lambda s: s.get_spectrum_by_index(fast=99), "out of range"),
])
def test_accessor_refusals_on_a_nest(nested, call, match):
    with pytest.raises((ValueError, IndexError), match=match):
        call(nested)


def test_fast_and_slow_refuse_on_a_flat_sweep(flat):
    with pytest.raises(ValueError, match="need a declared nest"):
        flat.get_spectrum_at(fast=2.0)


def test_nearest_index_rejects_an_unknown_axis(nested):
    with pytest.raises(ValueError, match="neither a known sweep type"):
        nested.nearest_index(1.0, axis="nope")


# ---------------------------------------------------------------------------
# Out-of-range reporting
# ---------------------------------------------------------------------------


def test_a_value_off_the_axis_warns_and_names_both(nested):
    with pytest.warns(UserWarning, match="found no point there") as caught:
        idx = nested.nearest_index(500.0, axis="slow")
    assert idx == N_SLOW - 1
    assert "Looking up 500" in str(caught[0].message)
    assert "using index 2 at 10" in str(caught[0].message)


def test_an_exact_value_is_silent(nested):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert nested.nearest_index(4.0, axis="fast") == 2


def test_a_value_between_two_points_is_silent(nested):
    """Half a step is the threshold, so asking between grid points is fine."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert nested.nearest_index(3.0, axis="fast") in (1, 2)


# ---------------------------------------------------------------------------
# axis= — locating a point by a quantity other than the declared axis
# ---------------------------------------------------------------------------

# A 1-D field sweep driven by both gates at a fixed ratio: the sweep is declared
# in the field, but a point is just as well identified by either gate voltage.
_N_FIELD = 7
_FIELD   = np.linspace(2.0, 26.0, _N_FIELD)          # V_bot - V_top
FIXED_RATIO = {
    "V_A":              20.0 - _FIELD / 2,           # top: 19 .. 7, monotonic
    "V_B":              20.0 + _FIELD / 2,
    "Excitation Power": np.linspace(1e-6, 2e-6, _N_FIELD),
    "Scanner X":        np.full(_N_FIELD, 5.0),
}


@pytest.fixture
def field_sweep(tmp_path, geometry):
    path = tmp_path / "fixed_ratio.csv"
    make_spectral_csv(path, params=FIXED_RATIO)
    return AttoCubeSpectralSweep(str(path), spectra_type="PL", gates=GATES,
                                 geometry=geometry, sweep="electric_field")


def test_a_point_can_be_found_by_a_curated_quantity(field_sweep):
    idx = field_sweep.nearest_index(15.0, axis="top_voltage")
    assert np.isclose(field_sweep.v_top[idx], 15.0)
    assert np.array_equal(field_sweep.get_spectrum_at(15.0, axis="top_voltage"),
                          field_sweep.best_energy_spectra[:, idx])


def test_a_point_can_be_found_by_a_raw_row(field_sweep):
    assert (field_sweep.nearest_index(15.0, axis="V_A")
            == field_sweep.nearest_index(15.0, axis="top_voltage"))


def test_axis_does_not_change_the_declared_sweep(field_sweep):
    """Looking up by another quantity is a lookup, not a redeclaration."""
    assert field_sweep.sweep_type == "electric_field"
    field_sweep.get_spectrum_at(15.0, axis="top_voltage")
    assert field_sweep.sweep_type == "electric_field"


def test_axis_returns_one_spectrum(field_sweep):
    assert (field_sweep.get_spectrum_at(15.0, axis="top_voltage").shape
            == (field_sweep.n_pixels,))


def test_an_unknown_axis_lists_both_vocabularies(field_sweep):
    with pytest.raises(ValueError) as exc:
        field_sweep.nearest_index(1.0, axis="nope")
    msg = str(exc.value)
    assert "Sweep types" in msg and "File rows" in msg
    assert "'fast' and 'slow'" in msg


def test_axis_cannot_be_combined_with_the_nest_axes(nested):
    with pytest.raises(ValueError, match="cannot be combined"):
        nested.get_spectrum_at(2.0, axis="V_A", fast=1.0)


def test_a_non_injective_quantity_warns_naming_every_match(tmp_path):
    """A hysteresis loop passes the same gate voltage twice."""
    loop = np.concatenate([np.linspace(0.0, 6.0, 4), np.linspace(4.0, 0.0, 3)])
    params = {"V_A": loop, "V_B": 30.0 - loop,
              "Excitation Power": np.full(loop.size, 2e-6),
              "Scanner X": np.full(loop.size, 5.0)}
    path = tmp_path / "hysteresis.csv"
    make_spectral_csv(path, params=params)
    scan = AttoCubeSpectralSweep(str(path), spectra_type="PL")

    with pytest.warns(UserWarning, match="at 2 sweep points") as caught:
        idx = scan.nearest_index(4.0, axis="V_A")
    assert idx == 2
    assert "indices 2, 4" in str(caught[0].message)


def test_a_value_midway_between_distinct_points_is_not_a_duplicate(field_sweep):
    """The duplicate test is on the coordinate, not on the distance."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        field_sweep.nearest_index(16.0, axis="top_voltage")   # between 17 and 15


# ---------------------------------------------------------------------------
# A8 — a sweep axis that does not label its points individually
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sweep, n_distinct", [
    ("piezo_x", N_FAST),     # the inner quantity: restarts every row
    ("piezo_y", N_SLOW),     # the outer one: holds still through a row
])
def test_either_quantity_of_a_nest_warns_when_declared_as_the_axis(
        raster_csv, sweep, n_distinct):
    """Both leave repeats, so both collapse a map — direction is beside the point."""
    with pytest.warns(UserWarning, match="does not label them individually") as c:
        AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL", sweep=sweep)
    msg = str(c[0].message)
    assert f"only {n_distinct} different values across {N_SWEEPS}" in msg
    assert "Scanner X (4) × Scanner Y (3)" in msg
    assert "as_grid()" in msg


def test_a_nest_no_grid_search_can_see_still_warns(field_csv, geometry):
    """
    The case a grid-detection guard would have missed.

    Both gates vary at every point of the field x power scan, so ``sweep_grid()``
    finds nothing to report — while ``sweep="power"`` collapses a map exactly as
    a raster's slow axis does.
    """
    with pytest.warns(UserWarning, match="does not label them individually") as c:
        scan = AttoCubeSpectralSweep(str(field_csv), spectra_type="PL",
                                     gates=GATES, geometry=geometry,
                                     sweep="power")
    assert scan.sweep_grid() is None
    assert "fast_sweep=" in str(c[0].message)


def test_an_undeclared_sweep_on_a_raster_is_silent(raster_csv):
    """The flat index labels every point, so there is nothing to say."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL")


def test_an_axis_that_labels_every_point_is_silent(field_csv, geometry):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AttoCubeSpectralSweep(str(field_csv), spectra_type="PL", gates=GATES,
                              geometry=geometry, sweep="top_voltage")


# ---------------------------------------------------------------------------
# An ambiguous coordinate is refused by the accessors, warned by nearest_index
# ---------------------------------------------------------------------------


@pytest.fixture
def undeclared_raster(raster_csv):
    """A raster nobody declared — flat as far as the loader knows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AttoCubeSpectralSweep(str(raster_csv), spectra_type="PL")


def test_an_ambiguous_coordinate_is_refused_not_silently_narrowed(
        undeclared_raster):
    with pytest.raises(ValueError) as exc:
        undeclared_raster.get_spectrum_at(5.0, axis="piezo_y")
    msg = str(exc.value)
    assert "does not identify one spectrum" in msg
    assert "4 sweep points (indices 4, 5, 6, 7)" in msg
    # It names the complete answer rather than only refusing.
    assert "fast_sweep='Scanner X'" in msg and "slow_sweep='Scanner Y'" in msg
    assert "get_spectrum_by_index()" in msg


def test_the_declared_nest_answers_what_the_refusal_points_at(nested):
    """The route the message names returns every match, not one of them."""
    block = nested.get_spectrum_at(slow=5.0)
    assert block.shape == (nested.n_pixels, N_FAST)


def test_nearest_index_still_warns_rather_than_raising(undeclared_raster):
    """A single index is its whole contract, so it has nothing better to return."""
    with pytest.warns(UserWarning, match="does not identify one"):
        idx = undeclared_raster.nearest_index(5.0, axis="piezo_y")
    assert idx == N_FAST      # first point of the second row


def test_an_unambiguous_coordinate_still_works(field_sweep):
    assert (field_sweep.get_spectrum_at(15.0, axis="top_voltage").shape
            == (field_sweep.n_pixels,))


def test_by_index_is_unaffected_by_ambiguity(undeclared_raster):
    """An index says which point it means, so there is nothing to refuse."""
    assert (undeclared_raster.get_spectrum_by_index(5).shape
            == (undeclared_raster.n_pixels,))


def test_repr_distinguishes_a_declared_nest_from_a_detected_one(nested, flat):
    assert "Nesting" in repr(nested)
    assert "detected, not declared" in repr(flat)
    assert "fast_sweep='Scanner X'" in repr(flat)


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------


def test_declared_nest_survives_a_round_trip(nested, tmp_path):
    path = tmp_path / "nested.h5"
    nested.to_hdf5(str(path))
    back = AttoCubeSpectralSweep(str(path), spectra_type="PL")
    assert back.is_nested
    assert back.nesting.fast_type == "Scanner X"
    assert back.nesting.slow_type == "Scanner Y"
    assert np.allclose(back.nesting.fast_axis, nested.nesting.fast_axis)
    assert np.allclose(back.nesting.slow_axis, nested.nesting.slow_axis)
    assert np.array_equal(back.get_spectrum_at(fast=2.0, slow=5.0),
                          nested.get_spectrum_at(fast=2.0, slow=5.0))


def test_a_derived_nest_survives_a_round_trip(field_csv, tmp_path, geometry):
    scan = AttoCubeSpectralSweep(str(field_csv), spectra_type="PL",
                                 gates=GATES, geometry=geometry,
                                 fast_sweep="electric_field", slow_sweep="power")
    path = tmp_path / "derived.h5"
    scan.to_hdf5(str(path))
    back = AttoCubeSpectralSweep(str(path), spectra_type="PL")
    assert back.nesting.fast_type == "electric_field"
    assert np.allclose(back.nesting.fast_axis, scan.nesting.fast_axis)


def test_an_undeclared_scan_gains_no_nest_on_read(flat, tmp_path):
    path = tmp_path / "flat.h5"
    flat.to_hdf5(str(path))
    back = AttoCubeSpectralSweep(str(path), spectra_type="PL")
    assert not back.is_nested


def test_the_argument_still_overrides_the_file(flat, tmp_path):
    path = tmp_path / "flat.h5"
    flat.to_hdf5(str(path))
    back = AttoCubeSpectralSweep(str(path), spectra_type="PL",
                                 fast_sweep="Scanner X", slow_sweep="Scanner Y")
    assert back.nesting.shape == (N_SLOW, N_FAST)
