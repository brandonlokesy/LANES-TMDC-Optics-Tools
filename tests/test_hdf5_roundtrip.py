"""
Round-trip tests for tmdc_optics_tools.hdf5.

The point of storing metadata alongside the data is that a re-read reproduces the
same object, so the test that matters is CSV -> scan -> HDF5 -> scan.  Anything
the writer omits shows up here as a difference, which is why the reader exists at
all rather than the format being write-only.

Reuses the synthetic CSV builder from test_loaders rather than forking a second
copy of the export layout.
"""

import numpy as np
import pytest

from tmdc_optics_tools import hdf5
from tmdc_optics_tools.loaders import (
    AttoCubeSpectralSweep,
    AttoCubeTRPLSweep,
    DeviceGeometry,
    StackLayer,
)

from test_loaders import GATES, N_SWEEPS, PARAMS, make_spectral_csv


@pytest.fixture
def geom():
    return DeviceGeometry(
        tmdc_stack   = [StackLayer("MoSe2"), StackLayer("WSe2", n_layers=2)],
        d_hbn_top    = 53.0,
        d_hbn_bottom = 46.0,
        label        = "hBN/MoSe2/WSe2/hBN",
    )


@pytest.fixture
def scan(tmp_path, geom):
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    return AttoCubeSpectralSweep(
        str(csv), spectra_type="PL", sweep="electric_field", geometry=geom,
        gates=GATES,
    )


@pytest.fixture
def reloaded(scan, tmp_path):
    scan.to_hdf5(tmp_path / "scan.h5")
    return AttoCubeSpectralSweep(tmp_path / "scan.h5")


# ---------------------------------------------------------------------------
# Data survives the round trip
# ---------------------------------------------------------------------------


def test_arrays_round_trip_exactly(scan, reloaded):
    assert np.array_equal(reloaded.wavelength, scan.wavelength)
    assert np.array_equal(reloaded.spectra_roi1, scan.spectra_roi1)
    assert np.array_equal(reloaded.spectra_roi2, scan.spectra_roi2)


def test_every_parameter_row_round_trips(scan, reloaded):
    assert set(reloaded.parameter_labels) == set(scan.parameter_labels)
    for label in scan.parameter_labels:
        assert np.array_equal(reloaded.parameters[label],
                              scan.parameters[label]), label


def test_parameter_labels_containing_a_slash_survive(tmp_path):
    # "/" is the HDF5 path separator: an unsanitised label would silently
    # become a subgroup, and the label must come back as it went in.
    params = dict(PARAMS)
    params["I_A/I_B"] = np.linspace(1.0, 2.0, N_SWEEPS)
    csv = tmp_path / "odd.csv"
    make_spectral_csv(csv, params=params)

    scan = AttoCubeSpectralSweep(str(csv), spectra_type="PL")
    scan.to_hdf5(tmp_path / "odd.h5")
    back = AttoCubeSpectralSweep(tmp_path / "odd.h5")

    assert "I_A/I_B" in back.parameters
    assert np.array_equal(back.parameters["I_A/I_B"], params["I_A/I_B"])


# ---------------------------------------------------------------------------
# Metadata survives the round trip
# ---------------------------------------------------------------------------


def test_measurement_metadata_restored_without_being_re_supplied(scan, reloaded):
    # No spectra_type=, no sweep=, no geometry= passed on re-read.
    assert reloaded.spectra_type == "PL"
    assert reloaded.sweep_type   == "electric_field"
    assert reloaded.sweep_axis_label == scan.sweep_axis_label
    assert reloaded.roi == scan.roi


def test_geometry_restored_and_field_axis_reproduced(scan, reloaded):
    assert reloaded.geometry is not None
    assert np.allclose(reloaded.sweep_axis, scan.sweep_axis)
    assert reloaded.geometry.stack_label == scan.geometry.stack_label
    assert reloaded.geometry.eps_stack == pytest.approx(scan.geometry.eps_stack)
    assert [l.material for l in reloaded.geometry.tmdc_stack] == ["MoSe2", "WSe2"]
    assert reloaded.geometry.tmdc_stack[1].n_layers == 2


def test_geometry_with_no_top_hbn_round_trips(tmp_path):
    # d_hbn_top=None has no HDF5 attribute type; it is stored as NaN and must
    # come back as None, not as a NaN thickness that poisons eps_stack.
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    single = DeviceGeometry(tmdc_stack=[StackLayer("WSe2")],
                            d_hbn_top=None, d_hbn_bottom=50.0)
    scan = AttoCubeSpectralSweep(str(csv), spectra_type="PL",
                                 sweep="bottom_voltage", geometry=single,
                                 gates=GATES)
    scan.to_hdf5(tmp_path / "single.h5")
    back = AttoCubeSpectralSweep(tmp_path / "single.h5")

    assert back.geometry.d_hbn_top is None
    assert back.geometry.d_hbn_bottom == 50.0
    assert np.isfinite(back.geometry.eps_stack)


def test_curated_overrides_restored(tmp_path, geom):
    # An unusual gate wiring is exactly the metadata that is nowhere else
    # recorded; reverting to the class default on read would flip the field sign.
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    scan = AttoCubeSpectralSweep(
        str(csv), spectra_type="PL", sweep="electric_field", geometry=geom,
        gates={"top": "V_B", "bottom": "V_A"},
        curated_scales={"power": 1.0},
    )
    scan.to_hdf5(tmp_path / "wired.h5")
    back = AttoCubeSpectralSweep(tmp_path / "wired.h5")

    assert back.gates == {"top": "V_B", "bottom": "V_A"}
    assert back.curated_parameters["v_top"][0] == "V_B"
    assert back.curated_parameters["v_bot"][0] == "V_A"
    assert back.curated_parameters["power"][1] == 1.0
    assert np.allclose(back.ef, scan.ef)


def test_undeclared_wiring_stays_undeclared_on_read(tmp_path):
    # The curated dump always carries a resolved label for both gate rows, so
    # without a separate record a round trip would turn an unstated wiring into a
    # stated one -- laundering the assumption into provenance.
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    scan = AttoCubeSpectralSweep(str(csv), spectra_type="PL")
    assert scan.gates is None
    scan.to_hdf5(tmp_path / "unwired.h5")
    back = AttoCubeSpectralSweep(tmp_path / "unwired.h5")

    assert back.gates is None
    with pytest.raises(ValueError, match="not declared"):
        back.v_top


def test_raw_row_sweep_metadata_restored(tmp_path):
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    scan = AttoCubeSpectralSweep(str(csv), spectra_type="PL",
                                 sweep="Galvo_X", sweep_unit="V")
    scan.to_hdf5(tmp_path / "galvo.h5")
    back = AttoCubeSpectralSweep(tmp_path / "galvo.h5")

    assert back.sweep_type == "Galvo_X"
    assert back.sweep_axis_label == "Galvo_X (V)"
    assert np.allclose(back.sweep_axis, PARAMS["Galvo_X"])


def test_argument_overrides_stored_spectra_type_with_a_warning(scan, tmp_path):
    scan.to_hdf5(tmp_path / "scan.h5")
    with pytest.warns(UserWarning, match="records spectra_type"):
        back = AttoCubeSpectralSweep(tmp_path / "scan.h5", spectra_type="R")
    assert back.spectra_type == "R"


# ---------------------------------------------------------------------------
# Corrections are recorded as provenance, never replayed
# ---------------------------------------------------------------------------


def test_corrections_are_not_replayed_on_read(tmp_path, geom):
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    corrected = AttoCubeSpectralSweep(
        str(csv), spectra_type="PL", sweep="electric_field", geometry=geom,
        gates=GATES, bg_region_nm=(806.0, 809.0), apply_jacobian=True,
    )
    corrected.to_hdf5(tmp_path / "corrected.h5")
    back = AttoCubeSpectralSweep(tmp_path / "corrected.h5")

    # Recorded, so the session's choices are inspectable...
    assert back.source_metadata["apply_jacobian"] is True
    assert back.source_metadata["bg_region_nm"] == pytest.approx((806.0, 809.0))
    # ...but not re-applied: loading is not deciding.
    assert back.apply_jacobian is False
    assert back.bg_region_nm is None
    assert back.energy_spectra_bg is None


def test_stored_spectra_are_raw(tmp_path, geom):
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    corrected = AttoCubeSpectralSweep(
        str(csv), spectra_type="PL", sweep="electric_field", geometry=geom,
        gates=GATES, bg_region_nm=(806.0, 809.0), apply_jacobian=True,
    )
    corrected.to_hdf5(tmp_path / "corrected.h5")
    back = AttoCubeSpectralSweep(tmp_path / "corrected.h5")
    # The file holds the untouched detector counts, not the corrected arrays.
    assert np.array_equal(back.spectra, corrected.spectra)


def test_reapplying_corrections_on_read_reproduces_them(tmp_path, geom):
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    kwargs = dict(spectra_type="PL", sweep="electric_field", geometry=geom,
                  gates=GATES, bg_region_nm=(806.0, 809.0), apply_jacobian=True)
    original = AttoCubeSpectralSweep(str(csv), **kwargs)
    original.to_hdf5(tmp_path / "scan.h5")
    back = AttoCubeSpectralSweep(tmp_path / "scan.h5", **kwargs)

    assert np.allclose(back.energy_spectra, original.energy_spectra)
    assert np.allclose(back.energy_spectra_bg, original.energy_spectra_bg)


# ---------------------------------------------------------------------------
# Write guards and format checks
# ---------------------------------------------------------------------------


def test_overwrite_is_opt_in(scan, tmp_path):
    out = tmp_path / "scan.h5"
    scan.to_hdf5(out)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        scan.to_hdf5(out)
    assert scan.to_hdf5(out, overwrite=True) == out


def test_parent_directory_created(scan, tmp_path):
    out = tmp_path / "nested" / "deeper" / "scan.h5"
    assert scan.to_hdf5(out) == out
    assert out.exists()


def test_foreign_hdf5_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    stranger = tmp_path / "stranger.h5"
    with h5py.File(stranger, "w") as hf:
        hf.attrs["format"] = "something.else"
    with pytest.raises(ValueError, match="is not a"):
        AttoCubeSpectralSweep(stranger, spectra_type="PL")


def test_unknown_suffix_names_the_supported_formats(tmp_path):
    stray = tmp_path / "scan.txt"
    stray.write_text("not a scan\n")
    with pytest.raises(ValueError, match=r"\.csv"):
        AttoCubeSpectralSweep(stray, spectra_type="PL")


# ---------------------------------------------------------------------------
# Both axis kinds through one format
# ---------------------------------------------------------------------------


def test_trpl_sweep_round_trips(tmp_path):
    # An assembled 4-file directory collapses to one self-describing archive.
    trpl = AttoCubeTRPLSweep("examples/data/TRPL", bg_region_ns=(0.0, 1.0),
                             gates=GATES)
    out  = trpl.to_hdf5(tmp_path / "trpl.h5")
    back = AttoCubeTRPLSweep(out)

    assert back.n_sweeps == trpl.n_sweeps == 3
    assert back.n_bins == trpl.n_bins
    assert np.array_equal(back.time, trpl.time)
    assert np.array_equal(back.decays, trpl.decays)
    assert back.spectra_type == "TRPL"
    assert np.allclose(back.v_top, trpl.v_top)
    # Provenance recorded, correction not replayed.
    assert back.source_metadata["bg_region_ns"] == pytest.approx((0.0, 1.0))
    assert back.bg_region_ns is None
    assert back.decays_bg is None


def test_trpl_source_files_recorded(tmp_path):
    trpl = AttoCubeTRPLSweep("examples/data/TRPL")
    back = AttoCubeTRPLSweep(trpl.to_hdf5(tmp_path / "trpl.h5"))
    names = back.source_metadata["source_files"]
    assert len(names) == 3
    assert [n[-10:] for n in names] == ["iter_0.csv", "iter_1.csv", "iter_2.csv"]


def test_spectral_h5_rejected_by_the_trpl_class(scan, tmp_path):
    out = scan.to_hdf5(tmp_path / "spectral.h5")
    with pytest.raises(ValueError, match="AttoCubeSpectralSweep"):
        AttoCubeTRPLSweep(out)


def test_trpl_h5_rejected_by_the_spectral_class(tmp_path):
    trpl = AttoCubeTRPLSweep("examples/data/TRPL")
    out  = trpl.to_hdf5(tmp_path / "trpl.h5")
    with pytest.raises(ValueError, match="AttoCubeTRPLSweep"):
        AttoCubeSpectralSweep(out, spectra_type="TRPL")


def test_axis_dataset_named_for_its_quantity(scan, tmp_path):
    h5py = pytest.importorskip("h5py")
    trpl = AttoCubeTRPLSweep("examples/data/TRPL")
    scan.to_hdf5(tmp_path / "spectral.h5")
    trpl.to_hdf5(tmp_path / "trpl.h5")

    with h5py.File(tmp_path / "spectral.h5", "r") as hf:
        assert "axes/wavelength" in hf and "axes/time" not in hf
        assert set(hf["spectra"]) == {"roi1", "roi2"}
        assert hf["axes/wavelength"].attrs["units"] == "nm"
        assert hf["metadata"].attrs["axis_kind"] == "wavelength"
    with h5py.File(tmp_path / "trpl.h5", "r") as hf:
        assert "axes/time" in hf and "axes/wavelength" not in hf
        assert set(hf["decays"]) == {"counts"}
        assert hf["axes/time"].attrs["units"] == "ns"
        assert hf["metadata"].attrs["axis_kind"] == "time"


# ---------------------------------------------------------------------------
# Reference spectra travel with the archive
# ---------------------------------------------------------------------------


def test_reference_stored_as_an_array_not_a_path(tmp_path, geom):
    # A path goes stale; the array makes the archive stand alone.  It is still
    # provenance, so the contrast is not rebuilt on read without being asked.
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    ref = np.linspace(50.0, 60.0, 10)
    scan = AttoCubeSpectralSweep(str(csv), spectra_type="R", reference=ref,
                                 bg_spectrum=np.full(10, 2.0))
    back = AttoCubeSpectralSweep(scan.to_hdf5(tmp_path / "rc.h5"),
                                 spectra_type="R")

    assert np.allclose(back.source_metadata["reference"], ref)
    assert np.allclose(back.source_metadata["bg_spectrum"], 2.0)
    assert back.contrast is None            # recorded, not replayed
    # ...and re-supplying it from the archive reproduces the contrast exactly.
    again = AttoCubeSpectralSweep(
        tmp_path / "rc.h5", spectra_type="R",
        reference=back.source_metadata["reference"],
        bg_spectrum=back.source_metadata["bg_spectrum"],
    )
    assert np.allclose(again.contrast, scan.contrast)


def test_auxiliary_spectra_live_beside_the_signal(tmp_path):
    # They are measured arrays on the file's own axis, not descriptions of the
    # measurement, so /auxiliary is their home rather than /metadata.
    h5py = pytest.importorskip("h5py")
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    ref  = np.linspace(50.0, 60.0, 10)
    scan = AttoCubeSpectralSweep(str(csv), spectra_type="RC", reference=ref,
                                 reference_scale=2.0,
                                 bg_spectrum=np.full(10, 2.0))

    with h5py.File(scan.to_hdf5(tmp_path / "rc.h5"), "r") as hf:
        assert set(hf["auxiliary"]) == {"bg_spectrum", "reference"}
        assert "bg_spectrum" not in hf["metadata"]
        assert "reference"   not in hf["metadata"]

        # The two scalars a contrast depends on hang off the array they qualify,
        # and the stored values already carry the scaling the name reports.
        stored = hf["auxiliary/reference"]
        assert stored.attrs["contrast_mode"] == "contrast"
        assert stored.attrs["scale_applied"] == 2.0
        assert np.allclose(stored[()], ref * 2.0)
        assert "reference_scale" not in hf["metadata"].attrs


def test_no_auxiliary_group_when_no_auxiliary_spectra(scan, tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(scan.to_hdf5(tmp_path / "plain.h5"), "r") as hf:
        assert "auxiliary" not in hf


def test_reference_attrs_come_back_as_provenance(tmp_path):
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv)
    ref  = np.linspace(50.0, 60.0, 10)
    scan = AttoCubeSpectralSweep(str(csv), spectra_type="RC", reference=ref,
                                 reference_scale=2.0)
    meta = hdf5.read_sweep(scan.to_hdf5(tmp_path / "rc.h5"))["metadata"]
    assert meta["contrast_mode"]   == "contrast"
    assert meta["reference_scale"] == 2.0
    assert np.allclose(meta["reference"], ref * 2.0)


def test_older_major_format_version_is_refused(scan, tmp_path):
    # A 1.x file kept the auxiliary spectra somewhere this reader does not look,
    # so a tolerant read would drop a recorded reference without saying so.
    h5py = pytest.importorskip("h5py")
    out = scan.to_hdf5(tmp_path / "scan.h5")
    with h5py.File(out, "r+") as hf:
        hf.attrs["format_version"] = "1.1"

    with pytest.raises(ValueError, match="format_version"):
        hdf5.read_sweep(out)
    with pytest.raises(ValueError, match="format_version"):
        AttoCubeSpectralSweep(out)


def test_read_sweep_returns_the_payload_contract(scan, tmp_path):
    scan.to_hdf5(tmp_path / "scan.h5")
    payload = hdf5.read_sweep(tmp_path / "scan.h5")
    assert set(payload) == {"wavelength", "roi1", "roi2",
                            "parameters", "metadata", "axis_kind"}
    assert payload["roi1"].shape == scan.spectra_roi1.shape
    # axis_kind is how the loader rejects an .h5 of the wrong measured axis, the
    # way a header does for a CSV.
    assert payload["axis_kind"] == "wavelength"
