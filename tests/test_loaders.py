"""
Tests for AttoCubeSpectralSweep parameter loading.

Builds a small synthetic spectral CSV in the AttoCube export layout so the suite
runs without the lab network share.  Focuses on the generalized parameter store
(every labeled row exposed via ``parameters`` / ``get_parameter`` / ``[]``), the
curated attributes, and the deprecated ``AttoCubePLVabScan`` shim.

The fixture CSV is written from the same reading of the export layout as the
parser, so it cannot catch a misunderstanding shared by both — see E9 in
``dev/audit-2026-07.md``.  It does pin the decoding *contract*.
"""

import numpy as np
import pytest

from tmdc_optics_tools.loaders import (
    AttoCubePLVabScan,
    AttoCubeSpectralSweep,
    AttoCubeTRPLSweep,
    DeviceGeometry,
    _read_block_layout,
)

# 3 sweeps x 10 pixels; the first eight pixel rows also carry labeled scalars,
# leaving 2 unlabeled pixel rows to exercise the labeled-row filter.
N_SWEEPS = 3
N_PIXELS = 10
WAVELENGTH = 800.0 + np.arange(N_PIXELS)          # 800 .. 809 nm

PARAMS = {
    "V_A":              np.array([0.0, 0.5, 1.0]),
    "V_B":              np.array([0.0, -0.5, -1.0]),
    "Excitation Power": np.array([1e-6, 2e-6, 3e-6]),
    "I_A":              np.array([1e-9, 2e-9, 3e-9]),
    "I_B":              np.array([4e-9, 5e-9, 6e-9]),
    "Scanner X":        np.array([5.0, 5.0, 5.0]),
    "Scanner Y":        np.array([7.0, 7.5, 8.0]),
    "Galvo_X":          np.array([10.0, 11.0, 12.0]),
}
POWER_SCALE = 0.303e6  # AttoCubePLVabScan default


def _roi1(r, i):
    return 100 + r * 10 + i


def _roi2(r, i):
    return 200 + r * 10 + i


def make_spectral_csv(
    path,
    params      : dict = None,
    zero_blocks : int  = 0,
    interleave  : bool = False,
) -> None:
    """
    Write a spectral CSV with labeled parameter rows + 2 padding columns.

    Parameters
    ----------
    params : dict, optional
        Overrides the default :data:`PARAMS` row set, so a test can supply
        awkward labels (e.g. one containing ``/``) without a second builder.
    zero_blocks : int
        Append this many extra *declared* blocks filled entirely with literal
        zeros — what the real exporter does when it over-allocates the header.
        These are numeric, not empty, so no NaN-based strip removes them; the
        314 MB reflectance raster ships 2091 of them.
    interleave : bool
        Put the zero-filled blocks *before* the real ones instead of after,
        to exercise the anomaly warning.  Ignored when ``zero_blocks == 0``.
    """
    params = PARAMS if params is None else params
    labels = list(params)                          # first len(params) rows

    n_declared = N_SWEEPS + zero_blocks
    header = ["Parameters Labels"]
    for i in range(n_declared):
        header += [f"Par_{i}", f"Wavelength{i}", f"ExpROI1_{i}", f"ExpROI2_{i}"]
    header += ["", ""]                             # padding columns

    zeros = ["0.0", "0.0", "0.0", "0.0"]
    lines = [",".join(header)]
    for r in range(N_PIXELS):
        label = labels[r] if r < len(labels) else ""
        par = params[label] if label else np.zeros(N_SWEEPS)
        real = []
        for i in range(N_SWEEPS):
            real += [f"{par[i]}", f"{WAVELENGTH[r]}",
                     f"{_roi1(r, i)}", f"{_roi2(r, i)}"]
        pad = zeros * zero_blocks
        row = [label] + (pad + real if interleave else real + pad)
        row += ["", ""]                            # padding cells
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "scan.csv"
    make_spectral_csv(path)
    return path


@pytest.fixture
def scan(csv_path):
    return AttoCubeSpectralSweep(str(csv_path), spectra_type="PL")


# ---------------------------------------------------------------------------
# Generic parameter store
# ---------------------------------------------------------------------------


def test_all_labeled_parameters_exposed(scan):
    assert set(scan.parameter_labels) == set(PARAMS)
    for label, expected in PARAMS.items():
        assert np.allclose(scan.parameters[label], expected), label


def test_parameter_arrays_have_sweep_length(scan):
    assert scan.n_sweeps == N_SWEEPS
    assert scan.n_pixels == N_PIXELS
    for arr in scan.parameters.values():
        assert arr.shape == (N_SWEEPS,)


def test_unlabeled_pixel_rows_excluded(scan):
    # Only the labeled rows become parameters; the 2 trailing pixel rows do not.
    assert len(scan.parameters) == len(PARAMS)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def test_get_parameter_and_scale(scan):
    assert np.allclose(scan.get_parameter("Galvo_X"), PARAMS["Galvo_X"])
    assert np.allclose(scan.get_parameter("Galvo_X", scale=2.0),
                       PARAMS["Galvo_X"] * 2.0)


def test_getitem_sugar(scan):
    assert np.allclose(scan["Galvo_X"], PARAMS["Galvo_X"])


def test_missing_parameter_raises_with_available(scan):
    with pytest.raises(KeyError) as exc:
        scan.get_parameter("does_not_exist")
    assert "Galvo_X" in str(exc.value)  # error lists available labels


# ---------------------------------------------------------------------------
# Backward-compatible curated attributes
# ---------------------------------------------------------------------------


def test_curated_attributes_unchanged(scan):
    assert np.allclose(scan.v_top, PARAMS["V_A"])           # raw
    assert np.allclose(scan.v_bot, PARAMS["V_B"])           # raw
    assert np.allclose(scan.power, PARAMS["Excitation Power"] * POWER_SCALE)
    assert np.allclose(scan.Ich1, PARAMS["I_A"] * 1e9)      # -> nA
    assert np.allclose(scan.Ich2, PARAMS["I_B"] * 1e9)      # -> nA


def test_curated_attrs_are_raw_views_of_parameters(scan):
    # parameters holds raw units; curated power/current are scaled, not raw.
    assert np.allclose(scan.v_top, scan.parameters["V_A"])
    assert np.allclose(scan.power / POWER_SCALE, scan.parameters["Excitation Power"])


def test_ef_none_without_geometry(scan):
    assert scan.ef is None


def test_undeclared_sweep_gives_index_axis_not_a_guess(scan):
    # V_A, V_B, Excitation Power and Scanner Y all vary in the fixture, so any
    # auto-pick would be a coin toss.  The axis must fall back to the index.
    assert scan.sweep_type == "index"
    assert np.allclose(scan.sweep_axis, np.arange(N_SWEEPS))
    assert scan.sweep_axis_label == "Sweep index"


# ---------------------------------------------------------------------------
# Spectra / axes
# ---------------------------------------------------------------------------


def test_spectra_and_wavelength(scan):
    assert np.allclose(scan.wavelength, WAVELENGTH)
    expected = np.array([[_roi1(r, i) for i in range(N_SWEEPS)]
                         for r in range(N_PIXELS)], dtype=float)
    assert np.allclose(scan.spectra, expected)
    # energy axis is ascending
    assert np.all(np.diff(scan.energy) > 0)


def test_roi2_selection(csv_path):
    scan2 = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL", roi=2)
    expected = np.array([[_roi2(r, i) for i in range(N_SWEEPS)]
                         for r in range(N_PIXELS)], dtype=float)
    assert np.allclose(scan2.spectra, expected)


def test_both_rois_always_loaded(scan):
    # roi= selects what `spectra` points at; neither ROI is discarded at load,
    # which is where a reference channel would live for a non-PL measurement.
    roi1 = np.array([[_roi1(r, i) for i in range(N_SWEEPS)]
                     for r in range(N_PIXELS)], dtype=float)
    roi2 = np.array([[_roi2(r, i) for i in range(N_SWEEPS)]
                     for r in range(N_PIXELS)], dtype=float)
    assert np.allclose(scan.spectra_roi1, roi1)
    assert np.allclose(scan.spectra_roi2, roi2)
    assert scan.spectra is scan.spectra_roi1


def test_padding_columns_stripped(scan):
    # 2 padding cols make raw n_cols = 14 (not divisible by 4); they must be
    # stripped to 12 -> 3 sweeps without raising.
    assert scan.n_sweeps == N_SWEEPS


# ---------------------------------------------------------------------------
# Curated registry (#2) and property backing (#1)
# ---------------------------------------------------------------------------


def test_curated_parameters_registry(scan):
    reg = scan.curated_parameters
    assert reg["v_top"] == ("V_A", 1.0, "V")
    assert reg["power"] == ("Excitation Power", 0.303e6, "µW")
    assert reg["Ich1"] == ("I_A", 1e9, "nA")
    # The scanners are piezos; the rows carry drive voltage, not a distance.
    assert reg["scanner_x"] == ("Scanner X", 1.0, "V")
    assert reg["scanner_y"] == ("Scanner Y", 1.0, "V")


def test_scanner_position_properties(scan):
    assert np.allclose(scan.scanner_x, PARAMS["Scanner X"])
    assert np.allclose(scan.scanner_y, PARAMS["Scanner Y"])
    # scale 1.0 -> curated view equals the raw parameter row
    assert np.allclose(scan.scanner_x, scan.parameters["Scanner X"])


def test_curated_properties_are_read_only(scan):
    # Properties backed by self.parameters -> no setter.
    with pytest.raises(AttributeError):
        scan.v_top = np.zeros(N_SWEEPS)
    with pytest.raises(AttributeError):
        scan.ef = np.zeros(N_SWEEPS)


def test_constructor_overrides_flow_through_registry(csv_path):
    s = AttoCubeSpectralSweep(
        str(csv_path), spectra_type="PL",
        curated_labels={"v_top": "V_B"}, curated_scales={"power": 1.0},
    )
    # scale override: power now equals the raw row
    assert np.allclose(s.power, s.parameters["Excitation Power"])
    assert s.curated_parameters["power"] == ("Excitation Power", 1.0, "µW")
    # label override: v_top now reads the V_B row
    assert np.allclose(s.v_top, s.parameters["V_B"])
    assert s.curated_parameters["v_top"][0] == "V_B"


def test_unknown_curated_name_rejected(csv_path):
    with pytest.raises(ValueError, match="not a curated parameter"):
        AttoCubeSpectralSweep(str(csv_path), spectra_type="PL",
                              curated_labels={"v_topp": "V_B"})


def test_ef_property_with_geometry(csv_path):
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    s = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="electric_field", geometry=geom)
    assert s.ef is not None
    assert np.allclose(s.ef, geom.electric_field(s.v_top, s.v_bot))
    assert np.allclose(s.sweep_axis, s.ef)
    assert s.sweep_axis_label == r"$E_F$ (mV/nm)"


# ---------------------------------------------------------------------------
# Missing curated rows (E1): load, then raise only if accessed
# ---------------------------------------------------------------------------


def test_missing_curated_row_still_loads(csv_path):
    # A file from a different instrument configuration must not be unloadable
    # just because one curated row is absent.
    s = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL",
                              curated_labels={"power": "NoSuchRow"})
    assert s.n_sweeps == N_SWEEPS
    with pytest.raises(KeyError, match="Galvo_X"):   # error lists what is available
        s.power


def test_declared_sweep_requires_its_own_row(csv_path):
    # The fail-fast that remains is the one the caller's declaration implies.
    with pytest.raises(KeyError, match="power"):
        AttoCubeSpectralSweep(str(csv_path), spectra_type="PL", sweep="power",
                              curated_labels={"power": "NoSuchRow"})


def test_electric_field_sweep_requires_geometry(csv_path):
    with pytest.raises(ValueError, match="DeviceGeometry"):
        AttoCubeSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="electric_field")


# ---------------------------------------------------------------------------
# spectra_type and sweep resolution
# ---------------------------------------------------------------------------


def test_spectra_type_is_required(csv_path):
    with pytest.raises(ValueError, match="spectra_type is required"):
        AttoCubeSpectralSweep(str(csv_path))


def test_spectra_type_vocabulary_enforced(csv_path):
    with pytest.raises(ValueError, match="not a recognised measurement type"):
        AttoCubeSpectralSweep(str(csv_path), spectra_type="photoluminescence")


def test_signal_label_follows_spectra_type(csv_path):
    pl = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL")
    rc = AttoCubeSpectralSweep(str(csv_path), spectra_type="RC")
    assert pl.signal_label == "PL intensity (counts)"
    assert rc.signal_label == r"$\Delta R/R_0$"     # dimensionless: no unit
    assert rc.spectroscopy == "Reflectance contrast"


def test_raw_row_as_sweep_axis(csv_path):
    s = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="Galvo_X", sweep_unit="V")
    assert np.allclose(s.sweep_axis, PARAMS["Galvo_X"])
    assert s.sweep_axis_label == "Galvo_X (V)"


def test_unknown_sweep_lists_both_registries(csv_path):
    with pytest.raises(ValueError) as exc:
        AttoCubeSpectralSweep(str(csv_path), spectra_type="PL", sweep="nonsense")
    assert "electric_field" in str(exc.value)   # known sweep types
    assert "Galvo_X" in str(exc.value)          # rows in this file


def test_piezo_sweep_uses_scanner_row(csv_path):
    s = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="piezo_y")
    assert np.allclose(s.sweep_axis, PARAMS["Scanner Y"])
    assert s.sweep_axis_label == r"Piezo $y$ (V)"


def test_old_position_sweep_key_is_refused_with_the_new_name(csv_path):
    # No shim for the rename: the unknown-key path lists the valid types, so the
    # error hands back the name that replaced it.
    with pytest.raises(ValueError) as exc:
        AttoCubeSpectralSweep(str(csv_path), spectra_type="PL",
                              sweep="position_y")
    assert "piezo_y" in str(exc.value)


# ---------------------------------------------------------------------------
# Diagnostics: what actually varied
# ---------------------------------------------------------------------------


def test_varying_parameters_excludes_static_rows(scan):
    varying = scan.varying_parameters()
    assert "Scanner X" not in varying          # constant at 5.0 in the fixture
    assert "V_A" in varying and "Scanner Y" in varying
    assert varying["V_A"] == (0.0, 1.0, 1.0)   # (min, max, span)


def test_gate_mode_detects_antisymmetric_sweep(scan):
    # V_A goes 0 -> +1 while V_B goes 0 -> -1: a field-like sweep.
    assert scan.gate_mode == "dual-gate, anti-correlated (field-like)"


def test_gate_mode_none_when_gate_rows_absent(csv_path):
    s = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL",
                              curated_labels={"v_top": "NoSuchRow"})
    assert s.gate_mode is None


def test_repr_survives_missing_rows(csv_path):
    # __repr__ raised for every geometry once already (A2); it must not raise
    # here either, whatever the file happens to lack.
    s = AttoCubeSpectralSweep(str(csv_path), spectra_type="PL",
                              curated_labels={"power": "NoSuchRow"})
    assert "AttoCubeSpectralSweep" in repr(s)
    assert "Photoluminescence" in repr(s)


# ---------------------------------------------------------------------------
# Block-layout detection and zero-filled (unwritten) blocks
# ---------------------------------------------------------------------------


def test_layout_read_from_header_alone(csv_path):
    layout = _read_block_layout(str(csv_path))
    assert layout["kind"] == "spectral"
    assert layout["block_width"] == 4
    assert layout["n_blocks"] == N_SWEEPS
    assert layout["roles"] == ("Par", "Wavelength", "ExpROI1", "ExpROI2")


def test_label_column_is_not_mistaken_for_a_block(csv_path):
    # The header's own first column is "Parameters Labels"; a prefix match on
    # "Par" would treat it as a block start and give block_width == 1.
    assert _read_block_layout(str(csv_path))["block_width"] == 4


def test_trailing_zero_blocks_dropped(tmp_path):
    # The exporter over-allocates and fills the surplus with literal zeros, not
    # blanks, so nothing NaN-based removes them.  Keeping them would fabricate
    # sweep points that were never measured.
    csv = tmp_path / "overallocated.csv"
    make_spectral_csv(csv, zero_blocks=N_SWEEPS)
    s = AttoCubeSpectralSweep(str(csv), spectra_type="PL")

    assert s.n_sweeps == N_SWEEPS
    assert s.n_declared_sweeps == 2 * N_SWEEPS
    assert not (s.spectra.sum(axis=0) == 0).any()
    assert "zero-filled and dropped" in repr(s)
    # Parameters are trimmed in step with the spectra, or every sweep point
    # would be misaligned against its own parameter values.
    assert np.allclose(s.parameters["V_A"], PARAMS["V_A"])
    for arr in s.parameters.values():
        assert arr.shape == (N_SWEEPS,)


def test_interleaved_zero_blocks_warn_and_keep_everything(tmp_path):
    # Interleaving would mean the format model is wrong, so guessing which
    # columns to discard could silently misalign the sweep.  Keep all, warn.
    csv = tmp_path / "interleaved.csv"
    make_spectral_csv(csv, zero_blocks=2, interleave=True)
    with pytest.warns(UserWarning, match="interleaved"):
        s = AttoCubeSpectralSweep(str(csv), spectra_type="PL")
    assert s.n_sweeps == N_SWEEPS + 2


def test_all_blocks_unwritten_names_the_metadata_companion(tmp_path):
    # A spectral-layout file whose every block is zero-filled is the metadata
    # companion written alongside a TRPL sweep.
    csv = tmp_path / "companion.csv"
    make_spectral_csv(csv, zero_blocks=0)
    text = csv.read_text().splitlines()
    # Blank out the wavelength and both ROI columns, keeping the Par values.
    rewritten = [text[0]]
    for line in text[1:]:
        cells = line.split(",")
        for i in range(N_SWEEPS):
            for off in (2, 3, 4):                  # Wavelength, ROI1, ROI2
                cells[1 + i * 4 + off - 1] = "0.0"
        rewritten.append(",".join(cells))
    csv.write_text("\n".join(rewritten) + "\n")

    with pytest.raises(ValueError, match="metadata companion"):
        AttoCubeSpectralSweep(str(csv), spectra_type="PL")


def test_headerless_csv_named_by_row_count(tmp_path):
    # A 2-row spectrum and a real-space image are both bare numeric grids with no
    # header, so only the row count separates them.  Each must be pointed at the
    # class that reads it, rather than both at whichever was guessed.
    spectrum = tmp_path / "one_spectrum.csv"
    spectrum.write_text("800.0,801.0,802.0\n10.0,11.0,12.0\n")
    with pytest.raises(ValueError, match="SingleSpectrum"):
        AttoCubeSpectralSweep(str(spectrum), spectra_type="PL")

    image = tmp_path / "frame.csv"
    image.write_text("1,2,3\n4,5,6\n7,8,9\n")
    with pytest.raises(ValueError, match="AttoCubePLScanRealSpace"):
        AttoCubeSpectralSweep(str(image), spectra_type="PL")


def test_row_count_in_that_message_is_not_a_capped_read(tmp_path):
    # The check stops after three lines, so it must describe the shape as
    # "more than two rows" and never report the bounded count as the real one.
    image = tmp_path / "tall.csv"
    image.write_text("\n".join(",".join("1" for _ in range(4)) for _ in range(50)) + "\n")
    with pytest.raises(ValueError, match="more than two rows"):
        AttoCubeSpectralSweep(str(image), spectra_type="PL")


def test_single_spectrum_rejection_reaches_the_trpl_class_too(tmp_path):
    # Both sweep classes share the layout reader, so both give the same guidance.
    spectrum = tmp_path / "one_spectrum.csv"
    spectrum.write_text("800.0,801.0\n10.0,11.0\n")
    with pytest.raises(ValueError, match="SingleSpectrum"):
        AttoCubeTRPLSweep(str(spectrum))


def test_selected_roi_all_zero_warns(tmp_path):
    # ExpROI2 is blank except on two-spot galvo scans, and a flat zero array is
    # otherwise indistinguishable from a valid dark measurement.
    csv = tmp_path / "scan.csv"
    make_spectral_csv(csv, params={"V_A": PARAMS["V_A"]})
    text = csv.read_text().splitlines()
    rewritten = [text[0]]
    for line in text[1:]:
        cells = line.split(",")
        for i in range(N_SWEEPS):
            cells[1 + i * 4 + 3] = "0.0"           # ExpROI2
        rewritten.append(",".join(cells))
    csv.write_text("\n".join(rewritten) + "\n")

    with pytest.warns(UserWarning, match="ExpROI2"):
        AttoCubeSpectralSweep(str(csv), spectra_type="PL", roi=2)


# ---------------------------------------------------------------------------
# Deprecated AttoCubePLVabScan shim
# ---------------------------------------------------------------------------


def test_shim_warns_and_reproduces_old_defaults(csv_path):
    with pytest.warns(FutureWarning, match="AttoCubeSpectralSweep"):
        s = AttoCubePLVabScan(str(csv_path))
    assert s.spectra_type == "PL"
    # Old gate_axis was v_top when no geometry was supplied.
    assert s.sweep_type == "top_voltage"
    assert np.allclose(s.gate_axis, s.v_top)


def test_shim_with_geometry_uses_field_axis(csv_path):
    geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
    with pytest.warns(FutureWarning):
        s = AttoCubePLVabScan(str(csv_path), geometry=geom)
    assert np.allclose(s.gate_axis, s.ef)
    assert s.gate_axis_label == r"$E_F$ (mV/nm)"


def test_shim_translates_old_label_arguments(csv_path):
    with pytest.warns(FutureWarning):
        s = AttoCubePLVabScan(str(csv_path), power_scale=1.0,
                              top_gate_label="V_B")
    assert s.curated_parameters["v_top"][0] == "V_B"
    assert np.allclose(s.power, s.parameters["Excitation Power"])
