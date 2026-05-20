# tmdc_optics_tools

A Python toolkit for gate-dependent photoluminescence spectroscopy on TMDC monolayers and van der Waals heterostructure devices. Covers data loading from AttoCube cryogenic confocal setups, device geometry modelling, spectral processing, peak fitting, DC Stark shift and dipole length extraction, and publication-ready plotting and animation.

> ⚠️ **Alpha stage.** This library is under active development. Class names, function signatures, and module structure are all subject to change without notice. Pin to a specific commit if you need stability.

---

## Contents

- [Installation](#installation)
- [Package structure](#package-structure)
- [Key workflows](#key-workflows)
  - [1. Define device geometry](#1-define-device-geometry)
  - [2. Load a gate-dependent PL scan](#2-load-a-gate-dependent-pl-scan)
  - [3. Plot a PL map](#3-plot-a-pl-map)
  - [4. Inspect a single spectrum](#4-inspect-a-single-spectrum)
  - [5. Fit a peak across a sweep](#5-fit-a-peak-across-a-sweep)
  - [6. Extract the excitonic dipole length](#6-extract-the-excitonic-dipole-length)
  - [7. Real-space PL imaging](#7-real-space-pl-imaging)
  - [8. Check for dielectric breakdown](#8-check-for-dielectric-breakdown)
- [Module reference](#module-reference)

---

## Installation

Clone the repository and install in editable mode:

```bash
https://github.com/brandonlokesy/LANES-TMDC-Optics-Tools.git
cd LANES-TMDC-Optics-Tools
pip install -e .
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-image`

Optional (diverging colormaps used in plots):
```bash
pip install cmcrameri
```

---

## Package structure

```
LANES-TMDC-Optics-Tools/
├── constants.py   # Physical constants, material parameters (ε, thickness, exciton energies)
├── loaders.py     # DeviceGeometry, AttoCubePLVabScan, AttoCubePLScanRealSpace, image classes
├── processing.py  # Normalisation, smoothing, background subtraction, spectral conversions
├── fitting.py     # Lorentzian/Gaussian fitting, multi-peak fitting, dipole length extraction
└── plotting.py    # PL maps, spectrum plots, real-space animations, Stark shift plots
```

---

## Key workflows

### 1. Define device geometry

The `DeviceGeometry` class models the dielectric stack and computes the displacement field from gate voltages. It is required for converting raw gate voltages into a physical field axis.

**Simple monolayer:**
```python
from tmdc_optics_tools.loaders import DeviceGeometry

geom = DeviceGeometry.from_single(
    tmdc         = "WS2",
    d_hbn_top    = 30,   # nm
    d_hbn_bottom = 50,   # nm
)
```

**Heterobilayer (e.g. MoSe2/WSe2):**
```python
from tmdc_optics_tools.loaders import DeviceGeometry, StackLayer

geom = DeviceGeometry(
    tmdc_stack   = [StackLayer("MoSe2"), StackLayer("WSe2")],
    d_hbn_top    = 30,
    d_hbn_bottom = 50,
    label        = "hBN/MoSe2/WSe2/hBN",
)
```

`StackLayer` looks up monolayer thickness and dielectric constant from `constants.py` automatically; override with `d_monolayer` and `eps` if needed. Supported materials out of the box: `WS2`, `WSe2`, `MoSe2`, `MoS2`.

The dielectric constant value for the heterostructure layer (excluding hBN) can be called with `DeviceGeometry.eps_2d`, the heterostructure thickness with `DeviceGeometry.heterostructure_thickness`, the stack with `DeviceGeometry.stack_label`.

---

### 2. Load a gate-dependent PL scan

These scans are a map of the PL spectra with respect to the voltages A and B applied to the sample. Typically, we apply these voltages to the top and bottom gate to tune the electric field applied to the heterostructure. When the sample geometry is given, the electric field applied to the heterostructure is calculated.

```python
from tmdc_optics_tools.loaders import AttoCubePLVabScan

scan = AttoCubePLVabScan(
    path     = "PL_dual_gate_sweep_26_05_15_11_42_07_iter_0.csv",
    geometry = geom,   # optional — enables displacement field axis
)
print(scan)
# AttoCubePLScan — 101 sweeps × 1340 pixels
#   λ range : 850.0 – 1000.0 nm  (1.240 – 1.459 eV)
#   V_top   : -5.0 → 5.0 V
#   E_F     : -12.3 → 12.3 mV/nm
```

Key attributes after loading:

| Attribute | Description |
|---|---|
| `scan.wavelength` | Spectrometer wavelength axis (nm) |
| `scan.energy` | Photon energy axis (eV) |
| `scan.spectra` | Raw PL counts, shape `(n_pixels, n_sweeps)` |
| `scan.energy_spectra` | PL on the energy axis |
| `scan.v_top`, `scan.v_bot` | Gate voltages (V) |
| `scan.ef` | Displacement field (mV/nm), `None` if no geometry |
| `scan.power` | Excitation power (µW) |
| `scan.Ich1`, `scan.Ich2` | Leakage currents (nA) |

---

### 3. Plot a PL map

```python
from tmdc_optics_tools import plotting

plotting.set_style("paper")   # or "talk", "poster". Optional

fig, ax, mesh = plotting.plot_pl_map_Vab_scan(
    scan,
    x_axis        = "energy",      # or "wavelength"
    cmap          = "vik",
    median_kernel = 3,             # 2D median filter; set to 1 to disable
)
```

The y-axis is automatically the displacement field if a geometry was supplied, otherwise the top gate voltage.

---

### 4. Inspect a single spectrum

```python
fig, ax, line = plotting.plot_spectrum(
    scan,
    sweep_index = 50,
    x_axis      = "energy",
    normalize   = True,
)
ax.set_xlim(1.30, 1.45)
```

---

### 5. Fit a peak across a sweep

Fit a Lorentzian (or Gaussian) to a chosen spectral window at every sweep point:

```python
from tmdc_optics_tools import fitting

results = fitting.fit_scan_peak(
    scan,
    x_axis    = "energy",
    x_range   = (1.30, 1.42),   # eV — zoom into your exciton
    model     = "lorentzian",
    bg_region = (1.20, 1.28),   # eV — region used for background subtraction
)

# results is a list of FitResult, one per sweep
print(results[50])
# FitResult [lorentzian]  R²=0.9971
#   amplitude    = 4312.1 ± 38
#   center       = 1.3847 ± 0.00021
#   fwhm         = 0.00831 ± 0.00019
```

For a single spectrum:
```python
x = scan.energy
y = scan.energy_spectra[:, 50]

result = fitting.fit_lorentzian(x, y, p0=(y.max(), 1.385, 0.01))
```

Multi-peak fitting:
```python
result = fitting.fit_multi_lorentzian(
    x, y,
    n_peaks = 2,
    p0      = [(4000, 1.385, 0.01), (1500, 1.40, 0.01)],
)
```

---

### 6. Extract the excitonic dipole length

The DC Stark shift gives the out-of-plane dipole length of an interlayer exciton. `extract_dipole_length` fits a Lorentzian at every sweep point and performs a weighted linear fit E(F) = slope · F + intercept to extract d = |slope| × 1000 nm.

```python
result = fitting.extract_dipole_length(
    scan,
    x_range      = (1.30, 1.42),   # eV — spectral window for peak fitting
    model        = "lorentzian",
    Efield_range = (-8, 8),        # mV/nm — restrict to linear Stark regime
    bg_region    = (1.20, 1.28),   # eV — background subtraction
)
print(result)
# DipoleResult
#   Dipole length : 0.5821 ± 0.0034 nm  (5.82 Å)
#   Slope dE/dF   : -5.821e-04 ± 3.4e-06 eV/(mV/nm)
#   Intercept E₀  : 1.3849 ± 0.0002 eV
#   R²            : 0.9963
#   Sweep points  : 87 / 101 converged

# Plot it
fig, ax = plotting.plot_stark_shift(result, show_fit=True)
```

---

### 7. Real-space PL imaging

Load a folder of real-space PL image CSVs (one per gate voltage step) and animate them:

```python
from tmdc_optics_tools.loaders import AttoCubePLScanRealSpace

rs_scan = AttoCubePLScanRealSpace(
    path   = "./images/",
    prefix = "PLdualgatesweep_iter_",
)

fig, anim = plotting.animate_real_space_PL_map(
    rs_scan,
    var_array = scan.ef,
    var_label = "E-field",
    units     = r"mV nm$^{-1}$",
    title     = "Device A — gate sweep",
)

# Save as gif or mp4
anim.save("gate_sweep.gif", fps=5)
```

Optionally annotate the laser spot position using a reference image:

```python
from tmdc_optics_tools.loaders import AttoCubeLaserReferenceImage

laser_ref = AttoCubeLaserReferenceImage("laser_ref.csv")
print(laser_ref)
# Center: (63.4, 71.2) px  |  1/e² Radius: 8.3 px

rs_scan = AttoCubePLScanRealSpace(..., laser_ref=laser_ref)
```

---

### 8. Check for dielectric breakdown

Plot gate leakage currents and excitation power together to verify the device was not in breakdown during a sweep:

```python
fig, ax_left, ax_right = plotting.plot_current(scan)
```

---

## Module reference

### `constants`
Literature values for hBN, WS2, WSe2, MoSe2, MoS2: out-of-plane dielectric constants, monolayer thicknesses, approximate exciton energies and binding energies, and bulk/bilayer bandgaps.

### `loaders`
| Class | Purpose |
|---|---|
| `StackLayer` | One material slab in a vdW stack |
| `DeviceGeometry` | Dielectric stack model; computes ε_eff, optical thickness, displacement field |
| `AttoCubePLVabScan` | Gate-dependent PL scan from AttoCube confocal CSV |
| `AttoCubePLScanRealSpace` | Sequence of real-space PL image CSVs |
| `AttoCubeSampleImage` | White-light sample reference image |
| `AttoCubeLaserReferenceImage` | Laser spot image with fitted 1/e² radius |

### `processing`
| Function | Purpose |
|---|---|
| `normalise_peak` | Normalise each spectrum to its maximum |
| `normalise_area` | Normalise each spectrum to its integrated area |
| `subtract_background` | Subtract a constant background from a spectral region |
| `smooth_median` | Median filter (1D or 2D) |
| `smooth_savgol` | Savitzky-Golay smoothing |
| `crop` | Crop spectra and x-axis to a range |
| `wavelength_to_energy` / `energy_to_wavelength` | Unit conversion |
| `jacobian_correction_wvl2E` | Apply dλ/dE Jacobian when converting to energy axis |

### `fitting`
| Function / Class | Purpose |
|---|---|
| `fit_lorentzian` | Single Lorentzian peak fit |
| `fit_gaussian` | Single Gaussian peak fit |
| `fit_multi_lorentzian` | Sum of N Lorentzians, with automatic peak detection |
| `fit_scan_peak` | Fit a single peak at every sweep point in a scan |
| `extract_dipole_length` | DC Stark shift → weighted linear fit → dipole length |
| `FitResult` | Dataclass: params, errors, y_fit, residuals, R² |
| `DipoleResult` | Dataclass: slope, intercept, dipole length ± error, R² |

### `plotting`
| Function | Purpose |
|---|---|
| `set_style` | Apply publication-ready Matplotlib rcParams |
| `plot_pl_map_Vab_scan` | 2D PL intensity map (energy/wavelength vs gate/field) |
| `plot_spectrum` | Single spectrum from a scan |
| `plot_current` | Leakage current and power monitor |
| `plot_real_space_PL_map` | Single real-space PL image |
| `animate_real_space_PL_map` | Animated gate-dependent real-space PL |
| `plot_stark_shift` | Peak energy vs field with linear fit overlay |
| `save_figure` | Save figure to disk (png, pdf, or both) |