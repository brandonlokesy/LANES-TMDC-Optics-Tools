# Loaders

Data loaders for device geometry and AttoCube confocal scans — spectral sweeps
over any scanned parameter, time-resolved PL, real-space image sequences, and
single spectra or images.

The two sweep classes share a private base and differ only in their measured
axis, which is the whole reason they are separate:

| Class | Axis | Export layout | Input |
|---|---|---|---|
| `AttoCubeSpectralSweep` | wavelength / energy | `[Par, Wavelength, ExpROI1, ExpROI2]` | one `.csv` or `.h5` |
| `AttoCubeTRPLSweep` | time (ns) | `[Par, Wavelength, Exp]` — the "Wavelength" column holds **time** | one `.csv`, a **directory**, or `.h5` |

A single decay is simply `n_sweeps == 1`, so that is not what divides them.
`energy = hc/t` is meaningless and divides by zero at `t = 0`, so the energy
machinery — `energy`, `energy_spectra`, `apply_jacobian` — does not exist on a
TRPL sweep, and neither does `spectra`: handing one to a spectral plot raises
rather than drawing time as if it were wavelength. Each class rejects the other's
files by name.

::: tmdc_optics_tools.loaders
