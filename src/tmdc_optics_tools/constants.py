# tmdc_optics_tools/constants.py

from scipy.constants import h,c,e,hbar,epsilon_0

# ----- Physical constants ----- #
HC_EV_NM = (h * c / e) * 1e9   # eV·nm

# Re-exported under project names so every module takes them from one place.
EPS_0    = epsilon_0           # F/m
E_CHARGE = e                   # C

# ----- Spectroscopy types ----- #
# Controlled vocabulary for what a spectrum *is*.  Shared by the AttoCube
# loaders (where it is recorded per scan and written into exported HDF5) and by
# the reference-data subpackage, which re-exports it — one vocabulary, so a tag
# means the same thing in a lab file and in a digitised literature dataset.
# Raman is out of scope: the group measures it on a separate instrument whose
# export this package does not read.
SPECTROSCOPY_TYPES = {
    "PL":   "Photoluminescence",
    "R":    "Reflectance",
    "RC":   "Reflectance contrast",
    "T":    "Transmission",
    "A":    "Absorption",
    "TRPL": "Time-resolved photoluminescence",
}

# Axis label for the measured signal, per spectroscopy type: (name, unit).
# The unit is a separate field, and empty for the dimensionless ratios, which is
# why this is not simply "<name> (counts)" everywhere.

SIGNAL_LABELS = {
    "PL":   ("PL intensity",          "counts"),
    "R":    ("Reflected intensity",   "counts"), 
    "RC":   (r"$\Delta R/R_0$",       ""),
    "T":    ("Transmitted intensity", "counts"),
    "A":    ("Absorbance",            ""),
    "TRPL": ("PL intensity",          "counts"),
}

# ----- Spectral x-axis vocabulary ----- #
# The two orderings a spectrum can be served on, as (name, unit) — the same shape
# as SIGNAL_LABELS above, so an axis label composes the same way.  The keys are
# what every x_axis= argument in the package accepts.
X_AXES = {
    "energy":     ("Energy",     "eV"),
    "wavelength": ("Wavelength", "nm"),
}


def _x_axis_name_unit(x_axis: str, what: str = None) -> tuple:
    """
    Return ``(name, unit)`` for a spectral axis, refusing anything else.

    The one check behind every ``x_axis=`` in the package.  Each caller still
    picks its own arrays, but none of them decides what the vocabulary is, so an
    unrecognised value cannot be read as one of the two by the branch below it.

    Parameters
    ----------
    x_axis : str
        Axis name to check, a key of :data:`X_AXES`.
    what : str, optional
        Calling function, spelled ``"pixel_slice()"``, used to prefix the
        message.  Omitted where the caller's other messages are unprefixed.

    Returns
    -------
    tuple of (str, str)
        Quantity name and unit, e.g. ``("Energy", "eV")``.

    Raises
    ------
    ValueError
        If *x_axis* is not a key of :data:`X_AXES`.
    """
    try:
        return X_AXES[x_axis]
    except (KeyError, TypeError):       # TypeError: an unhashable x_axis
        # Derived from the table, so a key added there cannot go unmentioned here.
        raise ValueError(
            f"{what + ': ' if what else ''}x_axis must be "
            f"{' or '.join(repr(key) for key in X_AXES)}, got {x_axis!r}."
        ) from None


# ----- Material dielectric constants ----- #
# Sources cited as comments where known
EPS_HBN   = 3.9    # hBN out-of-plane, Laturia et al. 2018
EPS_WS2   = 6.1    # WS2, npj 2D Materials and Applications volume 2, Article number: 6 (2018) 
EPS_WSE2  = 7.4    # WSe2
EPS_MOSE2 = 7.2    # MoSe2
EPS_MOS2  = 6.2    # MoS2

# ----- TMDC monolayer thickness (nm) ----- #
T_MONOLAYER = {
    "WS2"   : 0.65,
    "WSe2"  : 0.65,
    "MoSe2" : 0.65,
    "MoS2"  : 0.65,
}

# ----- TMDC dielectric constants lookup ----- #
EPS_TMDC = {
    "WS2"   : EPS_WS2,
    "WSe2"  : EPS_WSE2,
    "MoSe2" : EPS_MOSE2,
    "MoS2"  : EPS_MOS2,
    "HS" :    7.5,   # "HS" = heterostructure; use this value as an approximation
}

# ----- TMDC approximate exciton energies (eV) ----- #
# Rough literature values for encapsulated monolayers at low temperature;
# useful as starting guesses for fits. Update as needed.
EXCITON_ENERGY = {
    "WS2"   : {"XA0": 2.02, # Scientific Reports 5, 9218 (2015), Nature volume 513, pages 214–218 (2014)
               "XB0": 2.41, # Phys. Rev. Lett. 113, 076802 (2014). Value at 5K. 
               },
    "WSe2"  : {"XA0": 1.75, # Nature Nanotechnology 8, 634–638 (2013)
                },
    "MoSe2" : {"XA0": 1.66, # Nature Communications 4, Article number: 1474 (2013). Value at 20 K
               },
    "MoS2"  : {"XA0": 1.86, # Phys. Rev. B 94, 075440 (2016), Phys. Rev. Lett. 105, 136805 (2010)
               "XB0" : 2.00, # Phys. Rev. B 94, 075440 (2016).
               },
}

INTERLAYER_EXCITON_ENERGY = {
    "WSe2/MoS2" : 1.55, # Proc. Natl. Acad. Sci. U.S.A. 111 (17) 6198-6202 (2014)
}

for heterostructure in INTERLAYER_EXCITON_ENERGY.copy().keys():
    layer1, layer2 = heterostructure.split("/")
    INTERLAYER_EXCITON_ENERGY[layer2 + "/" + layer1] = INTERLAYER_EXCITON_ENERGY[heterostructure]

BINDING_ENERGY = {
    "MoS2" : 0.310, # Phys. Rev. B 94, 075440 (2016)
    "WSe2" : 0.5, # Nano Lett. 2015, 15, 10, 6494–6500
    "WS2" : 0.32, # Phys. Rev. Lett. 113, 076802 (2014). Value at 5K. 
    "MoSe2" : 0.55, # Nat Mat. 13, 1091–1095 (2014). Value at 5 K.
}

# BANDGAP_ENERGY_ML = {
#     "WS2"   : 2.15,
#     "WSe2"  : 1.80,
#     "MoSe2" : 1.75,
#     "MoS2"  : 2.16, 
# }

BANDGAP_ENERGY_BL = {
    "MoS2" : 1.60, # Phys. Rev. Lett. 105, 136805 (2010)
}

BANDGAP_ENERGY_BULK = {
    "MoS2" : 1.29,  # Phys. Rev. Lett. 105, 136805 (2010).
    "WS2"  : 1.4,  # J. Phys. Chem. 1982, 86, 4, 463–467
    "MoSe2" : 1.1, #  J. Phys. Chem. 86, 463–467 (1982)
    "WSe2" : 1.2,  #  J. Phys. Chem. 86, 463–467 (1982)
}