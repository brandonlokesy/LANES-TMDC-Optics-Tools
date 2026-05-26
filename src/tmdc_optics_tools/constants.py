# tmdc_optics_tools/constants.py

from scipy.constants import h,c,e,hbar

# ----- Physical constants ----- #
HC_EV_NM = (h * c / e) * 1e9   # eV·nm

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
    "WSe2"  : {"XA0": 1.75, # Nature Nanotechnology volume 8, pages 634–638 (2013)
                },
    "MoSe2" : {"XA0": 1.66, # Nature Communications volume 4, Article number: 1474 (2013). Value at 20 K
               },
    "MoS2"  : {"XA0": 1.86, # Phys. Rev. B 94, 075440 (2016), Phys. Rev. Lett. 105, 136805 (2010)
               "XB0" : 2.00, # Phys. Rev. B 94, 075440 (2016).
               },
}

INTERLAYER_EXCITON_ENERGY = {
    "WSe2/MoS2" : 1.55, # Proc. Natl. Acad. Sci. U.S.A. 111 (17) 6198-6202 (2014)
}

for heterostructure in INTERLAYER_EXCITON_ENERGY.keys():
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