import requests, zipfile, io
import numpy as np
import scipy.io
import h5py
from pathlib import Path

HEADERS = {"LANES": "LANES-Tools/1.0"}

class Tagarelli2023Processor:
    def __init__(self, meta):
        self.meta = meta
        self.out_path = Path(f"reference_data/data/{meta['material']}__{meta['source']}.h5")

    def fetch(self):
        url = "https://zenodo.org/records/7660668/files/Figure%201.zip?download=1"
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return zipfile.ZipFile(io.BytesIO(r.content))

    def run(self):
        z = self.fetch()

        mats = {
            "E=0":    "Figure 1d/hb_0field.mat",
            "E=+300": "Figure 1d/homobilayer_ef300_power_spectral.mat",
            "E=-300": "Figure 1d/homobilayer_efm300_spectral.mat",
        }

        with h5py.File(self.out_path, "w") as hf:
            # Write metadata
            for k, v in self.meta.items():
                if k != "processor":
                    hf.attrs[k] = v

            for label, path in mats.items():
                with z.open(path) as f:
                    mat = scipy.io.loadmat(io.BytesIO(f.read()))
                    spectrum = mat["spectra"][:, 39].squeeze()
                    energy   = mat["energy"].squeeze()

                grp = hf.create_group(label)
                grp.create_dataset("energy",   data=energy)
                grp.create_dataset("spectrum", data=spectrum)
                grp.attrs["energy_unit"]   = "eV"
                grp.attrs["spectrum_unit"] = "counts"
                grp.attrs["field_mV_nm"]   = int(label.split("=")[-1].replace("+", ""))