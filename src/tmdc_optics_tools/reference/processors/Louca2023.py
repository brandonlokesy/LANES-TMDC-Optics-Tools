import pandas as pd
import h5py
from .processor import Processor

class Louca2023Processor(Processor):
    def run(self):
        url = "https://archive.materialscloud.org/records/q0deg-ag137/files/Fig1c.xlsx?download=1"

        df = pd.read_csv(url)

        energy = df['Energy (eV)']
        spectra = df['RC']

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            hf.create_dataset('energy', data = energy)
            hf.create_dataset('spectra', data = spectra)