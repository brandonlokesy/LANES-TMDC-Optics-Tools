import io
import pickle
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from ...constants import HC_EV_NM

from .processor import Processor

class Alexeev2019ProcessorMoSe2(Processor):
    def run(self):

        url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-0986-9/MediaObjects/41586_2019_986_MOESM4_ESM.xlsx"
        print(f"Fetching data from {url} from Nature.")
        
        df = pd.read_excel(url, sheet_name= "(a) Normalised PL", header = [0,1])

        energy = df[("Energy, eV" ,'Unnamed: 0_level_1')]
        spectra_MoSe2 = df['Normalised PL intensity', 'MoSe2']

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            hf.create_dataset("energy", data = energy)
            hf.create_dataset("spectra", data = spectra_MoSe2)


        print(f"  → Saved to {self.out_path}")

class Alexeev2019ProcessorWS2(Processor):
    def run(self):

        url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-0986-9/MediaObjects/41586_2019_986_MOESM2_ESM.xlsx"
        print(f"Fetching data from {url} from Nature. "  )
        
        df = pd.read_excel(url, sheet_name="(a) Full Spectra", header = [0,1])

        energy = df[("Energy,eV" ,'Unnamed: 0_level_1')]
        spectra_WS2 = df[('Intensity, cts/s', 'WS2')]

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            hf.create_dataset("energy", data = energy)
            hf.create_dataset("spectra", data = spectra_WS2)

        print(f"  → Saved to {self.out_path}")
