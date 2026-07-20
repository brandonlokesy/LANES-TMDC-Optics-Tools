from .processor import Processor
import h5py
import scipy.io
import io
import numpy as np

FILES = {
    "WSe2": "fang_RT_moire/Fig2/Fig2a/R16C_superK_L532_0.1_uW_1024MoS2-WSe2-PDMS-sample8_WSe2_free_50um_150lmm_20s.sif.mat",
    "MoS2": "fang_RT_moire/Fig2/Fig2a/R15C_superK_L532_0.1_uW_0523MoS2-WSe2-PDMS-sample4_MoS2_free_50um_150lmm_20s.sif.mat",
}


class Lin2024ProcessorWSe2(Processor):
    def run(self):
        print(f"Fetching zip from Zenodo ({self.meta['dataset_doi']})...")
        z= self._fetch_zip(
            "https://zenodo.org/records/13629284/files/fang_RT_moire.zip?download=1"
        )

        filepath = FILES["WSe2"]

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            print("Processing file...")

            with z.open(filepath) as f:
                mat = scipy.io.loadmat(io.BytesIO(f.read()))

            energy = mat['pl_eV'].squeeze().astype(np.float64)
            spectra = mat["RC"].squeeze().astype(np.float64)

            hf.create_dataset('energy', data = energy)
            hf.create_dataset('spectra', data=spectra)

        print(f"  → Saved to {self.out_path}")

class Lin2024ProcessorMoS2(Processor):
    def run(self):
        print(f"Fetching zip from Zenodo ({self.meta['dataset_doi']})...")
        z= self._fetch_zip(
            "https://zenodo.org/records/13629284/files/fang_RT_moire.zip?download=1"
        )

        filepath = FILES["MoS2"]

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            print("Processing file...")

            with z.open(filepath) as f:
                mat = scipy.io.loadmat(io.BytesIO(f.read()))

            energy = mat['pl_eV'].squeeze().astype(np.float64)
            spectra = mat["RC"].squeeze().astype(np.float64)

            hf.create_dataset('energy', data = energy)
            hf.create_dataset('spectra', data=spectra)

        print(f"  → Saved to {self.out_path}")


                
