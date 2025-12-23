import unittest
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import wfdb

class TestRawData(unittest.TestCase):
    def setUp(self):
        self.raw_dir = Path("data/raw")
        self.output_dir = Path("tests/verification_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = ['ptbxl', 'sami_trop', 'code15']

    def test_folder_existence(self):
        for ds in self.datasets:
            ds_dir = self.raw_dir / ds
            self.assertTrue(ds_dir.exists(), f"{ds} raw folder missing")

    def test_file_counts(self):
        for ds in self.datasets:
            ds_dir = self.raw_dir / ds
            files = list(ds_dir.glob("*"))
            self.assertGreater(len(files), 0, f"No files in {ds} raw folder")

    def test_sample_signal(self):
        for ds in self.datasets:
            ds_dir = self.raw_dir / ds
            sample_file = None

            if ds == 'ptbxl':
                # PTB-XL: .hea in subfolders
                for root, _, files in os.walk(ds_dir):
                    hea_files = [f for f in files if f.endswith('.hea')]
                    if hea_files:
                        sample_file = Path(root) / hea_files[0]
                        break
            elif ds == 'sami_trop':
                sample_file = next(ds_dir.glob("exams.hdf5"), None)
            elif ds == 'code15':
                sample_file = next(ds_dir.glob("exams_part*.hdf5"), None)

            self.assertIsNotNone(sample_file, f"No sample file found in {ds}")

            # Load signal
            signal = None
            if ds == 'ptbxl':
                sample_rec = sample_file.with_suffix('')
                signal = wfdb.rdsamp(str(sample_rec))[0]
            else:
                with h5py.File(sample_file, 'r') as f:
                    signal = f['tracings'][0]

            self.assertIsNotNone(signal, f"Failed to load signal for {ds}")
            self.assertEqual(signal.shape[1], 12, f"{ds} sample not 12 leads")

            plt.figure(figsize=(10, 4))
            plt.plot(signal[:, 0])
            plt.title(f"{ds.upper()} Raw Sample (Lead I)")
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.savefig(self.output_dir / f"{ds}_raw_sample.png")
            plt.close()

if __name__ == "__main__":
    unittest.main()