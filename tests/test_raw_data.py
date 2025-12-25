import unittest
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import wfdb
import pandas as pd

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

    def test_sample_load(self):
        samples = {'ptbxl': '1', 'sami_trop': '4991', 'code15': '14'}
        for ds, sample_id in samples.items():
            try:
                if ds == 'ptbxl':
                    df = pd.read_csv(self.raw_dir / ds / "ptbxl_database.csv")
                    row = df[df['ecg_id'] == int(sample_id)].iloc[0]
                    path = self.raw_dir / ds / row['filename_hr']
                    signal, _ = wfdb.rdsamp(str(path))
                elif ds == 'sami_trop':
                    df = pd.read_csv(self.raw_dir / ds / "exams.csv")
                    h5_idx = df[df['exam_id'] == int(sample_id)].index[0]
                    with h5py.File(self.raw_dir / ds / "exams.hdf5", 'r') as h5:
                        signal = h5['tracings'][h5_idx].astype(np.float32)
                elif ds == 'code15':
                    df = pd.read_csv(self.raw_dir / ds / "exams.csv")
                    row = df[df['exam_id'] == int(sample_id)].iloc[0]
                    trace_file = row['trace_file']
                    with h5py.File(self.raw_dir / ds / trace_file, 'r') as h5:
                        idx = np.where(h5['exam_id'][:] == int(sample_id))[0][0]
                        signal = h5['tracings'][idx].astype(np.float32)

                self.assertEqual(signal.shape[1], 12, f"{ds} sample not 12 leads")

                plt.figure(figsize=(10, 4))
                plt.plot(signal[:, 0])
                plt.title(f"{ds.upper()} Raw Sample (Lead I)")
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")
                plt.grid(True)
                plt.savefig(self.output_dir / f"{ds}_raw_sample.png")
                plt.close()
            except Exception as e:
                self.fail(f"Raw data test failed for {ds}: {e}")

if __name__ == "__main__":
    unittest.main()