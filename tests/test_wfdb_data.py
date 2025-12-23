import unittest
import os
import wfdb
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class TestWFDBData(unittest.TestCase):
    def setUp(self):
        self.base_path = Path("data/official_wfdb")
        self.output_dir = Path("tests/verification_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = ['ptbxl', 'sami_trop', 'code15']

    def test_folder_existence(self):
        for ds in self.datasets:
            ds_dir = self.base_path / ds
            self.assertTrue(ds_dir.exists(), f"{ds} WFDB folder missing")

    def test_record_counts(self):
        for ds in self.datasets:
            ds_dir = self.base_path / ds
            hea_count = 0
            for root, dirs, files in os.walk(ds_dir):
                hea_count += len([f for f in files if f.endswith('.hea')])
            # Adjust for your subset
            expected = {'ptbxl': 21799, 'sami_trop': 1631, 'code15': 39798}
            self.assertEqual(hea_count, expected.get(ds, 0), f"{ds} record count mismatch")

    def test_sample_header(self):
        for ds in self.datasets:
            ds_dir = self.base_path / ds
            sample_rec = None
            for root, dirs, files in os.walk(ds_dir):
                hea_files = [f for f in files if f.endswith('.hea')]
                if hea_files:
                    sample_rec = Path(root) / hea_files[0].replace('.hea', '')
                    break
            self.assertIsNotNone(sample_rec, f"No sample record in {ds}")
            header = wfdb.rdheader(str(sample_rec))
            chagas = [line for line in header.comments if 'Chagas label' in line]
            self.assertTrue(chagas, f"No Chagas label in {ds} header")

    def test_sample_signal(self):
        for ds in self.datasets:
            ds_dir = self.base_path / ds
            sample_rec = None
            for root, dirs, files in os.walk(ds_dir):
                hea_files = [f for f in files if f.endswith('.hea')]
                if hea_files:
                    sample_rec = Path(root) / hea_files[0].replace('.hea', '')
                    break
            self.assertIsNotNone(sample_rec, f"No sample record in {ds}")
            signal, fields = wfdb.rdsamp(str(sample_rec))
            self.assertEqual(signal.shape[1], 12, f"{ds} sample not 12 leads")
            plt.figure(figsize=(10, 4))
            plt.plot(signal[:, 0])
            plt.title(f"{ds.upper()} WFDB Sample (Lead I)")
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.savefig(self.output_dir / f"{ds}_wfdb_sample.png")
            plt.close()

if __name__ == "__main__":
    unittest.main()