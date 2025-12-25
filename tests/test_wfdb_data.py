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
            self.assertTrue((self.base_path / ds).exists(), f"{ds} WFDB folder missing")

    def test_record_count(self):
        for ds in self.datasets:
            records = self.find_records(self.base_path / ds)
            self.assertGreater(len(records), 0, f"No WFDB records in {ds}")

    def test_sample_load(self):
        samples = {'ptbxl': '1', 'sami_trop': '4991', 'code15': '14'}
        for ds, sample_id in samples.items():
            try:
                path = self.get_sample_path(ds, sample_id)
                signal, _ = wfdb.rdsamp(str(path))
                self.assertEqual(signal.shape[1], 12, f"{ds} WFDB sample not 12 leads")

                plt.figure(figsize=(10, 4))
                plt.plot(signal[:, 0])
                plt.title(f"{ds.upper()} WFDB Sample (Lead I)")
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")
                plt.grid(True)
                plt.savefig(self.output_dir / f"{ds}_wfdb_sample.png")
                plt.close()
            except Exception as e:
                self.fail(f"WFDB test failed for {ds}: {e}")

    def find_records(self, dir_path: Path) -> list:
        paths = []
        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.endswith('.hea'):
                    paths.append(Path(root) / f.replace('.hea', ''))
        return paths

    def get_sample_path(self, ds, id_val: str) -> Path:
        if ds == 'ptbxl':
            numeric_id = int(id_val)
            subfolder = f"{numeric_id // 1000:05d}"
            candidates = [
                self.base_path / ds / f"records500/{subfolder}/{numeric_id:05d}_hr",
                self.base_path / ds / f"records100/{subfolder}/{numeric_id:05d}_lr"
            ]
            for path in candidates:
                if path.with_suffix('.hea').exists():
                    return path
            raise FileNotFoundError(f"No WFDB for {ds} ID {id_val}")
        else:
            return self.base_path / ds / id_val

if __name__ == "__main__":
    unittest.main()