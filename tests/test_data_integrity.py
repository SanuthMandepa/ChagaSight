import unittest
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

class TestDataIntegrity(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("tests/verification_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = ['ptbxl', 'sami_trop', 'code15']
        self.base_path = Path("data/official_wfdb")
        self.processed_100 = Path("data/processed/1d_signals_100hz")
        self.processed_img = Path("data/processed/2d_images")

    def test_folders_exist(self):
        for ds in self.datasets:
            self.assertTrue((self.base_path / ds).exists(), f"{ds} WFDB folder missing")
            self.assertTrue((self.processed_100 / ds).exists(), f"{ds} 100Hz folder missing")
            self.assertTrue((self.processed_img / ds).exists(), f"{ds} images folder missing")

    def test_record_count(self):
        for ds in self.datasets:
            wfdb_count = self.count_wfdb_records(self.base_path / ds, ds)  # Pass ds
            fm_count = len(list((self.processed_100 / ds).glob("*.npy")))
            img_count = len(list((self.processed_img / ds).glob("*_img.npy")))

            print(f"{ds.upper()}: WFDB={wfdb_count} | 100Hz FM={fm_count} | Images={img_count}")
            self.assertEqual(fm_count, wfdb_count, f"{ds} 100Hz FM count mismatch with WFDB")
            self.assertEqual(img_count, wfdb_count, f"{ds} image count mismatch with WFDB")

    def test_sample_shapes_and_visualization(self):
        samples = {'ptbxl': '1', 'sami_trop': '4991', 'code15': '14'}  # Valid IDs
        for ds, sid in samples.items():
            fm_path = self.processed_100 / ds / f"{sid}.npy"
            img_path = self.processed_img / ds / f"{sid}_img.npy"

            self.assertTrue(fm_path.exists(), f"Missing 100Hz signal: {fm_path}")
            self.assertTrue(img_path.exists(), f"Missing image: {img_path}")

            fm_signal = np.load(fm_path)
            img = np.load(img_path)

            self.assertEqual(fm_signal.shape, (1000, 12), f"{ds} 100Hz wrong shape")
            self.assertEqual(img.shape, (3, 24, 2048), f"{ds} image wrong shape")

            # Visualize contour image
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i in range(3):
                axs[i].imshow(img[i], cmap='gray', aspect='auto')
                axs[i].set_title(f"Channel {i+1}")
                axs[i].axis('off')
            plt.suptitle(f"{ds.upper()} — Contour Image (ID: {sid})")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{ds}_contour_sample.png", dpi=150)
            plt.close()

    def count_wfdb_records(self, dir_path: Path, dataset: str) -> int:
        """Count .hea files. For PTB-XL, only count _hr to avoid double-counting."""
        count = 0
        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.endswith('.hea'):
                    if dataset == 'ptbxl':
                        if '_hr' in f:  # Only count high-res
                            count += 1
                    else:
                        count += 1
        return count

if __name__ == "__main__":
    unittest.main(verbosity=2)