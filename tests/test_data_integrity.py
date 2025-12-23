import unittest
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class TestDataIntegrity(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("tests/verification_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = ['ptbxl', 'sami_trop', 'code15']
        self.base_path = Path("data/official_wfdb")
        self.processed_path = Path("data/processed/2d_images")
        self.metadata_path = Path("data/processed/metadata")

    def test_counts(self):
        for ds in self.datasets:
            ds_dir = self.base_path / ds
            hea_count = 0
            for root, dirs, files in os.walk(ds_dir):
                hea_count += len([f for f in files if f.endswith('.hea')])
            self.assertGreater(hea_count, 0, f"{ds} WFDB count 0")
            
            img_dir = self.processed_path / ds
            img_count = len(list(img_dir.glob("*.npy")))
            self.assertEqual(img_count, hea_count, f"{ds} image count mismatch")

    def test_metadata(self):
        for ds in self.datasets:
            metadata_file = self.metadata_path / f"{ds}_processed.csv"
            self.assertTrue(metadata_file.exists(), f"{ds} metadata missing")
            df = pd.read_csv(metadata_file)
            self.assertGreater(len(df), 0, f"{ds} metadata empty")
            self.assertIn('label', df.columns, f"{ds} metadata no label")

    def test_comparisons(self):
        for ds in self.datasets:
            img_dir = self.processed_path / ds
            sample_img = next(img_dir.glob("*.npy"), None)
            self.assertIsNotNone(sample_img, f"No sample image in {ds}")
            img = np.load(sample_img)
            self.assertEqual(img.shape, (3, 24, 2048), f"{ds} image wrong shape")
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img[0], cmap='gray')
            axs[1].imshow(img[1], cmap='gray')
            axs[2].imshow(img[2], cmap='gray')
            plt.savefig(self.output_dir / f"{ds}_comparison_sample.png")
            plt.close()

if __name__ == "__main__":
    unittest.main()