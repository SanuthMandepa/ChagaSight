import unittest
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wfdb

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg
from src.preprocessing.normalization import normalize_dataset
from src.preprocessing.image_embedding import ecg_to_contour_image

class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("tests/verification_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = ['ptbxl', 'sami_trop', 'code15']
        self.sample_ids = {'ptbxl': '00001_hr', 'sami_trop': '294669', 'code15': '113'}

    def test_full_pipeline(self):
        for ds in self.datasets:
            wfdb_dir = Path("data/official_wfdb") / ds
            sample_id = self.sample_ids[ds]
            rec_path = wfdb_dir / sample_id if ds != 'ptbxl' else wfdb_dir / '00000' / sample_id
            self.assertTrue(rec_path.with_suffix('.hea').exists(), f"Sample missing for {ds}")
            
            signal, fields = wfdb.rdsamp(str(rec_path))
            fs = fields['fs']
            
            # Pipeline
            signal = remove_baseline(signal, fs=fs)
            signal_500, _ = resample_ecg(signal, fs, 500)
            signal_norm = normalize_dataset(signal_500)
            img = ecg_to_contour_image(signal_norm, 2048, (-3,3))
            
            self.assertEqual(img.shape, (3, 24, 2048))
            
            # Plot
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img[0], cmap='gray')
            axs[1].imshow(img[1], cmap='gray')
            axs[2].imshow(img[2], cmap='gray')
            plt.savefig(self.output_dir / f"{ds}_pipeline_sample.png")
            plt.close()

if __name__ == "__main__":
    unittest.main()