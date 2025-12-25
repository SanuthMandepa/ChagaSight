import unittest
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wfdb

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import normalize_dataset
from src.preprocessing.image_embedding import ecg_to_contour_image

class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("tests/verification_outputs/pipeline")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = ['ptbxl', 'sami_trop', 'code15']

    def test_pipeline(self):
        samples = {'ptbxl': '1', 'sami_trop': '4991', 'code15': '14'}
        for ds in self.datasets:
            sample_id = samples[ds]
            wfdb_path = self.get_wfdb_path(ds, sample_id)
            signal, fields = wfdb.rdsamp(str(wfdb_path))  # Correct: returns signal and dict
            fs = fields['fs']  # Extract fs properly

            # Baseline removal
            if ds == 'ptbxl':
                filtered = remove_baseline(signal, fs, 'bandpass', low_cut_hz=0.5, high_cut_hz=45.0, order=4)
            elif ds == 'sami_trop':
                filtered = remove_baseline(signal, fs, 'moving_average', window_seconds=0.2)
            else:
                filtered = signal  # CODE-15: no baseline

            # Resample to 500 Hz
            signal_500, _ = resample_ecg(filtered, fs, 500)
            signal_500 = pad_or_trim(signal_500, 5000)

            # Normalize only for PTB-XL and SaMi-Trop (CODE-15 already filtered)
            if ds != 'code15':
                signal_500_norm = normalize_dataset(signal_500)
            else:
                signal_500_norm = signal_500  # No z-score for CODE-15

            # Z-score check (only for datasets that normalize)
            if ds != 'code15':
                mean_per_lead = np.mean(signal_500_norm, axis=0)
                std_per_lead = np.std(signal_500_norm, axis=0)
                self.assertTrue(np.allclose(mean_per_lead, 0, atol=1e-5), f"{ds} mean not ~0")
                self.assertTrue(np.allclose(std_per_lead, 1, atol=1e-5), f"{ds} std not ~1")

            # Generate image
            img = ecg_to_contour_image(signal_500_norm, 2048, (-3, 3))
            self.assertEqual(img.shape, (3, 24, 2048), f"{ds} image shape wrong")

            # Save sample plot
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i in range(3):
                axs[i].imshow(img[i], cmap='gray', aspect='auto')
                axs[i].set_title(f"Channel {i+1}")
            plt.suptitle(f"{ds.upper()} — Full Pipeline Test")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{ds}_full_pipeline_sample.png")
            plt.close()

    def get_wfdb_path(self, ds, id_val: str) -> Path:
        if ds == 'ptbxl':
            numeric_id = int(id_val)
            subfolder = f"{numeric_id // 1000:05d}"
            candidates = [
                Path(f"data/official_wfdb/{ds}/records500/{subfolder}/{numeric_id:05d}_hr"),
                Path(f"data/official_wfdb/{ds}/records100/{subfolder}/{numeric_id:05d}_lr")
            ]
            for path in candidates:
                if path.with_suffix('.hea').exists():
                    return path
            raise FileNotFoundError(f"No WFDB for {ds} ID {id_val}")
        else:
            return Path(f"data/official_wfdb/{ds}/{id_val}")

if __name__ == "__main__":
    unittest.main()