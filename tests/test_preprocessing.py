import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg
from src.preprocessing.normalization import normalize_dataset

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("tests/verification_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fs = 400
        self.signal = np.random.normal(0, 1, (self.fs * 10, 12))

    def test_baseline_removal(self):
        filtered = remove_baseline(self.signal, self.fs, 'bandpass', low_cut_hz=0.5, high_cut_hz=45.0, order=4)
        self.assertEqual(filtered.shape, self.signal.shape, "Baseline removal changed shape")

        plt.figure(figsize=(10, 4))
        plt.plot(self.signal[:, 0], label="Original")
        plt.plot(filtered[:, 0], label="Filtered")
        plt.legend()
        plt.savefig(self.output_dir / "baseline_sample.png")
        plt.close()

    def test_resample(self):
        resampled_up, _ = resample_ecg(self.signal, self.fs, 500)
        self.assertGreater(resampled_up.shape[0], self.signal.shape[0])
        resampled_down, _ = resample_ecg(self.signal, self.fs, 100)
        self.assertLess(resampled_down.shape[0], self.signal.shape[0])

    def test_normalization(self):
        normalized = normalize_dataset(self.signal)
        self.assertAlmostEqual(normalized.mean(), 0, places=5)
        self.assertAlmostEqual(normalized.std(), 1, places=5)

if __name__ == "__main__":
    unittest.main()