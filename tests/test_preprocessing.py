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
        self.duration = 10
        self.t = np.linspace(0, self.duration, int(self.fs * self.duration))
        self.signal = np.sin(2 * np.pi * 5 * self.t) + 0.5 * np.sin(2 * np.pi * 0.5 * self.t) + np.random.normal(0, 0.1, len(self.t))
        self.signal = np.repeat(self.signal[:, np.newaxis], 12, axis=1)

    def test_baseline_removal(self):
        corrected = remove_baseline(self.signal, self.fs, 'moving_average')
        self.assertEqual(corrected.shape, self.signal.shape)
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.signal[:, 0], label='Raw')
        plt.plot(self.t, corrected[:, 0], label='Corrected')
        plt.legend()
        plt.title('Baseline Removal Test')
        plt.grid(True)
        plt.savefig(self.output_dir / "baseline_test.png")
        plt.close()

    def test_resample(self):
        resampled, _ = resample_ecg(self.signal, self.fs, 500)
        self.assertGreater(resampled.shape[0], self.signal.shape[0])
        resampled_down, _ = resample_ecg(self.signal, self.fs, 100)
        self.assertLess(resampled_down.shape[0], self.signal.shape[0])

    def test_normalization(self):
        normalized = normalize_dataset(self.signal)
        self.assertAlmostEqual(normalized.mean(), 0, places=5)
        self.assertAlmostEqual(normalized.std(), 1, places=5)

if __name__ == "__main__":
    unittest.main()