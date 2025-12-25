import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from src.preprocessing.image_embedding import ecg_to_contour_image

class TestImageEmbedding(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("tests/verification_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fs = 500
        self.duration = 10
        self.target_width = 2048
        self.clip_range = (-3, 3)

    def test_embedding_shape(self):
        signal = np.random.normal(0, 1, (self.fs * self.duration, 12))
        img = ecg_to_contour_image(signal, target_width=self.target_width, clip_range=self.clip_range)
        self.assertEqual(img.shape, (3, 24, self.target_width), "Wrong image shape")

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img[0], cmap='viridis')
        axs[1].imshow(img[1], cmap='viridis')
        axs[2].imshow(img[2], cmap='viridis')
        plt.savefig(self.output_dir / "embedding_sample.png")
        plt.close()

    def test_clipping(self):
        try:
            extreme_signal = np.random.normal(0, 10, (self.fs * self.duration, 12))
            clipped_signal = np.clip(extreme_signal, self.clip_range[0], self.clip_range[1])
            img_extreme = ecg_to_contour_image(clipped_signal, target_width=self.target_width, clip_range=self.clip_range)
            self.assertEqual(img_extreme.shape, (3, 24, self.target_width))
            self.assertTrue(np.all(clipped_signal >= self.clip_range[0]), "Signal not clipped below")
            self.assertTrue(np.all(clipped_signal <= self.clip_range[1]), "Signal not clipped above")
        except Exception as e:
            self.fail(f"Clipping test failed: {e}")

if __name__ == "__main__":
    unittest.main()