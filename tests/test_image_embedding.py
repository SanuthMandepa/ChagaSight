import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

from src.preprocessing.image_embedding import ecg_to_contour_image

class TestImageEmbedding(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("tests/verification_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fs = 500
        self.duration = 10
        self.target_width = 2048
        self.clip_range = (-3.0, 3.0)

    def create_test_signal(self, shape=(5000, 12)):
        t = np.linspace(0, self.duration, shape[0])
        base = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.1, len(t))
        signal = np.repeat(base[:, np.newaxis], 12, axis=1)
        return signal

    def test_embedding_shape(self):
        signal = self.create_test_signal()
        img = ecg_to_contour_image(signal, target_width=self.target_width, clip_range=self.clip_range)
        self.assertEqual(img.shape, (3, 24, self.target_width), "Output shape mismatch")

    def test_embedding_clip_range(self):
        signal = self.create_test_signal()
        img = ecg_to_contour_image(signal, target_width=self.target_width, clip_range=self.clip_range)
        self.assertTrue(np.all(img >= self.clip_range[0]), "Values below clip min")
        self.assertTrue(np.all(img <= self.clip_range[1]), "Values above clip max")

    def test_embedding_visual(self):
        signal = self.create_test_signal()
        img = ecg_to_contour_image(signal, target_width=self.target_width, clip_range=self.clip_range)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img[0], cmap='gray')
        axs[0].set_title('RA Contour')
        axs[1].imshow(img[1], cmap='gray')
        axs[1].set_title('LA Contour')
        axs[2].imshow(img[2], cmap='gray')
        axs[2].set_title('LL Contour')
        plt.suptitle('2D Contour Embedding Test')
        plt.tight_layout()
        plt.savefig(self.output_dir / "image_embedding_test.png")
        plt.close()

    def test_edge_cases(self):
        # All-zero
        zero_signal = np.zeros((5000, 12))
        img_zero = ecg_to_contour_image(zero_signal, target_width=self.target_width, clip_range=self.clip_range)
        self.assertEqual(img_zero.shape, (3, 24, self.target_width))
        # For all-zero, expect near-zero image
        self.assertTrue(np.all(np.abs(img_zero) <= 1e-6), "All-zero input not handled correctly")

        # All-constant
        const_signal = np.ones((5000, 12)) * 10
        img_const = ecg_to_contour_image(const_signal, target_width=self.target_width, clip_range=self.clip_range)
        self.assertEqual(img_const.shape, (3, 24, self.target_width))
        # Constant should produce near-zero contours
        self.assertTrue(np.all(np.abs(img_const) <= 1e-6), "All-constant input not handled correctly")

        # Extreme values
        extreme_signal = np.random.normal(0, 10, (5000, 12))
        img_extreme = ecg_to_contour_image(extreme_signal, target_width=self.target_width, clip_range=self.clip_range)
        self.assertEqual(img_extreme.shape, (3, 24, self.target_width))
        self.assertTrue(np.all(img_extreme >= self.clip_range[0]), "Extreme values not clipped below")
        self.assertTrue(np.all(img_extreme <= self.clip_range[1]), "Extreme values not clipped above")

if __name__ == "__main__":
    unittest.main()