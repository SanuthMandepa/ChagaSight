import unittest
import numpy as np
from pathlib import Path

from src.preprocessing.soft_labels import get_chagas_label, is_confident_label

class TestSoftLabels(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("tests/verification_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_ptbxl_labels(self):
        metadata = {'normal_ecg': True}
        label = get_chagas_label(metadata, 'ptbxl')
        self.assertEqual(label, 0.0)
        confident = is_confident_label(label, 'ptbxl')
        self.assertTrue(confident)

    def test_sami_trop_labels(self):
        metadata = {'chagas': True}
        label = get_chagas_label(metadata, 'sami_trop')
        self.assertEqual(label, 1.0)
        confident = is_confident_label(label, 'sami_trop')
        self.assertTrue(confident)

    def test_code15_labels(self):
        # Positive
        metadata = {'chagas': True}
        label = get_chagas_label(metadata, 'code15')
        self.assertEqual(label, 0.8)
        confident = is_confident_label(label, 'code15')
        self.assertTrue(confident)

        # Negative
        metadata = {'chagas': False}
        label = get_chagas_label(metadata, 'code15')
        self.assertEqual(label, 0.2)
        confident = is_confident_label(label, 'code15')
        self.assertTrue(confident)  # 0.2 < 0.3 → confident

        # Uncertain (e.g., 0.5)
        metadata = {'chagas': 0.5}
        label = get_chagas_label(metadata, 'code15')
        self.assertEqual(label, 0.5)
        confident = is_confident_label(label, 'code15')
        self.assertFalse(confident)  # 0.5 not <0.3 or >0.7

    def test_batch_labels(self):
        labels = np.array([0.1, 0.25, 0.8, 0.9, 0.5])
        confident = is_confident_label(labels, 'code15')
        expected = np.array([True, True, True, True, False])  # <0.3 or >0.7
        np.testing.assert_array_equal(confident, expected)

if __name__ == "__main__":
    unittest.main()