import unittest
import numpy as np

from src.preprocessing.soft_labels import get_chagas_label, is_confident_label

class TestSoftLabels(unittest.TestCase):
    def test_ptbxl_labels(self):
        metadata = {}  # No chagas info → strong negative
        label = get_chagas_label(metadata, 'ptbxl')
        self.assertEqual(label, 0.0, "PTB-XL should be strong negative (0.0)")

    def test_sami_trop_labels(self):
        metadata = {}
        label = get_chagas_label(metadata, 'sami_trop')
        self.assertEqual(label, 1.0, "SaMi-Trop should be strong positive (1.0)")

    def test_code15_labels(self):
        # Your code likely checks for 'chagas' key
        metadata_neg = {'chagas': 0}  # or missing → soft negative
        label_neg = get_chagas_label(metadata_neg, 'code15')
        self.assertAlmostEqual(label_neg, 0.2, delta=0.01, msg="CODE-15 negative should be 0.2")

        metadata_missing = {}  # Missing key → assume negative
        label_missing = get_chagas_label(metadata_missing, 'code15')
        self.assertAlmostEqual(label_missing, 0.2, delta=0.01, msg="Missing chagas key should default to soft negative")

        metadata_pos = {'chagas': 1}
        label_pos = get_chagas_label(metadata_pos, 'code15')
        self.assertAlmostEqual(label_pos, 0.8, delta=0.01, msg="CODE-15 positive should be 0.8")

    def test_confident_labels(self):
        self.assertTrue(is_confident_label(0.0, 'ptbxl'), "PTB-XL strong negative should be confident")
        self.assertTrue(is_confident_label(1.0, 'sami_trop'), "SaMi-Trop strong positive should be confident")
        self.assertTrue(is_confident_label(0.2, 'code15'), "CODE-15 soft negative should be confident")
        self.assertTrue(is_confident_label(0.8, 'code15'), "CODE-15 soft positive should be confident")
        self.assertFalse(is_confident_label(0.5, 'code15'), "CODE-15 uncertain (0.5) should not be confident")

    def test_batch_labels(self):
        labels = np.array([0.1, 0.25, 0.8, 0.9, 0.5])
        confident = is_confident_label(labels, 'code15')
        expected = np.array([True, True, True, True, False])  # <0.3 or >0.7
        np.testing.assert_array_equal(confident, expected)

if __name__ == "__main__":
    unittest.main(verbosity=2)