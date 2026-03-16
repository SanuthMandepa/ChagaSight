# tests/test_soft_labels.py
"""
Tests for soft label utilities.

FIXES vs original:
  - Our soft_labels.py has hard_to_soft_label() and vector_hard_to_soft()
    NOT get_chagas_label() / is_confident_label() (those don't exist)
  - Tests rewritten to match actual API
"""
import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.soft_labels import hard_to_soft_label, vector_hard_to_soft


class TestSoftLabels(unittest.TestCase):

    # ── hard_to_soft_label (scalar) ────────────────────────────────────────

    def test_code15_positive(self):
        """CODE-15% positive → 0.8 (Van Santvliet: uncertain labels)."""
        label = hard_to_soft_label(1, pos_soft=0.8, neg_soft=0.2)
        self.assertAlmostEqual(label, 0.8, delta=0.001)

    def test_code15_negative(self):
        """CODE-15% negative → 0.2."""
        label = hard_to_soft_label(0, pos_soft=0.8, neg_soft=0.2)
        self.assertAlmostEqual(label, 0.2, delta=0.001)

    def test_ptbxl_strong_negative(self):
        """PTB-XL / SaMi-Trop use hard labels (0.0 / 1.0)."""
        label_neg = hard_to_soft_label(0, pos_soft=1.0, neg_soft=0.0)
        label_pos = hard_to_soft_label(1, pos_soft=1.0, neg_soft=0.0)
        self.assertEqual(label_neg, 0.0)
        self.assertEqual(label_pos, 1.0)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            hard_to_soft_label(2)  # not 0 or 1

    # ── vector_hard_to_soft (batch) ────────────────────────────────────────

    def test_vector_shape(self):
        labels = np.array([0, 1, 0, 1, 0])
        result = vector_hard_to_soft(labels)
        self.assertEqual(result.shape, labels.shape)

    def test_vector_values(self):
        labels = np.array([0, 1, 0, 1])
        result = vector_hard_to_soft(labels, pos_soft=0.8, neg_soft=0.2)
        expected = np.array([0.2, 0.8, 0.2, 0.8], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=0.001)

    def test_vector_dtype_float32(self):
        labels = np.array([0, 1, 1, 0])
        result = vector_hard_to_soft(labels)
        self.assertEqual(result.dtype, np.float32)

    def test_vector_invalid_labels(self):
        with self.assertRaises(ValueError):
            vector_hard_to_soft(np.array([0, 1, 2]))  # 2 is invalid

    def test_all_datasets_label_mapping(self):
        """
        Verify the three label regimes in ChagaSight:
          PTB-XL   : hard negative  (0 → 0.0)
          SaMi-Trop: hard positive  (1 → 1.0)
          CODE-15% : soft           (0 → 0.2, 1 → 0.8)
        """
        ptbxl_neg   = hard_to_soft_label(0, pos_soft=1.0, neg_soft=0.0)
        samitrop_pos = hard_to_soft_label(1, pos_soft=1.0, neg_soft=0.0)
        code15_neg  = hard_to_soft_label(0, pos_soft=0.8, neg_soft=0.2)
        code15_pos  = hard_to_soft_label(1, pos_soft=0.8, neg_soft=0.2)

        self.assertEqual(ptbxl_neg,    0.0)
        self.assertEqual(samitrop_pos, 1.0)
        self.assertAlmostEqual(code15_neg, 0.2, delta=0.001)
        self.assertAlmostEqual(code15_pos, 0.8, delta=0.001)


if __name__ == '__main__':
    unittest.main(verbosity=2)
