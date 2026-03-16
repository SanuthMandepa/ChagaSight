# tests/test_data_integrity.py
"""
Tests for processed data integrity.

FIXES vs original:
  - 'sami_trop' → 'samitrop'
  - File naming: {id}.npy not {id}_img.npy
  - Signal shape: (12, 1000) not (1000, 12)
  - Image shape:  (3, 24, 2048) — correct
  - Auto-detect code15 folder (code15 or code15s)
"""
import unittest
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR   = Path("tests/verification_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_WFDB    = Path("data/official_wfdb")
PROC_100     = Path("data/processed/1d_signals_100hz")
PROC_IMG     = Path("data/processed/2d_images")
META_DIR     = Path("data/processed/metadata")


def get_ds_name(raw_name: str) -> str:
    """Map folder aliases to canonical names used in processed/."""
    return {'sami_trop': 'samitrop', 'code15s': 'code15'}.get(raw_name, raw_name)


class TestProcessedFolders(unittest.TestCase):

    def test_1d_signal_folders(self):
        for ds in ['ptbxl', 'samitrop', 'code15']:
            p = PROC_100 / ds
            if p.exists():
                files = list(p.glob('*.npy'))
                self.assertGreater(len(files), 0, f"{ds} 1D folder empty")

    def test_2d_image_folders(self):
        for ds in ['ptbxl', 'samitrop', 'code15']:
            p = PROC_IMG / ds
            if p.exists():
                files = list(p.glob('*.npy'))
                self.assertGreater(len(files), 0, f"{ds} 2D folder empty")

    def test_metadata_csv_exists(self):
        candidates = [
            META_DIR / 'combined_5fold.csv',
            META_DIR / 'combined_all.csv',
        ]
        found = any(p.exists() for p in candidates)
        if not found:
            self.skipTest("No metadata CSV found — run data preparation first")


class TestSignalShapes(unittest.TestCase):
    """Verify saved 1D signals have shape (12, 1000)."""

    def _check_ds(self, ds: str):
        folder = PROC_100 / ds
        if not folder.exists():
            self.skipTest(f"{ds} 1D folder not found: {folder}")
        files = list(folder.glob('*.npy'))[:5]
        if not files:
            self.skipTest(f"{ds} has no .npy files")
        for f in files:
            arr = np.load(f)
            self.assertEqual(arr.ndim, 2,
                             f"{f.name}: expected 2D array, got shape {arr.shape}")
            self.assertEqual(arr.shape[0], 12,
                             f"{f.name}: expected 12 leads first, got shape {arr.shape}")
            self.assertEqual(arr.shape[1], 1000,
                             f"{f.name}: expected 1000 samples, got shape {arr.shape}")
            self.assertEqual(arr.dtype, np.float32,
                             f"{f.name}: expected float32, got {arr.dtype}")
            self.assertTrue(np.all(np.isfinite(arr)),
                            f"{f.name}: contains NaN or Inf")

    def test_ptbxl_signal_shapes(self):   self._check_ds('ptbxl')
    def test_samitrop_signal_shapes(self): self._check_ds('samitrop')
    def test_code15_signal_shapes(self):  self._check_ds('code15')


class TestImageShapes(unittest.TestCase):
    """Verify saved 2D images have shape (3, 24, 2048) uint8."""

    def _check_ds(self, ds: str):
        folder = PROC_IMG / ds
        if not folder.exists():
            self.skipTest(f"{ds} 2D folder not found: {folder}")
        files = list(folder.glob('*.npy'))[:5]
        if not files:
            self.skipTest(f"{ds} has no image .npy files")
        for f in files:
            arr = np.load(f)
            self.assertEqual(arr.shape, (3, 24, 2048),
                             f"{f.name}: expected (3,24,2048), got {arr.shape}")
            self.assertEqual(arr.dtype, np.uint8,
                             f"{f.name}: expected uint8, got {arr.dtype}")
            self.assertGreaterEqual(arr.min(), 0)
            self.assertLessEqual(   arr.max(), 255)

    def test_ptbxl_image_shapes(self):    self._check_ds('ptbxl')
    def test_samitrop_image_shapes(self):  self._check_ds('samitrop')
    def test_code15_image_shapes(self):   self._check_ds('code15')


class TestSampleVisualization(unittest.TestCase):
    """Load one sample per dataset and produce diagnostic plots."""

    def _load_first(self, ds, kind):
        folder = (PROC_100 if kind == '1d' else PROC_IMG) / ds
        if not folder.exists():
            return None, None
        files = sorted(folder.glob('*.npy'))
        if not files:
            return None, None
        return np.load(files[0]), files[0].stem

    def test_visualization_1d(self):
        fig, axes = plt.subplots(3, 12, figsize=(20, 9))
        fig.suptitle('1D Signals: One lead per column, one dataset per row  '
                     'shape=(12, 1000) @ 100 Hz', fontweight='bold')
        ds_list = ['ptbxl', 'samitrop', 'code15']
        colors  = ['#2E86AB', '#A23B72', '#F18F01']
        for row_idx, (ds, color) in enumerate(zip(ds_list, colors)):
            arr, fname = self._load_first(ds, '1d')
            if arr is None:
                continue
            t = np.arange(arr.shape[1]) / 100
            for lead_idx in range(12):
                ax = axes[row_idx, lead_idx]
                ax.plot(t, arr[lead_idx], lw=0.5, color=color)
                if row_idx == 0:
                    lead_names = ['I','II','III','aVR','aVL','aVF',
                                  'V1','V2','V3','V4','V5','V6']
                    ax.set_title(lead_names[lead_idx], fontsize=7)
                if lead_idx == 0:
                    ax.set_ylabel(ds.upper(), fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'integrity_1d_overview.png', dpi=120)
        plt.close()

    def test_visualization_2d(self):
        ds_list = ['ptbxl', 'samitrop', 'code15']
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('2D Images: 3 WCT channels per dataset  shape=(3, 24, 2048) uint8',
                     fontweight='bold')
        ch_labels = ['Ch0 RA-ref', 'Ch1 LA-ref', 'Ch2 LL-ref']
        for row_idx, ds in enumerate(ds_list):
            arr, fname = self._load_first(ds, '2d')
            if arr is None:
                continue
            for ch in range(3):
                ax = axes[row_idx, ch]
                ax.imshow(arr[ch].astype(float), aspect='auto', cmap='RdBu_r',
                          vmin=0, vmax=255, interpolation='nearest')
                if row_idx == 0:
                    ax.set_title(ch_labels[ch], fontsize=9)
                if ch == 0:
                    ax.set_ylabel(f'{ds.upper()}\n{fname}', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'integrity_2d_overview.png', dpi=120)
        plt.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
