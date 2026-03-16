# tests/test_wfdb_data.py
"""
Tests for WFDB data loading.

FIXES:
  - 'sami_trop' → 'samitrop'
  - 'code15' WFDB folder may be 'code15s' (per uploaded files)
  - Sample IDs aligned with actual files (13, 14 for code15s)
"""
import unittest
import os
import wfdb
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = Path("tests/verification_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_PATH = Path("data/official_wfdb")


def find_code15_folder() -> Path:
    """Auto-detect code15 folder name (may be 'code15' or 'code15s')."""
    for name in ['code15', 'code15s']:
        p = BASE_PATH / name
        if p.exists():
            return p
    return BASE_PATH / 'code15'  # default


def find_samitrop_folder() -> Path:
    for name in ['samitrop', 'sami_trop']:
        p = BASE_PATH / name
        if p.exists():
            return p
    return BASE_PATH / 'samitrop'


class TestWFDBData(unittest.TestCase):

    def setUp(self):
        self.code15_dir  = find_code15_folder()
        self.samitrop_dir = find_samitrop_folder()
        self.ptbxl_dir   = BASE_PATH / 'ptbxl'

        # Real sample IDs from uploaded files
        self.samples = {
            'code15s':  ('13',   self.code15_dir),
            'samitrop': ('3629', self.samitrop_dir),
            'ptbxl':    ('1',    self.ptbxl_dir),
        }

    def test_folder_existence(self):
        for ds, (_, folder) in self.samples.items():
            if folder.exists():
                self.assertTrue(folder.is_dir(), f"{ds} path is not a directory")

    def test_sample_load_and_shape(self):
        for ds, (sample_id, folder) in self.samples.items():
            if not folder.exists():
                self.skipTest(f"Folder not found: {folder}")
            path = self._get_sample_path(ds, sample_id, folder)
            if path is None:
                continue
            signal, fields = wfdb.rdsamp(str(path))
            # WFDB returns (T, leads) — transpose to our (leads, T) convention
            signal_leads_first = signal.T.astype(np.float32)
            self.assertEqual(signal.shape[1], 12, f"{ds}: expected 12 leads")
            self.assertEqual(signal_leads_first.shape[0], 12)
            self.assertEqual(signal_leads_first.shape[1], signal.shape[0])

    def test_sampling_frequencies(self):
        """Verify expected sampling rates per dataset."""
        expected_fs = {
            'code15s':  400,
            'samitrop': 400,
            'ptbxl':    None,  # 100 or 500
        }
        for ds, (sample_id, folder) in self.samples.items():
            if not folder.exists():
                continue
            path = self._get_sample_path(ds, sample_id, folder)
            if path is None:
                continue
            _, fields = wfdb.rdsamp(str(path))
            if expected_fs[ds] is not None:
                self.assertEqual(fields['fs'], expected_fs[ds],
                                 f"{ds}: unexpected fs={fields['fs']}")

    def test_lead_names(self):
        """Verify standard 12-lead names."""
        expected = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        for ds, (sample_id, folder) in self.samples.items():
            if not folder.exists():
                continue
            path = self._get_sample_path(ds, sample_id, folder)
            if path is None:
                continue
            _, fields = wfdb.rdsamp(str(path))
            names = [n.upper() for n in fields['sig_name']]
            for exp in expected:
                self.assertIn(exp, names,
                              f"{ds}: missing lead {exp} in {names}")

    def test_visualization(self):
        for ds, (sample_id, folder) in self.samples.items():
            if not folder.exists():
                continue
            path = self._get_sample_path(ds, sample_id, folder)
            if path is None:
                continue
            signal, fields = wfdb.rdsamp(str(path))
            sig = signal.T.astype(np.float32)   # (12, T)
            fs  = fields['fs']
            t   = np.arange(sig.shape[1]) / fs

            fig, axes = plt.subplots(12, 1, figsize=(14, 18), sharex=True)
            lead_names = fields['sig_name']
            for i, ax in enumerate(axes):
                ax.plot(t, sig[i], lw=0.6, color='#2E86AB')
                ax.set_ylabel(lead_names[i], fontsize=7, rotation=0, labelpad=28)
                ax.grid(True, alpha=0.2)
                ax.set_yticks([])
            axes[-1].set_xlabel('Time (s)')
            plt.suptitle(f'{ds.upper()} — WFDB Sample ID={sample_id}  '
                         f'fs={fs} Hz  shape=(12, {sig.shape[1]})',
                         fontweight='bold')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'wfdb_{ds}_sample_12leads.png', dpi=120)
            plt.close()

    def _get_sample_path(self, ds, sample_id, folder):
        if 'ptbxl' in ds:
            numeric_id = int(sample_id)
            subfolder  = f"{numeric_id // 1000:05d}"
            for suffix in ['_lr', '_hr']:
                p = folder / f'records100/{subfolder}/{numeric_id:05d}{suffix}'
                if p.with_suffix('.hea').exists():
                    return p
                p = folder / f'records500/{subfolder}/{numeric_id:05d}{suffix}'
                if p.with_suffix('.hea').exists():
                    return p
            return None
        else:
            p = folder / sample_id
            if p.with_suffix('.hea').exists():
                return p
            return None


if __name__ == '__main__':
    unittest.main(verbosity=2)
