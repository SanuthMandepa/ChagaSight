# tests/test_full_pipeline.py
"""
End-to-end pipeline tests using real WFDB records.

FIXES vs original:
  - resample_signal() not resample_ecg()
  - normalize_per_lead() not normalize_dataset()
  - build_2d_image() not ecg_to_contour_image()
  - 'samitrop' not 'sami_trop'
  - Signal convention throughout: (12, T) leads-first
  - Auto-detects code15 WFDB folder (code15 or code15s)
  - Correct expected shape: (12, 1000) for 1D, (3, 24, 2048) for 2D
"""
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wfdb
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample        import resample_signal, pad_or_trim
from src.preprocessing.normalization   import normalize_per_lead
from src.preprocessing.image_embedding import build_2d_image

OUTPUT_DIR = Path("tests/verification_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_WFDB = Path("data/official_wfdb")

SAMPLES = {
    'ptbxl':    {'id': '1',    'native_fs': 500},
    'samitrop': {'id': '3629', 'native_fs': 400},
    'code15':   {'id': '13',   'native_fs': 400},
}
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


def _wfdb_folder(ds: str) -> Path:
    for name in [ds, f'{ds}s']:
        p = BASE_WFDB / name
        if p.exists():
            return p
    return BASE_WFDB / ds


def _load_wfdb(ds: str, record_id: str):
    folder = _wfdb_folder(ds)
    if ds == 'ptbxl':
        num = int(record_id)
        sub = f"{num // 1000:05d}"
        for suffix in ['_lr', '_hr', '']:
            for subdir in ['records100', 'records500', '']:
                if subdir:
                    p = folder / subdir / sub / f"{num:05d}{suffix}"
                else:
                    p = folder / sub / f"{num:05d}{suffix}"
                if p.with_suffix('.hea').exists():
                    sig, fields = wfdb.rdsamp(str(p))
                    return sig.T.astype(np.float32), float(fields['fs'])
        raise FileNotFoundError(f"PTB-XL WFDB not found for ID {record_id}")
    else:
        p = folder / record_id
        if not p.with_suffix('.hea').exists():
            raise FileNotFoundError(f"WFDB not found: {p}.hea")
        sig, fields = wfdb.rdsamp(str(p))
        return sig.T.astype(np.float32), float(fields['fs'])


class TestFullPipeline(unittest.TestCase):

    def _get_signal(self, ds):
        if ds not in SAMPLES:
            self.skipTest(f"No sample ID defined for {ds}")
        info = SAMPLES[ds]
        folder = _wfdb_folder(ds)
        if not folder.exists():
            self.skipTest(f"WFDB folder not found: {folder}")
        try:
            sig, fs = _load_wfdb(ds, info['id'])
        except FileNotFoundError as e:
            self.skipTest(str(e))
        return sig, fs

    # ── 1D FM pipeline tests ──────────────────────────────────────────────

    def _test_1d_pipeline(self, ds):
        sig, fs = self._get_signal(ds)
        # Shape check: (12, T) throughout
        self.assertEqual(sig.shape[0], 12, f"{ds}: first dim must be 12 leads")

        filtered = remove_baseline(sig, fs=fs)
        self.assertEqual(filtered.shape, sig.shape,
                         f"{ds}: baseline removal changed shape")

        resampled = resample_signal(filtered, original_fs=fs, target_fs=100)
        self.assertEqual(resampled.shape[0], 12)

        fixed = pad_or_trim(resampled, 1000)
        self.assertEqual(fixed.shape, (12, 1000),
                         f"{ds}: 1D FM output must be (12, 1000)")

        normalised = normalize_per_lead(fixed, clip_std=3.0)
        self.assertEqual(normalised.shape, (12, 1000))
        self.assertTrue(np.all(normalised >= -3.0))
        self.assertTrue(np.all(normalised <=  3.0))
        self.assertTrue(np.all(np.isfinite(normalised)))

    def test_ptbxl_1d(self):    self._test_1d_pipeline('ptbxl')
    def test_samitrop_1d(self): self._test_1d_pipeline('samitrop')
    def test_code15_1d(self):   self._test_1d_pipeline('code15')

    # ── 2D image pipeline tests ───────────────────────────────────────────

    def _test_2d_pipeline(self, ds):
        sig, fs = self._get_signal(ds)

        filtered  = remove_baseline(sig, fs=fs)
        res_500   = resample_signal(filtered, original_fs=fs, target_fs=500)
        fixed_500 = pad_or_trim(res_500, 5000)
        norm_500  = normalize_per_lead(fixed_500, clip_std=3.0)

        self.assertEqual(norm_500.shape, (12, 5000))
        self.assertTrue(np.all(norm_500 >= -3.0) and np.all(norm_500 <= 3.0))

        img = build_2d_image(norm_500, target_width=2048, random_crop=False)
        self.assertEqual(img.shape, (3, 24, 2048),
                         f"{ds}: 2D image must be (3, 24, 2048)")
        self.assertEqual(img.dtype, np.uint8)
        self.assertGreaterEqual(img.min(), 0)
        self.assertLessEqual(   img.max(), 255)

    def test_ptbxl_2d(self):    self._test_2d_pipeline('ptbxl')
    def test_samitrop_2d(self): self._test_2d_pipeline('samitrop')
    def test_code15_2d(self):   self._test_2d_pipeline('code15')

    # ── Combined visualization ────────────────────────────────────────────

    def test_visualization_all_datasets(self):
        """
        One comprehensive plot per dataset showing every pipeline stage.
        Compares across PTB-XL, SaMi-Trop, CODE-15%.
        """
        fig, big_axes = plt.subplots(3, 5, figsize=(22, 12))
        fig.suptitle('Full Pipeline: Raw → Baseline → 100Hz → 500Hz → 2D Image  (Lead I)',
                     fontweight='bold', fontsize=12)

        ds_colors = {'ptbxl': '#2E86AB', 'samitrop': '#A23B72', 'code15': '#F18F01'}
        col_titles = ['Raw WFDB', 'Baseline removed', '1D FM\n(100Hz, 1000samp)',
                      '500Hz z-scored', '2D Image Ch0\n(24×2048 uint8)']

        for col, title in enumerate(col_titles):
            big_axes[0, col].set_title(title, fontsize=9, fontweight='bold')

        for row, ds in enumerate(['ptbxl', 'samitrop', 'code15']):
            color = ds_colors[ds]
            try:
                sig, fs = _load_wfdb(ds, SAMPLES[ds]['id'])
            except FileNotFoundError:
                for ax in big_axes[row]:
                    ax.text(0.5, 0.5, f'{ds}\nnot found', ha='center', va='center',
                            transform=ax.transAxes, fontsize=8)
                continue

            filtered = remove_baseline(sig, fs=fs)
            fm_100   = normalize_per_lead(pad_or_trim(
                           resample_signal(filtered, fs, 100), 1000), clip_std=3.0)
            s500_z   = normalize_per_lead(pad_or_trim(
                           resample_signal(filtered, fs, 500), 5000), clip_std=3.0)
            img      = build_2d_image(s500_z, target_width=2048, random_crop=False)

            stages = [
                (sig[0],     np.arange(sig.shape[1]) / fs,          'mV'),
                (filtered[0],np.arange(filtered.shape[1]) / fs,     'mV'),
                (fm_100[0],  np.arange(1000) / 100,                  'z'),
                (s500_z[0],  np.arange(5000) / 500,                  'z'),
            ]
            for col, (y, t, ylabel) in enumerate(stages):
                ax = big_axes[row, col]
                ax.plot(t, y, lw=0.6, color=color)
                if col == 0:
                    ax.set_ylabel(f'{ds.upper()}\n{ylabel}', fontsize=8)
                ax.grid(True, alpha=0.25); ax.set_xticks([])

            # 2D image channel 0
            ax_img = big_axes[row, 4]
            ax_img.imshow(img[0].astype(float), aspect='auto', cmap='RdBu_r',
                          vmin=0, vmax=255, interpolation='nearest')
            ax_img.set_xticks([]); ax_img.set_yticks([])

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'pipeline_full_all_datasets.png', dpi=150)
        plt.close()
        print("  Saved: pipeline_full_all_datasets.png")


if __name__ == '__main__':
    unittest.main(verbosity=2)
