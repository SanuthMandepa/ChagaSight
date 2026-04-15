# tests/test_preprocessing.py
"""
Tests for ECG preprocessing pipeline.

FIXES vs original:
  - Signal shape: (12, T) not (T, 12)  — our convention is leads-first
  - resample_signal()  not resample_ecg()
  - normalize_per_lead()  not normalize_dataset()
  - remove_baseline() signature: (signal, fs=fs)  not positional method string
  - All assertions now use (12, T) shapes
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample        import resample_signal, pad_or_trim
from src.preprocessing.normalization   import normalize_per_lead

OUTPUT_DIR = Path("tests/verification_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
FS         = 400   # CODE-15% / SaMi-Trop native rate


class TestBaselineRemoval(unittest.TestCase):
    """Verify baseline removal keeps shape and reduces low-freq drift."""

    def setUp(self):
        np.random.seed(0)
        # (12, T) — our convention, leads first
        self.signal = np.random.normal(0, 0.5, (12, FS * 10)).astype(np.float32)
        # Inject synthetic baseline drift on each lead
        t = np.linspace(0, 10, FS * 10)
        for i in range(12):
            self.signal[i] += 0.5 * np.sin(2 * np.pi * 0.2 * t + i)

    def test_shape_preserved(self):
        out = remove_baseline(self.signal, fs=FS)
        self.assertEqual(out.shape, self.signal.shape,
                         "Baseline removal must not change signal shape (12, T)")

    def test_low_freq_reduced(self):
        out = remove_baseline(self.signal, fs=FS)
        # Low-freq energy should decrease after baseline removal
        from scipy.signal import welch
        f_orig, p_orig = welch(self.signal[0], fs=FS, nperseg=min(4000, FS * 10))
        f_filt, p_filt = welch(out[0],         fs=FS, nperseg=min(4000, FS * 10))
        lf_mask = f_orig < 0.5
        self.assertLess(p_filt[lf_mask].mean(), p_orig[lf_mask].mean(),
                        "Bandpass should remove <0.5 Hz energy")

    def test_visualization(self):
        out = remove_baseline(self.signal, fs=FS)
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        t = np.arange(self.signal.shape[1]) / FS
        axes[0].plot(t, self.signal[0], lw=0.8, color='steelblue')
        axes[0].set_title('Original Lead I (with synthetic baseline drift)')
        axes[0].set_ylabel('mV')
        axes[1].plot(t, out[0], lw=0.8, color='darkorange')
        axes[1].set_title('After baseline removal')
        axes[1].set_ylabel('mV')
        axes[2].plot(t, self.signal[0] - out[0], lw=0.8, color='firebrick')
        axes[2].set_title('Removed baseline component')
        axes[2].set_ylabel('mV')
        axes[2].set_xlabel('Time (s)')
        for ax in axes:
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'preproc_01_baseline_removal.png', dpi=150)
        plt.close()


class TestResampling(unittest.TestCase):
    """Verify resampling to 100 Hz and 500 Hz with correct shapes."""

    def setUp(self):
        np.random.seed(1)
        self.signal_400 = np.random.normal(0, 1, (12, FS * 10)).astype(np.float32)
        self.fs = FS

    def test_resample_to_100hz(self):
        out = resample_signal(self.signal_400, original_fs=self.fs, target_fs=100)
        self.assertEqual(out.shape[0], 12, "Leads dimension must stay 12")
        expected_T = round(self.signal_400.shape[1] * 100 / self.fs)
        self.assertAlmostEqual(out.shape[1], expected_T, delta=5,
                               msg="100 Hz output length incorrect")

    def test_resample_to_500hz(self):
        out = resample_signal(self.signal_400, original_fs=self.fs, target_fs=500)
        self.assertEqual(out.shape[0], 12)
        expected_T = round(self.signal_400.shape[1] * 500 / self.fs)
        self.assertAlmostEqual(out.shape[1], expected_T, delta=5)

    def test_pad_or_trim_100hz(self):
        """After resample → 100 Hz, pad/trim to exactly 1000 samples."""
        raw = resample_signal(self.signal_400, original_fs=self.fs, target_fs=100)
        fixed = pad_or_trim(raw, 1000)
        self.assertEqual(fixed.shape, (12, 1000), "Must produce exactly (12, 1000)")

    def test_pad_or_trim_500hz(self):
        raw = resample_signal(self.signal_400, original_fs=self.fs, target_fs=500)
        fixed = pad_or_trim(raw, 5000)
        self.assertEqual(fixed.shape, (12, 5000))

    def test_visualization(self):
        out_100 = pad_or_trim(resample_signal(self.signal_400, self.fs, 100),  1000)
        out_500 = pad_or_trim(resample_signal(self.signal_400, self.fs, 500),  5000)

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
        # Original 400 Hz
        t0 = np.arange(self.signal_400.shape[1]) / self.fs
        axes[0].plot(t0, self.signal_400[0], lw=0.6, color='steelblue',
                     label=f'Original 400 Hz ({self.signal_400.shape[1]} samples)')
        axes[0].set_title('Original (400 Hz)')
        # Resampled 100 Hz
        t1 = np.arange(out_100.shape[1]) / 100
        axes[1].plot(t1, out_100[0], lw=0.8, color='darkorange',
                     label=f'100 Hz ({out_100.shape[1]} samples)')
        axes[1].set_title('Resampled → 100 Hz, trimmed to 1000 samples (10 s)')
        # Resampled 500 Hz
        t2 = np.arange(out_500.shape[1]) / 500
        axes[2].plot(t2, out_500[0], lw=0.5, color='seagreen',
                     label=f'500 Hz ({out_500.shape[1]} samples)')
        axes[2].set_title('Resampled → 500 Hz, trimmed to 5000 samples (10 s)')
        for ax in axes:
            ax.set_ylabel('mV'); ax.grid(True, alpha=0.3)
        axes[2].set_xlabel('Time (s)')
        plt.suptitle('Lead I — Resampling Comparison', fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'preproc_02_resampling.png', dpi=150)
        plt.close()


class TestNormalization(unittest.TestCase):
    """Verify z-score per-lead normalization."""

    def setUp(self):
        np.random.seed(2)
        # Signals with very different amplitudes per lead (realistic ECG variation)
        self.signal = np.zeros((12, 1000), dtype=np.float32)
        for i in range(12):
            self.signal[i] = np.random.normal(loc=i * 0.3 - 1.5, scale=0.2 + i * 0.1, size=1000)

    def test_mean_near_zero(self):
        out = normalize_per_lead(self.signal)
        means = np.mean(out, axis=1)  # per lead
        np.testing.assert_allclose(means, 0, atol=1e-2,
                                   err_msg="Per-lead mean must be ~0 after z-score")

    def test_std_near_one(self):
        out = normalize_per_lead(self.signal)
        stds = np.std(out, axis=1)
        np.testing.assert_allclose(stds, 1, atol=1e-2,
                                   err_msg="Per-lead std must be ~1 after z-score")

    def test_shape_preserved(self):
        out = normalize_per_lead(self.signal)
        self.assertEqual(out.shape, self.signal.shape)

    def test_clip_std3(self):
        out = normalize_per_lead(self.signal, clip_std=3.0)
        self.assertTrue(np.all(out >= -3.0) and np.all(out <= 3.0),
                        "clip_std=3 must bound all values to [-3, 3]")

    def test_visualization(self):
        out = normalize_per_lead(self.signal, clip_std=3.0)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        for i in range(12):
            axes[0].plot(self.signal[i], lw=0.5, alpha=0.7, label=LEAD_NAMES[i])
            axes[1].plot(out[i],         lw=0.5, alpha=0.7, label=LEAD_NAMES[i])
        axes[0].set_title('Before normalization (different amplitude per lead)')
        axes[1].set_title('After z-score per-lead + clip to [-3, 3]')
        for ax in axes:
            ax.set_xlabel('Sample'); ax.set_ylabel('Amplitude'); ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=6, ncol=6, loc='upper right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'preproc_03_normalization.png', dpi=150)
        plt.close()


class TestFullPreprocessingChain(unittest.TestCase):
    """End-to-end chain: raw 400Hz → baseline → resample → normalize → shapes OK."""

    def setUp(self):
        np.random.seed(3)
        t = np.linspace(0, 10.24, 4096)  # 400 Hz, 10.24 s (CODE-15% style)
        self.raw_signal = np.zeros((12, 4096), dtype=np.float32)
        freqs = [1.0, 1.2, 0.9, 0.8, 1.1, 1.3, 0.7, 1.5, 0.6, 1.4, 0.8, 1.2]
        for i, f in enumerate(freqs):
            self.raw_signal[i] = (np.sin(2 * np.pi * f * t) +
                                  0.2 * np.sin(2 * np.pi * 50 * t) +
                                  0.3 * np.sin(2 * np.pi * 0.15 * t) +
                                  0.1 * np.random.randn(4096))
        self.fs_orig = 400

    def test_1d_fm_chain(self):
        """Code path for 1D FM input: baseline → 100 Hz → 1000 samples → z-score."""
        filtered   = remove_baseline(self.raw_signal, fs=self.fs_orig)
        resampled  = resample_signal(filtered, self.fs_orig, 100)
        fixed      = pad_or_trim(resampled, 1000)
        normalised = normalize_per_lead(fixed, clip_std=3.0)

        self.assertEqual(filtered.shape,   (12, 4096))
        self.assertEqual(fixed.shape,      (12, 1000))
        self.assertEqual(normalised.shape, (12, 1000))
        self.assertTrue(np.all(np.isfinite(normalised)))

    def test_2d_image_chain(self):
        """Code path for 2D image: baseline → 500 Hz → 5000 samples → z-score → clip."""
        filtered   = remove_baseline(self.raw_signal, fs=self.fs_orig)
        resampled  = resample_signal(filtered, self.fs_orig, 500)
        fixed      = pad_or_trim(resampled, 5000)
        normalised = normalize_per_lead(fixed, clip_std=3.0)

        self.assertEqual(fixed.shape,      (12, 5000))
        self.assertEqual(normalised.shape, (12, 5000))
        self.assertTrue(np.all(normalised >= -3.0) and np.all(normalised <= 3.0))

    def test_visualization(self):
        """Plot every intermediate step side-by-side for Lead I."""
        filtered  = remove_baseline(self.raw_signal, fs=self.fs_orig)
        res_100   = pad_or_trim(resample_signal(filtered, self.fs_orig, 100),   1000)
        norm_100  = normalize_per_lead(res_100, clip_std=3.0)
        res_500   = pad_or_trim(resample_signal(filtered, self.fs_orig, 500),   5000)
        norm_500  = normalize_per_lead(res_500, clip_std=3.0)

        steps = [
            ('Raw 400 Hz',                self.raw_signal[0], self.fs_orig),
            ('Baseline removed',          filtered[0],        self.fs_orig),
            ('100 Hz (FM input)',          res_100[0],         100),
            ('100 Hz z-scored+clip',      norm_100[0],        100),
            ('500 Hz (image input)',       res_500[0],         500),
            ('500 Hz z-scored+clip',      norm_500[0],        500),
        ]

        fig, axes = plt.subplots(len(steps), 1, figsize=(16, 14), sharex=False)
        colors = ['#2E86AB','#A23B72','#F18F01','#C73E1D','#3B1F2B','#44BBA4']
        for ax, (label, sig, fs), color in zip(axes, steps, colors):
            t = np.arange(len(sig)) / fs
            ax.plot(t, sig, lw=0.7, color=color)
            ax.set_title(f'{label}  shape=({len(sig)},)  fs={fs} Hz', fontsize=9)
            ax.set_ylabel('mV'); ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle('Full Preprocessing Pipeline — Lead I', fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'preproc_04_full_chain.png', dpi=150)
        plt.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
