# tests/test_image_embedding.py
"""
Tests for ECG 2D Image Construction.

FIXES vs original:
  - build_2d_image() not ecg_to_contour_image()
  - Signal shape (12, T) not (T, 12)
  - Added detailed 8×T → 24×T visualization showing the construction step-by-step
  - Added WCT channel explanation plots
"""
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.image_embedding import build_2d_image
from src.preprocessing.normalization   import normalize_per_lead

OUTPUT_DIR = Path("tests/verification_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
FS_500     = 500
DURATION   = 10
TARGET_W   = 2048


class TestImageEmbeddingShape(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Realistic synthetic ECG: (12, 5000) at 500 Hz, z-scored, clipped to [-3,3]
        self.signal = np.zeros((12, FS_500 * DURATION), dtype=np.float32)
        t = np.linspace(0, DURATION, FS_500 * DURATION)
        for i in range(12):
            self.signal[i] = (np.sin(2 * np.pi * 1.1 * t + i * 0.5) +
                              0.15 * np.random.randn(FS_500 * DURATION))
        self.signal = normalize_per_lead(self.signal, clip_std=3.0)

    def test_output_shape(self):
        img = build_2d_image(self.signal, target_width=TARGET_W)
        self.assertEqual(img.shape, (3, 24, TARGET_W),
                         f"Expected (3, 24, 2048), got {img.shape}")

    def test_output_dtype_uint8(self):
        img = build_2d_image(self.signal, target_width=TARGET_W)
        self.assertEqual(img.dtype, np.uint8, "Output must be uint8")

    def test_pixel_range(self):
        img = build_2d_image(self.signal, target_width=TARGET_W)
        self.assertGreaterEqual(img.min(), 0)
        self.assertLessEqual(   img.max(), 255)

    def test_three_channels_differ(self):
        """WCT re-referencing should produce 3 distinct channel views."""
        img = build_2d_image(self.signal, target_width=TARGET_W)
        # Channels should not all be identical
        self.assertFalse(np.allclose(img[0].astype(float), img[1].astype(float)),
                         "Channels 0 and 1 are identical — WCT re-referencing failed")
        self.assertFalse(np.allclose(img[1].astype(float), img[2].astype(float)),
                         "Channels 1 and 2 are identical — WCT re-referencing failed")

    def test_deterministic(self):
        """Same input must produce same output (no random)."""
        img1 = build_2d_image(self.signal, target_width=TARGET_W, random_crop=False)
        img2 = build_2d_image(self.signal, target_width=TARGET_W, random_crop=False)
        np.testing.assert_array_equal(img1, img2, "Same input must produce same image")


class TestImageConstruction8To24(unittest.TestCase):
    """
    Detailed visualization of HOW 12 leads become 24 rows (the '8→24' question).

    Kim et al. describe their original resolution as '8×timestamps'  (8 clinical rows).
    Our implementation uses 24 rows: each of the 12 leads is duplicated into 2 rows.

    Stacking rule (image_embedding._stack_leads_to_height24):
      Lead 0  (I)   → rows  0,  1
      Lead 1  (II)  → rows  2,  3
      Lead 2  (III) → rows  4,  5
      Lead 3  (aVR) → rows  6,  7
      Lead 4  (aVL) → rows  8,  9
      Lead 5  (aVF) → rows 10, 11
      Lead 6  (V1)  → rows 12, 13
      Lead 7  (V2)  → rows 14, 15
      Lead 8  (V3)  → rows 16, 17
      Lead 9  (V4)  → rows 18, 19
      Lead 10 (V5)  → rows 20, 21
      Lead 11 (V6)  → rows 22, 23
    """

    def setUp(self):
        np.random.seed(10)
        self.signal = np.zeros((12, FS_500 * DURATION), dtype=np.float32)
        t = np.linspace(0, DURATION, FS_500 * DURATION)
        for i in range(12):
            self.signal[i] = (np.sin(2 * np.pi * 1.2 * t + i * 0.4) +
                              0.1 * np.random.randn(FS_500 * DURATION))
        self.signal = normalize_per_lead(self.signal, clip_std=3.0)

    def test_row_duplication(self):
        """Rows 2k and 2k+1 must contain identical values for lead k."""
        img = build_2d_image(self.signal, target_width=TARGET_W, random_crop=False)
        for lead_idx in range(12):
            row_a = img[0, 2 * lead_idx,     :].astype(float)
            row_b = img[0, 2 * lead_idx + 1, :].astype(float)
            np.testing.assert_array_equal(
                row_a, row_b,
                f"Rows {2*lead_idx} and {2*lead_idx+1} (lead {LEAD_NAMES[lead_idx]}) must be identical")

    def test_visualization_8_to_24(self):
        """
        Generate the definitive '8×W → 24×W' explanation plot.

        Panel 1: 12 leads as (12, W) — the raw stacked form
        Panel 2: (24, W) after lead duplication — channel 0 (RA-referenced)
        Panel 3: All 3 channels overlaid (3, 24, W)
        Panel 4: Final uint8 image channel 0 as imshow
        """
        img = build_2d_image(self.signal, target_width=TARGET_W, random_crop=False)

        # Reconstruct the (12, W) view of channel 0 for illustration
        # by subsampling every other row (undo duplication)
        ch0_12row = img[0, ::2, :]   # shape (12, 2048) — one row per lead

        fig = plt.figure(figsize=(20, 18))
        gs  = gridspec.GridSpec(4, 1, hspace=0.45)

        # ── Panel 1: 12 leads (12, W) ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        im1 = ax1.imshow(ch0_12row.astype(float), aspect='auto',
                         cmap='RdBu_r', interpolation='nearest',
                         vmin=0, vmax=255)
        ax1.set_yticks(range(12))
        ax1.set_yticklabels(LEAD_NAMES, fontsize=8)
        ax1.set_title('Step 1: 12 leads × 1 row each  →  (12, 2048)  '
                      '[Channel 0: RA-referenced WCT]', fontsize=10)
        ax1.set_xlabel('Pixel column (time)')
        plt.colorbar(im1, ax=ax1, fraction=0.015, label='Pixel value [0-255]')

        # ── Panel 2: After duplication → (24, W) ───────────────────────────
        ax2 = fig.add_subplot(gs[1])
        im2 = ax2.imshow(img[0].astype(float), aspect='auto',
                         cmap='RdBu_r', interpolation='nearest',
                         vmin=0, vmax=255)
        # Mark the lead boundaries
        for row in range(0, 24, 2):
            ax2.axhline(row - 0.5, color='lime', lw=0.4, alpha=0.5)
        ax2.set_yticks([2*i + 0.5 for i in range(12)])
        ax2.set_yticklabels(LEAD_NAMES, fontsize=8)
        ax2.set_title('Step 2: Each lead duplicated → 2 rows  →  (24, 2048)  '
                      '— this is what the ViT2D encoder sees', fontsize=10)
        ax2.set_xlabel('Pixel column (time)')
        plt.colorbar(im2, ax=ax2, fraction=0.015, label='Pixel value [0-255]')

        # ── Panel 3: 3 channels stacked for comparison ─────────────────────
        ax3 = fig.add_subplot(gs[2])
        # Create RGB composite for visual inspection
        rgb = np.stack([
            img[0].astype(float) / 255,
            img[1].astype(float) / 255,
            img[2].astype(float) / 255,
        ], axis=-1)
        ax3.imshow(rgb, aspect='auto', interpolation='nearest')
        ax3.set_title('Step 3: All 3 channels (RGB composite)  →  (3, 24, 2048)\n'
                      'R=RA-ref  G=LA-ref  B=LL-ref  (WCT channels)', fontsize=10)
        ax3.set_yticks([2*i + 0.5 for i in range(12)])
        ax3.set_yticklabels(LEAD_NAMES, fontsize=8)
        ax3.set_xlabel('Pixel column (time)')

        # ── Panel 4: Each channel as separate uint8 image ──────────────────
        ax4 = fig.add_subplot(gs[3])
        # Stack all 3 channels vertically with separators
        combined = np.vstack([img[0], np.full((2, TARGET_W), 128, dtype=np.uint8),
                               img[1], np.full((2, TARGET_W), 128, dtype=np.uint8),
                               img[2]])
        ax4.imshow(combined.astype(float), aspect='auto', cmap='gray',
                   interpolation='nearest', vmin=0, vmax=255)
        ax4.axhline(24.5, color='red', lw=1.5, label='Channel boundary')
        ax4.axhline(50.5, color='red', lw=1.5)
        ax4.set_yticks([12, 37, 62])
        ax4.set_yticklabels(['Ch0\n(RA-ref)', 'Ch1\n(LA-ref)', 'Ch2\n(LL-ref)'],
                            fontsize=9)
        ax4.set_title('Step 4: Final uint8 image — 3 × (24, 2048) grayscale channels', fontsize=10)
        ax4.set_xlabel('Pixel column (time)')

        plt.suptitle(
            'ECG Image Construction: 12 leads → (12, W) → (24, W) → (3, 24, 2048) uint8\n'
            'Kim et al. (2025): "resized from 8×timestamps to 24×2048"\n'
            '(Their 8 rows = clinical printout layout; our 24 rows = 12 leads × 2 rows each)',
            fontsize=11, fontweight='bold', y=1.01
        )
        plt.savefig(OUTPUT_DIR / 'embed_01_8to24_construction.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: embed_01_8to24_construction.png")

    def test_visualization_wct_channels(self):
        """Show what each WCT channel represents and how they differ."""
        img = build_2d_image(self.signal, target_width=TARGET_W, random_crop=False)

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        channel_info = [
            ('Channel 0 — RA-referenced  (ref = Right Arm)',
             'LL, V1-V6, LA in RA reference frame', '#2E86AB'),
            ('Channel 1 — LA-referenced  (ref = Left Arm)',
             'RA, V1-V6, LL in LA reference frame', '#A23B72'),
            ('Channel 2 — LL-referenced  (ref = Left Leg)',
             'RA, V1-V6, LA in LL reference frame', '#F18F01'),
        ]
        for ax, (title, subtitle, color), ch in zip(axes, channel_info, range(3)):
            im = ax.imshow(img[ch].astype(float), aspect='auto',
                           cmap='RdBu_r', interpolation='nearest',
                           vmin=0, vmax=255)
            ax.set_title(f'{title}\n{subtitle}', fontsize=9)
            ax.set_yticks([2*i + 0.5 for i in range(12)])
            ax.set_yticklabels(LEAD_NAMES, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.015, label='[0-255]')
        axes[-1].set_xlabel('Pixel column (width=2048)')
        plt.suptitle('WCT Re-Referencing: 3 Body-Surface Contour Views\n'
                     'Kim et al. (2025): "Each channel corresponds to a distinct contour '
                     'on the body surface"',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'embed_02_wct_channels.png', dpi=150)
        plt.close()

    def test_pixel_encoding_explanation(self):
        """Show how signal amplitude [-3, 3] maps to pixel values [0, 255]."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Signal vs pixel value mapping
        x = np.linspace(-4, 4, 200)
        y_clipped = np.clip(x, -3, 3)
        y_pixel   = ((y_clipped + 3) / 6 * 255).astype(np.uint8).astype(float)

        axes[0].plot(x, y_clipped, color='steelblue', lw=2, label='Clipped signal [-3, 3]')
        axes[0].axvline(-3, color='r', ls='--', alpha=0.5, label='Clip boundary')
        axes[0].axvline( 3, color='r', ls='--', alpha=0.5)
        axes[0].set_xlabel('Normalized amplitude (z-score)'); axes[0].set_ylabel('After clip')
        axes[0].set_title('Step 1: Clip z-scored signal to [-3, 3]')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, y_pixel, color='darkorange', lw=2)
        axes[1].axhline(  0, color='r', ls='--', alpha=0.5, label='0 = min')
        axes[1].axhline(255, color='g', ls='--', alpha=0.5, label='255 = max')
        axes[1].set_xlabel('Normalized amplitude (z-score)')
        axes[1].set_ylabel('Pixel value [0-255]')
        axes[1].set_title('Step 2: Map [-3, 3] → [0, 255] uint8\n'
                          'pixel = (clip(x, -3, 3) + 3) / 6 × 255')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.suptitle('Pixel Encoding: Signal Amplitude → uint8', fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'embed_03_pixel_encoding.png', dpi=150)
        plt.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
