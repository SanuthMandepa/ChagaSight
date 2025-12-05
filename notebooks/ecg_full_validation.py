"""
Full visual validation for PTB-XL ECG (DETAILED EXPLANATION VERSION)

This script confirms whether the ECG → 1D → 2D image conversion pipeline is correct.

It produces **SEPARATE PLOTS** for:

1) Raw PTB-XL waveform (lead I)
2) Processed 1D ECG (your normalized 400 Hz signal)
3) 3-channel structured ECG image (24×2048 per channel)
4) Raw vs Processed 1D (alignment check)
5) Processed 1D vs Image Row (does image encode the ECG?)
6) Raw vs Image Row (cross-verification)

All images are saved under:
    notebooks/verification_outputs/full_validation/

RUN (from project root):
    python -m notebooks.ecg_full_validation --ecg-id 1 \
        --raw-rel records100/00000/00001_lr

IMPORTANT:
- ecg_id refers to your processed 1D and 2D files.
- raw-rel points to the correct WFDB file folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import wfdb

# SciPy resample helps compare different sample lengths
try:
    from scipy.signal import resample
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------

PTBXL_ROOT = Path("data/raw/ptbxl")
SIG_DIR    = Path("data/processed/1d_signals/ptbxl")
IMG_DIR    = Path("data/processed/2d_images/ptbxl")

OUT_DIR    = Path("notebooks/verification_outputs/full_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Utility to save images cleanly
def _save_fig(name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ Saved {path}")

# ---------------------------------------------------------------------
# 1) RAW PTB-XL PLOT
# ---------------------------------------------------------------------
"""
Raw PTB-XL signals:
- Sampling rate = 100 Hz
- Length ≈ 10 seconds → 1000 samples
- Values are already scaled to mV
"""
def plot_raw_only(raw_sig: np.ndarray, fs_raw: float, ecg_id: int, lead: int = 0):
    # Time axis (0–10 seconds)
    t_raw = np.arange(raw_sig.shape[0]) / fs_raw

    # Plot only first 10s (PTB-XL provides ~10s for 100Hz records)
    mask = t_raw <= 10.0
    t_raw_10 = t_raw[mask]
    raw_10 = raw_sig[mask, lead]

    plt.figure(figsize=(12, 3))
    plt.plot(t_raw_10, raw_10, label="Raw (100 Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"ECG {ecg_id} – Raw WFDB (lead {lead+1})")
    plt.legend()
    _save_fig(f"ecg{ecg_id}_raw_only_lead{lead+1}.png")


# ---------------------------------------------------------------------
# 2) PROCESSED 1D (400 Hz SIGNAL)
# ---------------------------------------------------------------------
"""
Your preprocessing pipeline performs:
- Baseline drift removal
- Resample from 100 Hz → 400 Hz
- Z-score normalization
→ Produces 4000 samples (10 seconds × 400 Hz)
"""
def plot_proc_only(proc_sig: np.ndarray, fs_proc: float, ecg_id: int, lead: int = 0):
    t_proc = np.arange(proc_sig.shape[0]) / fs_proc

    plt.figure(figsize=(12, 3))
    plt.plot(t_proc, proc_sig[:, lead], label="Processed (400 Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (z-score)")
    plt.title(f"ECG {ecg_id} – Processed 1D (lead {lead+1})")
    plt.legend()
    _save_fig(f"ecg{ecg_id}_proc_only_lead{lead+1}.png")


# ---------------------------------------------------------------------
# 3) 2D IMAGE CHANNELS
# ---------------------------------------------------------------------
"""
Each ECG is converted into:
    shape = (3, 24, 2048)

Why?
- 24 height: leads are arranged into spatial rows (+ padding)
- 2048 width: fixed-time projection after resampling
- Values 0–255: normalized grayscale intensities

The middle 8 rows contain the REAL ECG encoding.
The top & bottom are zero padding (expected).
"""
def plot_image_only(img: np.ndarray, ecg_id: int):
    assert img.ndim == 3 and img.shape[0] == 3, "Image must be (3, H, W)"

    for ch in range(3):
        plt.figure(figsize=(12, 3))
        plt.imshow(img[ch], aspect="auto", cmap="gray")
        plt.colorbar()
        plt.title(f"ECG {ecg_id} – 2D image channel {ch+1}")
        _save_fig(f"ecg{ecg_id}_img_only_ch{ch+1}.png")


# ---------------------------------------------------------------------
# 4) RAW vs PROCESSED 1D COMPARISON
# ---------------------------------------------------------------------
"""
Purpose:
- Check that preprocessing has NOT distorted morphology
- Peaks & troughs should align perfectly
- Differences due to normalization & baseline removal
"""
def plot_raw_vs_proc(raw_sig, proc_sig, fs_raw, fs_proc, ecg_id, lead=0):
    t_raw = np.arange(raw_sig.shape[0]) / fs_raw
    t_proc = np.arange(proc_sig.shape[0]) / fs_proc

    mask = t_raw <= 10.0
    t_raw_10 = t_raw[mask]
    raw_10 = raw_sig[mask, lead]

    plt.figure(figsize=(12, 3))
    plt.plot(t_raw_10, raw_10, label="Raw (100 Hz)", alpha=0.7)
    plt.plot(t_proc, proc_sig[:, lead], label="Processed (400 Hz)", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"ECG {ecg_id} – Raw vs Processed (lead {lead+1})")
    plt.legend()
    _save_fig(f"ecg{ecg_id}_raw_vs_proc_lead{lead+1}.png")


# ---------------------------------------------------------------------
# 5) PROCESSED 1D vs IMAGE ROW
# ---------------------------------------------------------------------
"""
Purpose:
- Validate that the 2D image actually contains the ECG signal.
- Row_pixels = grayscale intensity of the selected row.
- lead_resampled = processed signal rescaled to image width (2048).

This directly shows whether the image is an accurate embedding.
"""
def plot_proc_vs_imgrow(proc_sig, img, fs_proc, ecg_id,
                        lead=0, channel=0, row=10):
    if not HAVE_SCIPY:
        print("  ⚠ scipy not available – skipping proc vs imgrow")
        return

    H, W = img.shape[1], img.shape[2]
    if row >= H:
        print(f"  ⚠ row {row} out of range (H={H}); skipping")
        return

    T = proc_sig.shape[0]
    t_proc = np.arange(T) / fs_proc

    lead_sig = proc_sig[:, lead]
    lead_resampled = resample(lead_sig, W)

    row_pixels = img[channel, row]
    t_img = np.linspace(0, t_proc[-1], W)

    fig, ax1 = plt.subplots(figsize=(12, 3))

    ax1.plot(t_proc, lead_sig, label=f"Processed lead {lead+1} (4000)", alpha=0.3)
    ax1.plot(t_img, lead_resampled, label="Resampled to 2048", alpha=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("1D amplitude (processed)")

    ax2 = ax1.twinx()
    ax2.plot(t_img, row_pixels, color="black", alpha=0.5, label="Image row pixels")
    ax2.set_ylabel("Image intensity (0–255)")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"ECG {ecg_id} – Processed 1D vs 2D image row {row}")
    _save_fig(f"ecg{ecg_id}_proc_vs_img_ch{channel+1}_row{row}.png")


# ---------------------------------------------------------------------
# 6) RAW vs IMAGE ROW (Cross check)
# ---------------------------------------------------------------------
"""
Same as (5) but compares RAW WFDB → IMAGE.

Purpose:
- Detect if any preprocessing has broken morphology
- Confirm the 2D image still corresponds to the *true ECG*
"""
def plot_raw_vs_imgrow(raw_sig, img, fs_raw, ecg_id,
                       channel=0, row=10):
    if not HAVE_SCIPY:
        print("  ⚠ scipy not available – skipping raw vs imgrow")
        return

    H, W = img.shape[1], img.shape[2]
    if row >= H:
        print(f"  ⚠ row {row} out of range (H={H}); skipping")
        return

    t_raw = np.arange(raw_sig.shape[0]) / fs_raw
    mask = t_raw <= 10.0
    t_raw_10 = t_raw[mask]
    raw_10 = raw_sig[mask, 0]

    raw_resampled = resample(raw_10, W)
    t_img = np.linspace(0, t_raw_10[-1], W)

    row_pixels = img[channel, row]

    fig, ax1 = plt.subplots(figsize=(12, 3))

    ax1.plot(t_raw_10, raw_10, label="Raw (100 Hz)", alpha=0.3)
    ax1.plot(t_img, raw_resampled, label="Raw resampled (2048)", alpha=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (raw)")

    ax2 = ax1.twinx()
    ax2.plot(t_img, row_pixels, color="black", alpha=0.5, label="Image row")
    ax2.set_ylabel("Image intensity")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"ECG {ecg_id} – Raw vs 2D image row {row}")
    _save_fig(f"ecg{ecg_id}_raw_vs_img_ch{channel+1}_row{row}.png")


# ---------------------------------------------------------------------
# MAIN LOGIC FOR ONE ECG
# ---------------------------------------------------------------------
def check_ecg(ecg_id: int, raw_rel: str, row: int = 10):
    print(f"\n=== ECG_ID {ecg_id} ===")

    # -------------------------
    # Load processed 1D signal
    # -------------------------
    sig_path = SIG_DIR / f"{ecg_id}.npy"
    if not sig_path.exists():
        print(f"❌ Processed 1D not found: {sig_path}")
        return

    proc_sig = np.load(sig_path)
    fs_proc = 400.0
    print(f"Processed shape: {proc_sig.shape}, range {proc_sig.min():.2f} → {proc_sig.max():.2f}")

    # -------------------------
    # Load RAW WFDB signal
    # -------------------------
    raw_base = PTBXL_ROOT / raw_rel
    try:
        raw_record = wfdb.rdrecord(str(raw_base))
        raw_sig = raw_record.p_signal.astype("float32")
        fs_raw = float(raw_record.fs)
        print(f"Raw shape: {raw_sig.shape}, fs={fs_raw}")
    except Exception as e:
        print(f"❌ Could not load raw WFDB: {e}")
        return

    # -------------------------
    # Load structured image
    # -------------------------
    img_path = IMG_DIR / f"{ecg_id}_img.npy"
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return

    img = np.load(img_path)
    if img.ndim == 4:
        img = img[0]  # If batch dimension exists

    print(f"Image shape: {img.shape}, range {img.min():.1f} → {img.max():.1f}")

    # ---- Single-view plots ----
    plot_raw_only(raw_sig, fs_raw, ecg_id)
    plot_proc_only(proc_sig, fs_proc, ecg_id)
    plot_image_only(img, ecg_id)

    # ---- Pairwise plots ----
    plot_raw_vs_proc(raw_sig, proc_sig, fs_raw, fs_proc, ecg_id)
    plot_proc_vs_imgrow(proc_sig, img, fs_proc, ecg_id, row=row)
    plot_raw_vs_imgrow(raw_sig, img, fs_raw, ecg_id, row=row)


# ---------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecg-id", type=int, required=True,
                        help="Which ECG ID to validate (matches processed files)")
    parser.add_argument("--raw-rel", required=True,
                        help="Relative WFDB path under data/raw/ptbxl/")
    parser.add_argument("--row", type=int, default=10,
                        help="Which row of the image to compare (default 10)")
    args = parser.parse_args()

    print(f"Output directory: {OUT_DIR.resolve()}")
    check_ecg(args.ecg_id, args.raw_rel, row=args.row)


if __name__ == "__main__":
    main()

