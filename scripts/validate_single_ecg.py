# scripts/validate_single_ecg.py

"""
Visual sanity-check for ONE ECG across the full pipeline:
raw → baseline-removed → resampled → fixed-length → z-scored → 2D image.

Supported datasets (matching your README & folder structure):
- PTB-XL     : WFDB records under data/raw/ptbxl/ + 1D/2D under data/processed
- SaMi-Trop  : exams.hdf5 under data/raw/sami_trop/ + 1D/2D under data/processed
- CODE-15%   : sharded HDF5 under data/raw/code15/ + 1D/2D under data/processed

For each run we create a CLEAR, SORTED set of figures:

    notebooks/verification_outputs/pipeline/<dataset>/ecg_<ID>/
        01_raw_lead1.png
        02_baseline_removed_lead1.png
        03_resampled_400hz_lead1.png
        04_fixed_10s_lead1.png
        05_zscore_final_lead1.png

        06_raw_vs_baseline_lead1.png
        07_baseline_vs_resampled_lead1.png

        08_raw_12leads_grid.png
        09_zscore_12leads_grid.png

        10_raw_spectrogram_lead1.png
        11_zscore_spectrogram_lead1.png

        12_image_ch1_gray.png
        13_image_ch2_gray.png
        14_image_ch3_gray.png
        15_image_rgb_composite.png
        16_image_row10_ch1_strip.png

        17_zscore_hist_per_lead.png
        18_image_pixel_histogram.png
        19_lead_corr_raw.png
        20_lead_corr_zscore.png

        21_raw_vs_zscore_lead1.png
        22_zscore_vs_image_row10.png      (requires SciPy)
        23_raw_vs_image_row10.png         (requires SciPy)

USAGE (from project root):

    # PTB-XL (you ALREADY tested this pattern)
    python -m scripts.validate_single_ecg ^
        --dataset ptbxl ^
        --id 1 ^
        --ptbxl-raw-rel records100/00000/00001_lr

    # SaMi-Trop (pick a valid exam_id from data/raw/sami_trop/exams.csv)
    python -m scripts.validate_single_ecg ^
        --dataset sami_trop ^
        --id 294669

    # CODE-15% (exam_id from data/raw/code15/exams.csv)
    python -m scripts.validate_single_ecg ^
        --dataset code15 ^
        --id 1169160
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt
import wfdb
import h5py
import pandas as pd

# Optional – only needed for row-wise comparisons 1D ↔ 2D
try:
    from scipy.signal import resample as scipy_resample
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---- use your REAL preprocessing modules ----
from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead

# ============================================================
# CONSTANT PATHS  (exactly aligned with README + your folders)
# ============================================================

PTB_RAW = Path("data/raw/ptbxl")
PTB_SIG = Path("data/processed/1d_signals/ptbxl")
PTB_IMG = Path("data/processed/2d_images/ptbxl")
PTB_META = PTB_RAW / "ptbxl_database.csv"

SAMI_RAW = Path("data/raw/sami_trop")
SAMI_SIG = Path("data/processed/1d_signals/sami_trop")
SAMI_IMG = Path("data/processed/2d_images/sami_trop")

CODE_RAW = Path("data/raw/code15")
CODE_SIG = Path("data/processed/1d_signals/code15")
CODE_IMG = Path("data/processed/2d_images/code15")
CODE_META = CODE_RAW / "exams.csv"

TARGET_FS: float = 400.0
TARGET_SEC: float = 10.0
TARGET_LEN: int = int(TARGET_FS * TARGET_SEC)

OUT_ROOT = Path("notebooks/verification_outputs/pipeline")


# ============================================================
# SMALL HELPERS
# ============================================================

def ensure_outdir(dataset: str, ecg_id: int) -> Path:
    out = OUT_ROOT / dataset / f"ecg_{ecg_id}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_fig(out_dir: Path, name: str):
    out = out_dir / name
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✔ Saved {out}")


def time_axis(n: int, fs: float) -> np.ndarray:
    return np.arange(n, dtype=float) / float(fs)


# ============================================================
# RAW LOADERS
# ============================================================

def load_raw_ptbxl(ecg_id: int, rel: str | None) -> Tuple[np.ndarray, float]:
    """Load raw PTB-XL WFDB (shape [T, 12], fs≈100)."""
    if rel is None:
        if not PTB_META.exists():
            raise FileNotFoundError(
                f"ptbxl_database.csv missing at {PTB_META}. "
                "Either provide --ptbxl-raw-rel or download metadata."
            )
        df = pd.read_csv(PTB_META)
        row = df.loc[df.ecg_id == ecg_id]
        if row.empty:
            raise ValueError(f"ecg_id {ecg_id} not found in {PTB_META}")
        rel = row.iloc[0]["filename_lr"]

    rec = wfdb.rdrecord(str(PTB_RAW / rel))
    sig = rec.p_signal.astype("float32")
    fs = float(rec.fs)
    print(f"  [PTB-XL] raw signal shape={sig.shape}, fs={fs}")
    return sig, fs


def load_raw_samitrop(exam_id: int) -> Tuple[np.ndarray, float]:
    """Load raw SaMi-Trop tracing from exams.hdf5 via exams.csv index."""
    exams_csv = SAMI_RAW / "exams.csv"
    exams_h5 = SAMI_RAW / "exams.hdf5"

    df = pd.read_csv(exams_csv)
    idx_arr = df.index[df.exam_id == exam_id]
    if len(idx_arr) == 0:
        raise ValueError(f"exam_id {exam_id} not found in {exams_csv}")
    idx = int(idx_arr[0])

    with h5py.File(exams_h5, "r") as f:
        tracings = f["tracings"][idx]  # (4096, 12) at 400 Hz

    sig = np.asarray(tracings, dtype="float32")
    fs = 400.0
    print(f"  [SaMi-Trop] raw signal shape={sig.shape}, fs={fs}")
    return sig, fs


def load_raw_code15(exam_id: int) -> Tuple[np.ndarray, float]:
    """
    Load raw CODE-15% tracing from sharded HDF5.

    exams.csv must have at least:
        - exam_id
        - trace_file (HDF5 filename containing that exam)
    """
    if not CODE_META.exists():
        raise FileNotFoundError(f"{CODE_META} not found")

    df = pd.read_csv(CODE_META)
    row = df.loc[df.exam_id == exam_id]
    if row.empty:
        raise ValueError(f"exam_id {exam_id} not found in CODE exams.csv")

    trace_file = CODE_RAW / row.iloc[0]["trace_file"]

    with h5py.File(trace_file, "r") as f:
        ids = np.asarray(f["exam_id"])
        idx_arr = np.where(ids == exam_id)[0]
        if len(idx_arr) == 0:
            raise ValueError(f"exam_id {exam_id} not present inside {trace_file}")
        idx = int(idx_arr[0])
        tracings = f["tracings"][idx]  # (T, 12) at 400 Hz

    sig = np.asarray(tracings, dtype="float32")
    fs = 400.0
    print(f"  [CODE-15%] raw signal shape={sig.shape}, fs={fs}")
    return sig, fs


# ============================================================
# PREPROCESSING PIPELINE (UPDATED for new function signatures)
# ============================================================

def run_pipeline(raw: np.ndarray, fs_in: float) -> Dict[str, np.ndarray]:
    """
    Apply EXACT pipeline used for your processed 1D signals:

        1) remove_baseline(raw, fs=fs_in)
        2) resample_ecg(..., fs_in, TARGET_FS)
           (now returns (signal, fs) tuple)
        3) pad_or_trim(..., TARGET_LEN)
        4) zscore_per_lead(...)

    Returns a dict of all stages so we can plot each one.
    """
    print("  → Baseline removal")
    # Use default baseline removal (highpass for PTB-XL, moving_average for others)
    baseline_method = "highpass" if fs_in == 100.0 else "moving_average"
    baseline = remove_baseline(raw, fs=fs_in, method=baseline_method)
    
    print("  → Resample to 400 Hz")
    # UPDATED: resample_ecg now returns (signal, fs)
    resampled, fs_out = resample_ecg(baseline, fs_in, TARGET_FS)

    print("  → Pad/Trim to 10s (4000 samples)")
    fixed = pad_or_trim(resampled, TARGET_LEN)

    print("  → Z-score per lead")
    z = zscore_per_lead(fixed)

    return {
        "raw": raw,
        "baseline": baseline,
        "resampled": resampled,
        "fs_resampled": fs_out,  # Keep track of resampled fs
        "fixed": fixed,
        "zscore": z,
    }


# ============================================================
# PLOTTING – 1D STAGES
# ============================================================

def plot_stages(stages: Dict[str, np.ndarray],
                fs_in: float,
                out_dir: Path,
                dataset: str,
                ecg_id: int) -> None:
    """Save 1D lead-1 plots for all key stages (01-05)."""

    # 01: raw
    t_raw = time_axis(stages["raw"].shape[0], fs_in)
    plt.figure(figsize=(12, 3))
    plt.plot(t_raw, stages["raw"][:, 0])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (raw units)")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 01 Raw lead I")
    save_fig(out_dir, "01_raw_lead1.png")

    # 02: baseline-removed
    t_b = time_axis(stages["baseline"].shape[0], fs_in)
    plt.figure(figsize=(12, 3))
    plt.plot(t_b, stages["baseline"][:, 0])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (baseline-removed)")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 02 Baseline-removed lead I")
    save_fig(out_dir, "02_baseline_removed_lead1.png")

    # 03: resampled
    t_r = time_axis(stages["resampled"].shape[0], TARGET_FS)
    plt.figure(figsize=(12, 3))
    plt.plot(t_r, stages["resampled"][:, 0])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (resampled)")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 03 Resampled to 400 Hz (lead I)")
    save_fig(out_dir, "03_resampled_400hz_lead1.png")

    # 04: fixed-length 10 s
    t_f = time_axis(stages["fixed"].shape[0], TARGET_FS)
    plt.figure(figsize=(12, 3))
    plt.plot(t_f, stages["fixed"][:, 0])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 04 Fixed 10 s window (lead I)")
    save_fig(out_dir, "04_fixed_10s_lead1.png")

    # 05: final z-scored
    plt.figure(figsize=(12, 3))
    plt.plot(t_f, stages["zscore"][:, 0])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (z-score)")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 05 Final z-scored lead I")
    save_fig(out_dir, "05_zscore_final_lead1.png")


def plot_overlay_stage_pairs(stages: Dict[str, np.ndarray],
                             fs_in: float,
                             out_dir: Path,
                             dataset: str,
                             ecg_id: int) -> None:
    """
    Extra overlays so you can visually see how each stage changes
    the waveform on top of the previous one.
    """

    # 06: raw vs baseline-removed
    t_raw = time_axis(stages["raw"].shape[0], fs_in)
    t_b = time_axis(stages["baseline"].shape[0], fs_in)
    plt.figure(figsize=(12, 3))
    plt.plot(t_raw, stages["raw"][:, 0], label="Raw lead I", alpha=0.5)
    plt.plot(t_b, stages["baseline"][:, 0],
             label="Baseline-removed lead I", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 06 Raw vs Baseline-removed (lead I)")
    plt.legend()
    save_fig(out_dir, "06_raw_vs_baseline_lead1.png")

    # 07: baseline-removed vs resampled
    t_r = time_axis(stages["resampled"].shape[0], TARGET_FS)
    plt.figure(figsize=(12, 3))
    plt.plot(t_b, stages["baseline"][:, 0],
             label="Baseline-removed (original fs)", alpha=0.5)
    plt.plot(t_r, stages["resampled"][:, 0],
             label="Resampled to 400 Hz", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 07 Baseline-removed vs Resampled (lead I)")
    plt.legend()
    save_fig(out_dir, "07_baseline_vs_resampled_lead1.png")


def plot_lead_grids(stages: Dict[str, np.ndarray],
                    fs_in: float,
                    out_dir: Path,
                    dataset: str,
                    ecg_id: int) -> None:
    """
    12-lead overview for raw and final z-scored signals.
    """

    # 08: raw – 12 lead grid
    sig = stages["raw"]
    t_raw = time_axis(sig.shape[0], fs_in)
    fig, axes = plt.subplots(3, 4, figsize=(14, 6), sharex=True)
    axes = axes.flatten()
    for i in range(12):
        axes[i].plot(t_raw, sig[:, i], linewidth=0.6)
        axes[i].set_title(f"Lead {i+1}", fontsize=8)
    fig.suptitle(f"{dataset.upper()} ECG {ecg_id} – 08 Raw 12-lead overview")
    for ax in axes[-4:]:
        ax.set_xlabel("Time (s)")
    fig.text(0.04, 0.5, "Amplitude", va="center", rotation="vertical")
    save_fig(out_dir, "08_raw_12leads_grid.png")

    # 09: z-scored – 12 lead grid
    sig_z = stages["zscore"]
    t_z = time_axis(sig_z.shape[0], TARGET_FS)
    fig, axes = plt.subplots(3, 4, figsize=(14, 6), sharex=True)
    axes = axes.flatten()
    for i in range(12):
        axes[i].plot(t_z, sig_z[:, i], linewidth=0.6)
        axes[i].set_title(f"Lead {i+1}", fontsize=8)
    fig.suptitle(f"{dataset.upper()} ECG {ecg_id} – 09 Z-scored 12-lead overview")
    for ax in axes[-4:]:
        ax.set_xlabel("Time (s)")
    fig.text(0.04, 0.5, "Amplitude (z-score)", va="center", rotation="vertical")
    save_fig(out_dir, "09_zscore_12leads_grid.png")


def plot_spectrograms(stages: Dict[str, np.ndarray],
                      fs_in: float,
                      out_dir: Path,
                      dataset: str,
                      ecg_id: int) -> None:
    """Simple spectrograms for raw and z-scored lead I."""

    # 10: raw spectrogram
    plt.figure(figsize=(10, 4))
    plt.specgram(stages["raw"][:, 0],
                 Fs=fs_in,
                 NFFT=256,
                 noverlap=128,
                 cmap="viridis")
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 10 Raw spectrogram (lead I)")
    save_fig(out_dir, "10_raw_spectrogram_lead1.png")

    # 11: z-scored spectrogram
    plt.figure(figsize=(10, 4))
    plt.specgram(stages["zscore"][:, 0],
                 Fs=TARGET_FS,
                 NFFT=256,
                 noverlap=128,
                 cmap="viridis")
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 11 Z-scored spectrogram (lead I)")
    save_fig(out_dir, "11_zscore_spectrogram_lead1.png")


# ============================================================
# PLOTTING – 2D IMAGE VIEWS
# ============================================================

def plot_image_views(img: np.ndarray,
                     out_dir: Path,
                     dataset: str,
                     ecg_id: int,
                     row: int = 10) -> None:
    """
    2D image diagnostics:
        12-14: each channel as gray image
        15   : RGB composite of 3 channels
        16   : horizontal strip for one row in channel 1
    """
    assert img.ndim == 3 and img.shape[0] == 3, "Expected image (3, H, W)"

    H, W = img.shape[1:]

    # 12-14: per-channel gray images
    for ch in range(3):
        plt.figure(figsize=(12, 3))
        plt.imshow(img[ch], aspect="auto", cmap="gray")
        plt.colorbar(label="Intensity (0–255)")
        plt.xlabel("Horizontal pixel index")
        plt.ylabel("Vertical pixel index (contour row)")
        plt.title(
            f"{dataset.upper()} ECG {ecg_id} – "
            f"{12+ch:02d} Structured image channel {ch+1} (contour map)"
        )
        save_fig(out_dir, f"{12+ch:02d}_image_ch{ch+1}_gray.png")

    # 15: RGB composite (just stacking 3 channels)
    rgb = np.stack(img, axis=-1)  # (H, W, 3)
    # normalize for display
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    plt.figure(figsize=(10, 4))
    plt.imshow(rgb_norm)
    plt.xlabel("Horizontal pixel index")
    plt.ylabel("Vertical pixel index (row)")
    plt.title(
        f"{dataset.upper()} ECG {ecg_id} – 15 RGB composite of 3 contour channels"
    )
    save_fig(out_dir, "15_image_rgb_composite.png")

    # 16: horizontal strip – channel 1, chosen row
    if row < 0 or row >= H:
        row = H // 2
    strip = img[0, row]

    plt.figure(figsize=(12, 3))
    plt.plot(np.arange(W), strip)
    plt.xlabel("Horizontal pixel index")
    plt.ylabel("Intensity (channel 1)")
    plt.title(
        f"{dataset.upper()} ECG {ecg_id} – 16 Channel 1 row {row} "
        "(1D slice through contour image)"
    )
    save_fig(out_dir, "16_image_row10_ch1_strip.png")


# ============================================================
# PLOTTING – STATISTICS & CORRELATIONS
# ============================================================

def plot_statistics_and_correlations(stages: Dict[str, np.ndarray],
                                     img: np.ndarray,
                                     out_dir: Path,
                                     dataset: str,
                                     ecg_id: int) -> None:
    """
    Extra diagnostics:
        17 – per-lead histogram of z-scored values
        18 – histogram of image pixel intensities (all channels)
        19 – raw lead-to-lead correlation matrix
        20 – z-scored lead-to-lead correlation matrix
    """

    # 17: histogram of z-scores
    z = stages["zscore"]
    plt.figure(figsize=(8, 5))
    for lead in range(min(12, z.shape[1])):
        plt.hist(
            z[:, lead],
            bins=40,
            alpha=0.3,
            density=True,
            label=f"Lead {lead+1}",
        )
    plt.xlabel("Z-score value")
    plt.ylabel("Density")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 17 Distribution of z-scores (all leads)")
    plt.legend(fontsize=6, ncol=3)
    save_fig(out_dir, "17_zscore_hist_per_lead.png")

    # 18: image pixel histogram (all channels)
    pixels = img.flatten().astype(float)
    plt.figure(figsize=(8, 5))
    plt.hist(pixels, bins=64, color="gray", alpha=0.8)
    plt.xlabel("Pixel intensity (0–255)")
    plt.ylabel("Count")
    plt.title(
        f"{dataset.upper()} ECG {ecg_id} – 18 Histogram of contour image pixel intensities"
    )
    save_fig(out_dir, "18_image_pixel_histogram.png")

    # 19: raw lead correlation
    raw = stages["raw"]
    corr_raw = np.corrcoef(raw, rowvar=False)  # 12×12
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr_raw, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, label="Pearson r")
    plt.xticks(range(12), [f"L{i+1}" for i in range(12)], rotation=45)
    plt.yticks(range(12), [f"L{i+1}" for i in range(12)])
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 19 Raw lead-to-lead correlation")
    save_fig(out_dir, "19_lead_corr_raw.png")

    # 20: z-scored lead correlation
    corr_z = np.corrcoef(z, rowvar=False)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr_z, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, label="Pearson r")
    plt.xticks(range(12), [f"L{i+1}" for i in range(12)], rotation=45)
    plt.yticks(range(12), [f"L{i+1}" for i in range(12)])
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 20 Z-scored lead-to-lead correlation")
    save_fig(out_dir, "20_lead_corr_zscore.png")


# ============================================================
# PLOTTING – 1D ↔ 2D COMPARISONS (REQUIRES SCIPY)
# ============================================================

def plot_pairwise_1d_2d(stages: Dict[str, np.ndarray],
                        img: np.ndarray,
                        fs_in: float,
                        out_dir: Path,
                        dataset: str,
                        ecg_id: int,
                        row: int = 10) -> None:
    """
    21 – raw vs z-scored 1D (overlay), lead I
    22 – z-scored lead I vs image row (channel 1, given row)
    23 – raw lead I vs image row (channel 1, given row)
    """
    # 21: raw vs final z-scored (lead 1)
    t_raw = time_axis(stages["raw"].shape[0], fs_in)
    t_z = time_axis(stages["zscore"].shape[0], TARGET_FS)

    plt.figure(figsize=(12, 3))
    plt.plot(t_raw, stages["raw"][:, 0], label="Raw lead I", alpha=0.4)
    plt.plot(t_z, stages["zscore"][:, 0], label="Final z-scored lead I", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"{dataset.upper()} ECG {ecg_id} – 21 Raw vs z-scored (lead I)")
    plt.legend()
    save_fig(out_dir, "21_raw_vs_zscore_lead1.png")

    if not HAVE_SCIPY:
        print("  ⚠ SciPy not installed – skipping 1D↔2D row comparisons (22, 23)")
        return

    H, W = img.shape[1:]
    if row < 0 or row >= H:
        row = H // 2

    # Shared row from image channel 1
    row_pixels = img[0, row].astype(float)
    t_img = np.linspace(0.0, TARGET_SEC, W)

    # 22: z-scored vs image row
    z = stages["zscore"][:, 0]
    z_resampled = scipy_resample(z, W)

    fig, ax1 = plt.subplots(figsize=(12, 3))
    ax1.plot(t_img, z_resampled, label="Z-scored (resampled to image width)", alpha=0.9)
    ax1.set_xlabel("Time mapped onto image row (s)")
    ax1.set_ylabel("1D ECG amplitude (z-score)")

    ax2 = ax1.twinx()
    ax2.plot(t_img, row_pixels, color="black", alpha=0.5,
             label=f"Image channel 1 – row {row}")
    ax2.set_ylabel("Image intensity (0–255)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(
        f"{dataset.upper()} ECG {ecg_id} – 22 Z-scored lead I vs contour image row {row}"
    )
    save_fig(out_dir, "22_zscore_vs_image_row10.png")

    # 23: raw vs image row (first TARGET_SEC seconds)
    t_raw_full = time_axis(stages["raw"].shape[0], fs_in)
    mask = t_raw_full <= TARGET_SEC
    raw_10 = stages["raw"][mask, 0]
    t_raw_10 = t_raw_full[mask]
    raw_resampled = scipy_resample(raw_10, W)

    fig, ax1 = plt.subplots(figsize=(12, 3))
    ax1.plot(t_raw_10, raw_10, label="Raw lead I (first 10 s)", alpha=0.4)
    ax1.plot(t_img, raw_resampled,
             label="Raw resampled to image width", alpha=0.9)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Raw amplitude")

    ax2 = ax1.twinx()
    ax2.plot(t_img, row_pixels, color="black", alpha=0.5,
             label=f"Image channel 1 – row {row}")
    ax2.set_ylabel("Image intensity (0–255)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(
        f"{dataset.upper()} ECG {ecg_id} – 23 Raw lead I vs contour image row {row}"
    )
    save_fig(out_dir, "23_raw_vs_image_row10.png")


# ============================================================
# MAIN VALIDATION WRAPPER
# ============================================================

def validate_single_ecg(dataset: str,
                        ecg_id: int,
                        ptbxl_raw_rel: str | None = None,
                        row: int = 10) -> None:
    dataset = dataset.lower()

    # 1) Load RAW + find processed file paths
    if dataset == "ptbxl":
        raw, fs = load_raw_ptbxl(ecg_id, ptbxl_raw_rel)
        sig_path = PTB_SIG / f"{ecg_id}.npy"
        img_path = PTB_IMG / f"{ecg_id}_img.npy"
    elif dataset in ("sami_trop", "sami", "samitrop"):
        raw, fs = load_raw_samitrop(ecg_id)
        sig_path = SAMI_SIG / f"{ecg_id}.npy"
        img_path = SAMI_IMG / f"{ecg_id}_img.npy"
        dataset = "sami_trop"
    elif dataset in ("code15", "code_15"):
        raw, fs = load_raw_code15(ecg_id)
        sig_path = CODE_SIG / f"{ecg_id}.npy"
        img_path = CODE_IMG / f"{ecg_id}_img.npy"
        dataset = "code15"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"  Raw signal: {raw.shape}, fs={fs}")
    print(f"  Processed 1D path: {sig_path}")
    print(f"  Processed 2D path: {img_path}")

    # 2) Re-run pipeline step-by-step
    print("\n  Re-running preprocessing pipeline:")
    stages = run_pipeline(raw, fs)
    print(f"  [CHECK] final z-scored shape: {stages['zscore'].shape}")

    # 3) Optionally compare with saved processed 1D (sanity check)
    if sig_path.exists():
        try:
            proc_disk = np.load(sig_path)
            if proc_disk.shape == stages["zscore"].shape:
                diff = np.abs(proc_disk - stages["zscore"]).mean()
                print(f"  [CHECK] mean |disk − recomputed| = {diff:.6e}")
                if diff > 1e-5:
                    warnings.warn(f"Significant difference between disk and recomputed: {diff:.2e}")
            else:
                print(
                    f"  ⚠ Processed 1D on disk has shape {proc_disk.shape}, "
                    f"but recomputed pipeline has shape {stages['zscore'].shape}"
                )
        except Exception as e:
            print(f"  ⚠ Could not compare with disk file: {e}")
    else:
        print(f"  ⚠ Processed 1D .npy not found at {sig_path}")

    # 4) Load 2D contour image
    if not img_path.exists():
        raise FileNotFoundError(f"2D image file not found: {img_path}")
    
    img = np.load(img_path)
    if img.ndim == 4:
        img = img[0]
    print(f"  [IMAGE] shape={img.shape}, range={img.min():.1f} → {img.max():.1f}")

    # 5) Produce all plots in sorted order
    out_dir = ensure_outdir(dataset, ecg_id)
    print(f"\n  Saving plots to: {out_dir}")

    # 1D time-domain stages + overlays
    print("  Creating 1D stage plots (01-11)...")
    plot_stages(stages, fs, out_dir, dataset, ecg_id)
    plot_overlay_stage_pairs(stages, fs, out_dir, dataset, ecg_id)
    plot_lead_grids(stages, fs, out_dir, dataset, ecg_id)
    plot_spectrograms(stages, fs, out_dir, dataset, ecg_id)

    # 2D image-space diagnostics
    print("  Creating 2D image plots (12-16)...")
    plot_image_views(img, out_dir, dataset, ecg_id, row=row)

    # statistics & correlations
    print("  Creating statistical plots (17-20)...")
    plot_statistics_and_correlations(stages, img, out_dir, dataset, ecg_id)

    # 1D ↔ 2D comparisons (row-wise)
    print("  Creating 1D↔2D comparison plots (21-23)...")
    plot_pairwise_1d_2d(stages, img, fs, out_dir, dataset, ecg_id, row=row)

    print(f"\n✔ Finished. Created {len(list(out_dir.glob('*.png')))} plots.")
    print(f"  Output folder:\n{out_dir}\n")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full visual sanity-check for ONE ECG across the pipeline."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ptbxl", "sami_trop", "samitrop", "sami", "code15", "code_15"],
        help="Which dataset the ECG belongs to.",
    )
    parser.add_argument(
        "--id",
        type=int,
        required=True,
        help="ecg_id / exam_id for the chosen dataset.",
    )
    parser.add_argument(
        "--ptbxl-raw-rel",
        help=(
            "PTB-XL ONLY: WFDB relative path from data/raw/ptbxl "
            "(e.g. records100/00000/00001_lr). "
            "If omitted, it will be looked up in ptbxl_database.csv."
        ),
    )
    parser.add_argument(
        "--row",
        type=int,
        default=10,
        help="Image row index used for 1D↔2D row comparisons (default: 10).",
    )

    args = parser.parse_args()
    validate_single_ecg(
        dataset=args.dataset,
        ecg_id=args.id,
        ptbxl_raw_rel=args.ptbxl_raw_rel,
        row=args.row,
    )


if __name__ == "__main__":
    main()