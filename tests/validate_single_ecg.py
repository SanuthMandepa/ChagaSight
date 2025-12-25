# tests/validate_single_ecg.py
"""
Visual sanity-check for ONE ECG across the full pipeline using the current clean setup.

Current pipeline (after switching to build_2d_images_wfdb.py):
- Loads WFDB signal directly
- Applies baseline removal, resampling, z-score, clipping
- Generates 2D contour image and saves 100 Hz raw signal
- Does NOT save intermediate 500 Hz z-scored signals

This script:
- Loads raw + WFDB
- Simulates the exact pipeline (same as build script)
- Loads the actual saved 100 Hz signal and 2D image
- Compares simulated vs saved
- Generates all 26 diagnostic plots

Usage:
python -m tests.validate_single_ecg --dataset ptbxl --id 1
python -m tests.validate_single_ecg --all
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import h5py
import pandas as pd
from scipy.signal import spectrogram

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import normalize_dataset
from src.preprocessing.image_embedding import ecg_to_contour_image

OUT_DIR_BASE = Path("tests/verification_outputs_2/pipeline")
TARGET_FS_IMAGE = 500.0
TARGET_FS_FM = 100.0
TARGET_DURATION_SEC = 10.0
TARGET_SAMPLES_500 = int(TARGET_DURATION_SEC * TARGET_FS_IMAGE)  # 5000
TARGET_SAMPLES_100 = int(TARGET_DURATION_SEC * TARGET_FS_FM)      # 1000
TARGET_WIDTH = 2048
CLIP_RANGE = (-3.0, 3.0)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def remove_zero_padding(signal: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    non_zero = np.any(np.abs(signal) > threshold, axis=1)
    start = np.argmax(non_zero)
    end = len(non_zero) - np.argmax(non_zero[::-1])
    return signal[start:end] if start < end else signal

def load_raw_signal(dataset: str, id_val: str, ptbxl_raw_rel: str = None) -> tuple[np.ndarray, float]:
    if dataset == 'ptbxl':
        if ptbxl_raw_rel is None:
            df = pd.read_csv(Path("data/raw/ptbxl/ptbxl_database.csv"))
            row = df[df['ecg_id'] == int(id_val)].iloc[0]
            ptbxl_raw_rel = row['filename_hr']
        path = Path("data/raw/ptbxl") / ptbxl_raw_rel
        record = wfdb.rdrecord(str(path))
        return record.p_signal.astype(np.float32), float(record.fs)
    
    elif dataset == 'sami_trop':
        csv_path = Path("data/raw/sami_trop/exams.csv")
        df = pd.read_csv(csv_path)
        exam_id = int(id_val)
        h5_idx = df[df['exam_id'] == exam_id].index[0]
        with h5py.File(Path("data/raw/sami_trop/exams.hdf5"), 'r') as h5:
            signal = h5['tracings'][h5_idx].astype(np.float32)
        return signal, 400.0
    
    elif dataset == 'code15':
        df = pd.read_csv(Path("data/raw/code15/exams.csv"))
        row = df[df['exam_id'] == int(id_val)].iloc[0]
        trace_file = row['trace_file']
        with h5py.File(Path("data/raw/code15") / trace_file, 'r') as h5:
            idx = np.where(h5['exam_id'][:] == int(id_val))[0][0]
            signal = h5['tracings'][idx].astype(np.float32)
        return signal, 400.0
    
    raise ValueError(f"Unsupported dataset: {dataset}")

def load_wfdb_signal(dataset: str, id_val: str) -> tuple[np.ndarray, float]:
    if dataset == 'ptbxl':
        numeric_id = int(id_val)
        subfolder = f"{numeric_id // 1000:05d}"
        candidates = [
            Path(f"data/official_wfdb/ptbxl/records500/{subfolder}/{numeric_id:05d}_hr"),
            Path(f"data/official_wfdb/ptbxl/records100/{subfolder}/{numeric_id:05d}_lr")
        ]
        for path in candidates:
            if path.with_suffix('.hea').exists():
                record = wfdb.rdrecord(str(path))
                return record.p_signal.astype(np.float32), float(record.fs)
        raise FileNotFoundError(f"PTB-XL WFDB not found for ID {id_val}")
    else:
        path = Path(f"data/official_wfdb/{dataset}/{id_val}")
        record = wfdb.rdrecord(str(path))
        return record.p_signal.astype(np.float32), float(record.fs)

def load_saved_100hz(dataset: str, id_val: str) -> np.ndarray:
    if dataset == 'ptbxl':
        file_id = f"{int(id_val):05d}_hr"
    else:
        file_id = id_val
    path = Path(f"data/processed/1d_signals_100hz/{dataset}/{file_id}.npy")
    if not path.exists():
        raise FileNotFoundError(f"Saved 100 Hz signal not found: {path}")
    return np.load(path)

def load_saved_image(dataset: str, id_val: str) -> np.ndarray:
    if dataset == 'ptbxl':
        file_id = f"{int(id_val):05d}_hr"
    else:
        file_id = id_val
    path = Path(f"data/processed/2d_images/{dataset}/{file_id}_img.npy")
    if not path.exists():
        raise FileNotFoundError(f"Saved image not found: {path}")
    return np.load(path)

def simulate_pipeline(wfdb_signal: np.ndarray, wfdb_fs: float, dataset: str) -> dict:
    """Reproduce the exact processing done in build_2d_images_wfdb.py"""
    signal = wfdb_signal
    if dataset in ['sami_trop', 'code15']:
        signal = remove_zero_padding(signal)

    # Baseline removal
    if dataset == 'ptbxl':
        filtered = remove_baseline(signal, wfdb_fs, 'bandpass', low_cut_hz=0.5, high_cut_hz=45.0, order=4)
    elif dataset == 'sami_trop':
        filtered = remove_baseline(signal, wfdb_fs, 'moving_average', window_seconds=0.2)
    else:
        filtered = signal

    # 500 Hz path
    signal_500, _ = resample_ecg(filtered, wfdb_fs, TARGET_FS_IMAGE)
    signal_500 = pad_or_trim(signal_500, TARGET_SAMPLES_500)
    zscored_500 = normalize_dataset(signal_500)
    zscored_500 = np.clip(zscored_500, CLIP_RANGE[0], CLIP_RANGE[1])

    # 100 Hz path
    signal_100, _ = resample_ecg(filtered, wfdb_fs, TARGET_FS_FM)
    signal_100 = pad_or_trim(signal_100, TARGET_SAMPLES_100)

    return {
        'filtered': filtered,
        'resampled_500': signal_500,
        'zscored_500': zscored_500,
        'resampled_100': signal_100,
    }

def verify_outputs(steps: dict, saved_100: np.ndarray, saved_image: np.ndarray):
    assert steps['zscored_500'].shape == (TARGET_SAMPLES_500, 12), "Simulated 500Hz wrong shape"
    assert saved_image.shape == (3, 24, TARGET_WIDTH), "Saved image wrong shape"

    means = np.mean(steps['zscored_500'], axis=0)
    stds = np.std(steps['zscored_500'], axis=0)
    if not (np.allclose(means, 0, atol=1e-5) and np.allclose(stds, 1, atol=1e-5)):
        print("Warning: Simulated 500Hz not perfectly z-scored")

    clipped = np.mean((steps['zscored_500'] <= CLIP_RANGE[0]) | (steps['zscored_500'] >= CLIP_RANGE[1])) * 100
    if clipped > 5:
        print(f"High clipping in simulated 500Hz: {clipped:.1f}%")

    assert np.allclose(steps['resampled_100'], saved_100, atol=1e-4), "100 Hz mismatch"

    sim_image = ecg_to_contour_image(steps['zscored_500'], target_width=TARGET_WIDTH, clip_range=CLIP_RANGE)
    assert np.allclose(sim_image, saved_image, atol=1e-3), "Image mismatch"

    print("✅ Pipeline verification successful")

# Plotting functions (unchanged)
def plot_signal_comparison(title: str, signals: list, fs_list: list, labels: list, lead: int = 0, fname: Path | None = None):
    plt.figure(figsize=(12, 4))
    for sig, fs, label in zip(signals, fs_list, labels):
        t = np.arange(sig.shape[0]) / fs
        plt.plot(t, sig[:, lead], label=label)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    if fname:
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()

def plot_12leads(signal: np.ndarray, fs: float, title: str, fname: Path):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    t = np.arange(signal.shape[0]) / fs
    for lead in range(12):
        axes[lead].plot(t, signal[:, lead])
        axes[lead].set_title(f"Lead {lead+1}")
        axes[lead].grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def plot_spectrogram(signal: np.ndarray, fs: float, title: str, fname: Path, lead: int = 0):
    f, t, Sxx = spectrogram(signal[:, lead], fs=fs, nperseg=256)
    plt.figure(figsize=(12, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.title(title)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def plot_image_channels(image: np.ndarray, out_dir: Path):
    for ch in range(3):
        plt.figure(figsize=(12, 4))
        plt.imshow(image[ch], aspect='auto', cmap='viridis')
        plt.title(f"Generated Contour Image - Channel {ch+1}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out_dir / f"{15 + ch}_image_channel{ch+1}.png", dpi=150)
        plt.close()

def plot_image_row_comparison(zscored_500: np.ndarray, image: np.ndarray, row: int, fname: Path):
    t = np.arange(zscored_500.shape[0]) / TARGET_FS_IMAGE
    plt.figure(figsize=(12, 4))
    plt.plot(t, zscored_500[:, 0], label="Simulated Z-scored 500 Hz (Lead I)")
    plt.plot(np.linspace(0, 10, image.shape[2]), image[0, row, :], label=f"Image Channel 1 Row {row}")
    plt.title("Simulated 500 Hz vs Generated Image Row")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def validate_single_ecg(dataset: str, ecg_id: str, row: int = 10):
    out_dir = OUT_DIR_BASE / dataset / f"ecg_{ecg_id}"
    ensure_dir(out_dir)
    print(f"📂 Outputs: {out_dir}")

    # Load signals
    raw_signal, raw_fs = load_raw_signal(dataset, ecg_id)
    wfdb_signal, wfdb_fs = load_wfdb_signal(dataset, ecg_id)

    # Simulate the exact build pipeline
    steps = simulate_pipeline(wfdb_signal, wfdb_fs, dataset)

    # Load actual saved outputs
    try:
        saved_100 = load_saved_100hz(dataset, ecg_id)
    except FileNotFoundError:
        print("Warning: Saved 100 Hz not found, skipping comparison")
        saved_100 = None

    try:
        saved_image = load_saved_image(dataset, ecg_id)
    except FileNotFoundError:
        print("Warning: Saved image not found, skipping comparison")
        saved_image = None

    # Final checks
    if saved_100 is not None and saved_image is not None:
        verify_outputs(steps, saved_100, saved_image)
    elif saved_100 is not None or saved_image is not None:
        print("Partial verification: Some saved files missing")
    else:
        print("No saved files found; simulation only")

    # === PLOTS ===
    # 01-02: Raw and WFDB
    plot_signal_comparison("01 Raw Lead I", [raw_signal], [raw_fs], ["Raw"], fname=out_dir / "01_raw_lead1.png")
    plot_signal_comparison("02 WFDB Lead I", [wfdb_signal], [wfdb_fs], ["WFDB"], fname=out_dir / "02_wfdb_lead1.png")

    # 03: Baseline removed
    plot_signal_comparison("03 Baseline Removed Lead I", [steps['filtered']], [wfdb_fs], ["Filtered"], fname=out_dir / "03_baseline_removed_lead1.png")

    # 04-06: 500 Hz path (simulated)
    plot_signal_comparison("04 Resampled 500 Hz Lead I", [steps['resampled_500']], [TARGET_FS_IMAGE], ["Resampled"], fname=out_dir / "04_resampled_500hz_lead1.png")
    plot_signal_comparison("05 Fixed 10s 500 Hz Lead I", [steps['resampled_500']], [TARGET_FS_IMAGE], ["Fixed Length"], fname=out_dir / "05_fixed_10s_500hz_lead1.png")
    plot_signal_comparison("06 Z-scored + Clipped 500 Hz Lead I", [steps['zscored_500']], [TARGET_FS_IMAGE], ["Final 500 Hz"], fname=out_dir / "06_zscore_clipped_500hz_lead1.png")

    # 07-08: 100 Hz path
    plot_signal_comparison("07 Simulated 100 Hz Lead I", [steps['resampled_100']], [TARGET_FS_FM], ["Simulated"], fname=out_dir / "07_resampled_100hz_lead1.png")
    if saved_100 is not None:
        plot_signal_comparison("08 Saved 100 Hz Lead I", [saved_100], [TARGET_FS_FM], ["Saved FM Signal"], fname=out_dir / "08_saved_100hz_lead1.png")

    # Comparisons
    plot_signal_comparison("09 Raw vs WFDB", [raw_signal, wfdb_signal], [raw_fs, wfdb_fs], ["Raw", "WFDB"], fname=out_dir / "09_raw_vs_wfdb_lead1.png")
    plot_signal_comparison("10 WFDB vs Simulated 500 Hz", [wfdb_signal, steps['zscored_500']], [wfdb_fs, TARGET_FS_IMAGE], ["WFDB", "Simulated 500 Hz"], fname=out_dir / "10_wfdb_vs_500hz_lead1.png")
    if saved_100 is not None:
        plot_signal_comparison("11 WFDB vs Saved 100 Hz", [wfdb_signal, saved_100], [wfdb_fs, TARGET_FS_FM], ["WFDB", "Saved 100 Hz"], fname=out_dir / "11_wfdb_vs_100hz_lead1.png")
        plot_signal_comparison("12 Simulated 500 Hz vs Saved 100 Hz", [steps['zscored_500'], saved_100], [TARGET_FS_IMAGE, TARGET_FS_FM], ["500 Hz", "100 Hz"], fname=out_dir / "12_500hz_vs_100hz_lead1.png")

    # Image plots
    if saved_image is not None:
        plot_image_channels(saved_image, out_dir)
        plot_image_row_comparison(steps['zscored_500'], saved_image, row, out_dir / "18_image_row_comparison.png")

    # Spectrograms
    plot_spectrogram(raw_signal, raw_fs, "19 Raw Spectrogram", out_dir / "19_raw_spectrogram.png")
    plot_spectrogram(wfdb_signal, wfdb_fs, "20 WFDB Spectrogram", out_dir / "20_wfdb_spectrogram.png")
    plot_spectrogram(steps['zscored_500'], TARGET_FS_IMAGE, "21 Simulated 500 Hz Spectrogram", out_dir / "21_500hz_spectrogram.png")
    if saved_100 is not None:
        plot_spectrogram(saved_100, TARGET_FS_FM, "22 Saved 100 Hz Spectrogram", out_dir / "22_100hz_spectrogram.png")

    # 12-lead grids
    plot_12leads(raw_signal, raw_fs, "23 Raw 12-Leads", out_dir / "23_raw_12leads.png")
    plot_12leads(wfdb_signal, wfdb_fs, "24 WFDB 12-Leads", out_dir / "24_wfdb_12leads.png")
    plot_12leads(steps['zscored_500'], TARGET_FS_IMAGE, "25 Simulated 500 Hz 12-Leads", out_dir / "25_500hz_12leads.png")
    if saved_100 is not None:
        plot_12leads(saved_100, TARGET_FS_FM, "26 Saved 100 Hz 12-Leads", out_dir / "26_100hz_12leads.png")

    print(f"✅ Validation complete for {dataset} ID {ecg_id}! Check folder: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ECG pipeline for a single record.")
    parser.add_argument("--dataset", required=False, choices=['ptbxl', 'sami_trop', 'code15'])
    parser.add_argument("--id", type=str, required=False)
    parser.add_argument("--row", type=int, default=10, help="Row to compare in image")
    parser.add_argument("--all", action='store_true', help="Run on sample from each dataset")

    args = parser.parse_args()

    if args.all:
        samples = {
            'ptbxl': '1',
            'sami_trop': '24028',
            'code15': '12345'  # Replace with a real ID from your CODE-15%
        }
        for ds, eid in samples.items():
            validate_single_ecg(ds, eid, row=args.row)
    elif args.dataset and args.id:
        validate_single_ecg(args.dataset, args.id, row=args.row)
    else:
        parser.print_help()