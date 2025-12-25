# tests/validate_single_ecg.py
"""
Visual sanity-check for ONE ECG across the full pipeline:
raw → WFDB (converted) → baseline-removed → resampled (500 Hz & 100 Hz) → fixed-length → z-scored (500 Hz only) → 2D image.

Supported datasets:
- ptbxl     : Raw WFDB under data/raw/ptbxl/ + Converted WFDB under data/official_wfdb/ptbxl/ + 1D under data/processed/1d_signals_500hz/ptbxl/ and 1d_signals_100hz/ptbxl/ + 2D under data/processed/2d_images/ptbxl/
- sami_trop : Raw HDF5 under data/raw/sami_trop/ + Converted WFDB under data/official_wfdb/sami_trop/ + 1D under 1d_signals_500hz/sami_trop/ and 1d_signals_100hz/sami_trop/ + 2D under 2d_images/sami_trop/
- code15    : Raw sharded HDF5 under data/raw/code15/ + Converted WFDB under data/official_wfdb/code15/ + 1D under 1d_signals_500hz/code15/ and 1d_signals_100hz/code15/ + 2D under 2d_images/code15/

Generates plots in: tests/verification_outputs/pipeline/<dataset>/ecg_<ID>/
    01_raw_lead1.png
    02_wfdb_lead1.png
    03_baseline_removed_lead1.png (if baseline applied)
    04_resampled_500hz_lead1.png
    05_fixed_10s_500hz_lead1.png
    06_zscore_500hz_lead1.png
    07_resampled_100hz_lead1.png
    08_fixed_10s_100hz_lead1.png

    Comparisons:
    09_raw_vs_wfdb_lead1.png
    10_raw_vs_500hz_lead1.png
    11_raw_vs_100hz_lead1.png
    12_wfdb_vs_500hz_lead1.png
    13_wfdb_vs_100hz_lead1.png
    14_500hz_vs_100hz_lead1.png

    Image:
    15_image_channel1.png
    16_image_channel2.png
    17_image_channel3.png
    18_image_row_comparison.png (1D z-scored vs image row)

    Spectrograms:
    19_raw_spectrogram.png
    20_wfdb_spectrogram.png
    21_500hz_spectrogram.png
    22_100hz_spectrogram.png

    12-lead grids:
    23_raw_12leads.png
    24_wfdb_12leads.png
    25_500hz_12leads.png
    26_100hz_12leads.png

Checks for issues: shape mismatches, frequency differences, normalization stats, clipping percentage.

Usage (per dataset/ID):
python -m tests.validate_single_ecg --dataset ptbxl --id 1 --ptbxl-raw-rel records500/00000/00001_hr  # For PTB-XL, provide rel path if needed

Or run for samples from all datasets:
python -m tests.validate_single_ecg --all
"""
# tests/validate_single_ecg.py
"""
Visual sanity-check for ONE ECG across the full pipeline:
raw → WFDB (converted) → baseline-removed → resampled (500 Hz & 100 Hz) → fixed-length → z-scored (500 Hz only) → 2D image.

Handles current project structure:
- PTB-XL WFDB: subfolders (records500/00000/00001_hr.dat)
- Processed 1D: numeric ID (1.npy, 2.npy, ...)
- Processed 2D: numeric ID (1_img.npy) or padded (00001_hr_img.npy)

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

OUT_DIR_BASE = Path("tests/verification_outputs")
TARGET_FS_IMAGE = 500.0
TARGET_FS_FM = 100.0
TARGET_DURATION_SEC = 10.0

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_raw_signal(dataset: str, id_val: str, ptbxl_raw_rel: str = None) -> tuple[np.ndarray, float]:
    """Load raw signal from original raw folder."""
    if dataset == 'ptbxl':
        if ptbxl_raw_rel is None:
            df = pd.read_csv(Path("data/raw/ptbxl/ptbxl_database.csv"))
            row = df[df['ecg_id'] == int(id_val)].iloc[0]
            ptbxl_raw_rel = row['filename_hr']  # Prefer high-res
        path = Path("data/raw/ptbxl") / ptbxl_raw_rel
        record = wfdb.rdrecord(str(path))
        return record.p_signal.astype(np.float32), float(record.fs)
    
    elif dataset == 'sami_trop':
        csv_path = Path("data/raw/sami_trop/exams.csv")
        df = pd.read_csv(csv_path)
        exam_id = int(id_val)
        if exam_id not in df['exam_id'].values:
            raise ValueError(f"SaMi-Trop exam_id {exam_id} not found")
        h5_idx = df[df['exam_id'] == exam_id].index[0]
        h5 = h5py.File(Path("data/raw/sami_trop/exams.hdf5"), 'r')
        signal = h5['tracings'][h5_idx].astype(np.float32)
        h5.close()
        return signal, 400.0
    
    elif dataset == 'code15':
        df = pd.read_csv(Path("data/raw/code15/exams.csv"))
        exam_id = int(id_val)
        row = df[df['exam_id'] == exam_id].iloc[0]
        trace_file = row['trace_file']
        h5 = h5py.File(Path("data/raw/code15") / trace_file, 'r')
        idx = np.where(h5['exam_id'][:] == exam_id)[0][0]
        signal = h5['tracings'][idx].astype(np.float32)
        h5.close()
        return signal, 400.0
    
    raise ValueError(f"Unsupported dataset: {dataset}")

def load_wfdb_signal(dataset: str, id_val: str) -> tuple[np.ndarray, float]:
    """Load converted WFDB signal. Handles PTB-XL subfolder structure."""
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
        raise FileNotFoundError(f"PTB-XL WFDB record not found for ID {id_val}")
    
    else:
        path = Path(f"data/official_wfdb/{dataset}/{id_val}")
        record = wfdb.rdrecord(str(path))
        return record.p_signal.astype(np.float32), float(record.fs)

def load_processed_500hz(dataset: str, id_val: str) -> np.ndarray:
    """Load processed 500 Hz signal – supports both naming styles."""
    numeric_id = str(int(id_val))
    if dataset == 'ptbxl':
        candidates = [
            f"{int(id_val):05d}_hr.npy",
            f"{int(id_val):05d}_lr.npy",
            f"{numeric_id}.npy"
        ]
    else:
        candidates = [f"{id_val}.npy"]
    
    for fn in candidates:
        path = Path(f"data/processed/1d_signals_500hz/{dataset}/{fn}")
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(f"500 Hz processed not found. Tried: {candidates}")

def load_processed_100hz(dataset: str, id_val: str) -> np.ndarray:
    """Load processed 100 Hz signal – supports both naming styles."""
    numeric_id = str(int(id_val))
    if dataset == 'ptbxl':
        candidates = [
            f"{int(id_val):05d}_hr.npy",
            f"{int(id_val):05d}_lr.npy",
            f"{numeric_id}.npy"
        ]
    else:
        candidates = [f"{id_val}.npy"]
    
    for fn in candidates:
        path = Path(f"data/processed/1d_signals_100hz/{dataset}/{fn}")
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(f"100 Hz processed not found. Tried: {candidates}")

def load_image(dataset: str, id_val: str) -> np.ndarray:
    """Load 2D image – supports both naming styles."""
    numeric_id = str(int(id_val))
    if dataset == 'ptbxl':
        candidates = [
            f"{int(id_val):05d}_hr_img.npy",
            f"{int(id_val):05d}_lr_img.npy",
            f"{numeric_id}_img.npy",
            f"{numeric_id}.npy"  # fallback
        ]
    else:
        candidates = [f"{id_val}_img.npy"]
    
    for fn in candidates:
        path = Path(f"data/processed/2d_images/{dataset}/{fn}")
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(f"Image not found. Tried: {candidates}")

def simulate_pipeline_steps(signal: np.ndarray, fs: float, dataset: str) -> dict:
    if dataset == 'ptbxl':
        baseline_removed = remove_baseline(signal, fs, 'bandpass', low_cut_hz=0.5, high_cut_hz=45.0, order=4)
    elif dataset == 'sami_trop':
        baseline_removed = remove_baseline(signal, fs, 'moving_average', window_seconds=0.2)
    else:
        baseline_removed = signal

    resampled_500, _ = resample_ecg(baseline_removed, fs, TARGET_FS_IMAGE)
    fixed_500 = pad_or_trim(resampled_500, int(TARGET_DURATION_SEC * TARGET_FS_IMAGE))
    zscored_500 = normalize_dataset(fixed_500)

    resampled_100, _ = resample_ecg(baseline_removed, fs, TARGET_FS_FM)
    fixed_100 = pad_or_trim(resampled_100, int(TARGET_DURATION_SEC * TARGET_FS_FM))

    return {
        'baseline_removed': baseline_removed,
        'resampled_500': resampled_500,
        'fixed_500': fixed_500,
        'zscored_500': zscored_500,
        'resampled_100': resampled_100,
        'fixed_100': fixed_100
    }

def check_issues(raw_signal, wfdb_signal, processed_500, processed_100, image, raw_fs, wfdb_fs):
    assert raw_signal.shape[1] == 12, "Raw not 12 leads"
    assert wfdb_signal.shape[1] == 12, "WFDB not 12 leads"
    assert processed_500.shape == (5000, 12), "500Hz wrong shape"
    assert processed_100.shape == (1000, 12), "100Hz wrong shape"
    assert image.shape == (3, 24, 2048), "Image wrong shape"

    if raw_fs != wfdb_fs:
        print("Warning: Raw FS != WFDB FS")

    means = np.mean(processed_500, axis=0)
    stds = np.std(processed_500, axis=0)
    if not np.allclose(means, 0, atol=1e-6) or not np.allclose(stds, 1, atol=1e-6):
        print("Warning: 500Hz not properly normalized")

    clipped_count = np.sum((processed_500 <= -3) | (processed_500 >= 3))
    clipped_pct = (clipped_count / processed_500.size) * 100
    if clipped_pct > 5:
        print(f"Warning: High clipping {clipped_pct:.1f}% in 500Hz")

    print("✅ No major issues detected.")

def plot_signal_comparison(title: str, signals: list, fs_list: list, labels: list, lead: int = 0, fname: str = None):
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
        plt.savefig(fname)
        plt.close()

def plot_12leads(signal: np.ndarray, fs: float, title: str, fname: str):
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    axes = axes.flatten()
    t = np.arange(signal.shape[0]) / fs
    for lead in range(12):
        axes[lead].plot(t, signal[:, lead])
        axes[lead].set_title(f"Lead {lead+1}")
        axes[lead].grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_spectrogram(signal: np.ndarray, fs: float, title: str, fname: str, lead: int = 0):
    f, t, Sxx = spectrogram(signal[:, lead], fs=fs)
    plt.figure(figsize=(12, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10))
    plt.title(title)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="Power (dB)")
    plt.savefig(fname)
    plt.close()

def plot_image_channels(image: np.ndarray, title_prefix: str, out_dir: Path):
    for ch in range(3):
        plt.figure(figsize=(12, 4))
        plt.imshow(image[ch], aspect='auto', cmap='viridis')
        plt.title(f"{title_prefix} Channel {ch+1}")
        plt.colorbar()
        plt.savefig(out_dir / f"{15 + ch}_image_channel{ch+1}.png")
        plt.close()

def plot_image_row_comparison(zscored_500: np.ndarray, image: np.ndarray, row: int, fs: float, fname: str):
    t = np.arange(zscored_500.shape[0]) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, zscored_500[:, 0], label="Z-scored 500 Hz (Lead I)")
    plt.plot(np.linspace(0, 10, image.shape[2]), image[0, row, :], label=f"Image Channel 1 Row {row}")
    plt.title("Z-scored 500 Hz vs Image Row Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(fname)
    plt.close()

def validate_single_ecg(dataset: str, ecg_id: str, ptbxl_raw_rel: str = None, row: int = 10):
    out_dir = OUT_DIR_BASE / "pipeline" / dataset / f"ecg_{ecg_id}"
    ensure_dir(out_dir)

    print(f"📂 Outputs: {out_dir}")

    # Load raw
    raw_signal, raw_fs = load_raw_signal(dataset, ecg_id, ptbxl_raw_rel)

    # Load WFDB
    wfdb_signal, wfdb_fs = load_wfdb_signal(dataset, ecg_id)

    # Simulate pipeline
    steps = simulate_pipeline_steps(wfdb_signal, wfdb_fs, dataset)

    # Load actual processed
    processed_500 = load_processed_500hz(dataset, ecg_id)
    processed_100 = load_processed_100hz(dataset, ecg_id)

    # Load image
    image = load_image(dataset, ecg_id)

    # Checks
    check_issues(raw_signal, wfdb_signal, processed_500, processed_100, image, raw_fs, wfdb_fs)

    # Plots
    plot_signal_comparison("Raw Lead I", [raw_signal], [raw_fs], ["Raw"], fname=out_dir / "01_raw_lead1.png")
    plot_signal_comparison("WFDB Lead I", [wfdb_signal], [wfdb_fs], ["WFDB"], fname=out_dir / "02_wfdb_lead1.png")
    if 'baseline_removed' in steps:
        plot_signal_comparison("Baseline Removed Lead I", [steps['baseline_removed']], [wfdb_fs], ["Baseline Removed"], fname=out_dir / "03_baseline_removed_lead1.png")

    plot_signal_comparison("Resampled 500 Hz Lead I", [steps['resampled_500']], [TARGET_FS_IMAGE], ["Resampled 500 Hz"], fname=out_dir / "04_resampled_500hz_lead1.png")
    plot_signal_comparison("Fixed 10s 500 Hz Lead I", [steps['fixed_500']], [TARGET_FS_IMAGE], ["Fixed 500 Hz"], fname=out_dir / "05_fixed_10s_500hz_lead1.png")
    plot_signal_comparison("Z-scored 500 Hz Lead I", [steps['zscored_500']], [TARGET_FS_IMAGE], ["Z-scored 500 Hz"], fname=out_dir / "06_zscore_500hz_lead1.png")

    plot_signal_comparison("Resampled 100 Hz Lead I", [steps['resampled_100']], [TARGET_FS_FM], ["Resampled 100 Hz"], fname=out_dir / "07_resampled_100hz_lead1.png")
    plot_signal_comparison("Fixed 10s 100 Hz Lead I", [steps['fixed_100']], [TARGET_FS_FM], ["Fixed 100 Hz"], fname=out_dir / "08_fixed_10s_100hz_lead1.png")

    # Comparisons
    plot_signal_comparison("Raw vs WFDB", [raw_signal, wfdb_signal], [raw_fs, wfdb_fs], ["Raw", "WFDB"], fname=out_dir / "09_raw_vs_wfdb_lead1.png")
    plot_signal_comparison("Raw vs Processed 500 Hz", [raw_signal, processed_500], [raw_fs, TARGET_FS_IMAGE], ["Raw", "Processed 500 Hz"], fname=out_dir / "10_raw_vs_500hz_lead1.png")
    plot_signal_comparison("Raw vs Processed 100 Hz", [raw_signal, processed_100], [raw_fs, TARGET_FS_FM], ["Raw", "Processed 100 Hz"], fname=out_dir / "11_raw_vs_100hz_lead1.png")
    plot_signal_comparison("WFDB vs Processed 500 Hz", [wfdb_signal, processed_500], [wfdb_fs, TARGET_FS_IMAGE], ["WFDB", "Processed 500 Hz"], fname=out_dir / "12_wfdb_vs_500hz_lead1.png")
    plot_signal_comparison("WFDB vs Processed 100 Hz", [wfdb_signal, processed_100], [wfdb_fs, TARGET_FS_FM], ["WFDB", "Processed 100 Hz"], fname=out_dir / "13_wfdb_vs_100hz_lead1.png")
    plot_signal_comparison("Processed 500 Hz vs 100 Hz", [processed_500, processed_100], [TARGET_FS_IMAGE, TARGET_FS_FM], ["500 Hz", "100 Hz"], fname=out_dir / "14_500hz_vs_100hz_lead1.png")

    # Image
    plot_image_channels(image, "Image", out_dir)
    plot_image_row_comparison(processed_500, image, row, TARGET_FS_IMAGE, out_dir / "18_image_row_comparison.png")

    # Spectrograms
    plot_spectrogram(raw_signal, raw_fs, "Raw Spectrogram", out_dir / "19_raw_spectrogram.png")
    plot_spectrogram(wfdb_signal, wfdb_fs, "WFDB Spectrogram", out_dir / "20_wfdb_spectrogram.png")
    plot_spectrogram(processed_500, TARGET_FS_IMAGE, "500 Hz Spectrogram", out_dir / "21_500hz_spectrogram.png")
    plot_spectrogram(processed_100, TARGET_FS_FM, "100 Hz Spectrogram", out_dir / "22_100hz_spectrogram.png")

    # 12-lead grids
    plot_12leads(raw_signal, raw_fs, "Raw 12-Leads", out_dir / "23_raw_12leads.png")
    plot_12leads(wfdb_signal, wfdb_fs, "WFDB 12-Leads", out_dir / "24_wfdb_12leads.png")
    plot_12leads(processed_500, TARGET_FS_IMAGE, "Processed 500 Hz 12-Leads", out_dir / "25_500hz_12leads.png")
    plot_12leads(processed_100, TARGET_FS_FM, "Processed 100 Hz 12-Leads", out_dir / "26_100hz_12leads.png")

    print(f"✅ Validation complete! Check {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ECG pipeline for a single record.")
    parser.add_argument("--dataset", required=False, choices=['ptbxl', 'sami_trop', 'code15'])
    parser.add_argument("--id", type=str, required=False)
    parser.add_argument("--row", type=int, default=10)
    parser.add_argument("--all", action='store_true')

    args = parser.parse_args()

    if args.all:
        samples = {
            'ptbxl': '1',
            'sami_trop': '24028',
            'code15': '12345'  # Replace with real ID
        }
        for ds, id_val in samples.items():
            validate_single_ecg(ds, id_val, row=args.row)
    elif args.dataset and args.id:
        validate_single_ecg(args.dataset, args.id, row=args.row)
    else:
        parser.print_help()