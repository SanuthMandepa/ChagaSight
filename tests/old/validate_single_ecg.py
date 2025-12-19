# tests/validate_single_ecg.py

"""
Visual sanity-check for ONE ECG across the full pipeline:
raw → baseline-removed (if applied) → resampled (500 Hz & 100 Hz) → fixed-length → z-scored (500 Hz only) → 2D image.

Supported datasets:
- ptbxl     : WFDB records under data/raw/ptbxl/ + 1D under data/processed/1d_signals_500hz/ptbxl/ and _100hz/ptbxl/ + 2D under data/processed/2d_images/ptbxl/
- sami_trop : exams.hdf5 under data/raw/sami_trop/ + 1D under _500hz/sami_trop/ and _100hz/sami_trop/ + 2D under 2d_images/sami_trop/
- code15    : sharded HDF5 under data/raw/code15/ + 1D under _500hz/code15/ and _100hz/code15/ + 2D under 2d_images/code15/

Generates plots in: tests/verification_outputs/pipeline/<dataset>/ecg_<ID>/
    01_raw_lead1.png
    02_baseline_removed_lead1.png (if baseline applied)
    03_resampled_500hz_lead1.png
    04_fixed_10s_500hz_lead1.png
    05_zscore_500hz_lead1.png
    06_resampled_100hz_lead1.png
    07_fixed_10s_100hz_lead1.png

    Comparisons:
    08_raw_vs_500hz_lead1.png
    09_raw_vs_100hz_lead1.png
    10_500hz_vs_100hz_lead1.png

    Image:
    11_image_channel1.png
    12_image_channel2.png
    13_image_channel3.png
    14_image_row_comparison.png (1D z-scored vs image row)

    Spectrograms:
    15_raw_spectrogram.png
    16_500hz_spectrogram.png
    17_100hz_spectrogram.png

    12-lead grids:
    18_raw_12leads.png
    19_500hz_12leads.png
    20_100hz_12leads.png
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import h5py
import pandas as pd
from scipy.signal import spectrogram

OUT_DIR_BASE = Path("tests/verification_outputs")
TARGET_FS_IMAGE = 500.0
TARGET_FS_FM = 100.0
TARGET_DURATION_SEC = 10.0


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_raw_signal(dataset: str, id: int, ptbxl_raw_rel: str = None) -> tuple[np.ndarray, float]:
    if dataset == 'ptbxl':
        if ptbxl_raw_rel is None:
            df = pd.read_csv(Path("data/raw/ptbxl/ptbxl_database.csv"))
            row = df[df['ecg_id'] == id].iloc[0]
            ptbxl_raw_rel = row['filename_hr'] if 'filename_hr' in row else row['filename_lr']
        path = Path("data/raw/ptbxl") / ptbxl_raw_rel
        record = wfdb.rdrecord(str(path))
        return record.p_signal.astype(np.float32), float(record.fs)
    
    elif dataset == 'sami_trop':
        # Load metadata to map exam_id → HDF5 index
        csv_path = Path("data/raw/sami_trop/exams.csv")
        df = pd.read_csv(csv_path)
        if id not in df['exam_id'].values:
            raise ValueError(f"SaMi-Trop exam_id {id} not found in exams.csv (valid IDs: {df['exam_id'].min()} to {df['exam_id'].max()})")
        
        h5_idx = df[df['exam_id'] == id].index[0]
        h5 = h5py.File(Path("data/raw/sami_trop/exams.hdf5"), 'r')
        tracings = h5['tracings']
        signal = tracings[h5_idx].astype(np.float32)
        h5.close()
        return signal, 400.0
    
    elif dataset == 'code15':
        df = pd.read_csv(Path("data/raw/code15/exams.csv"))
        row = df[df['exam_id'] == id].iloc[0]
        trace_file = row['trace_file']
        h5 = h5py.File(Path("data/raw/code15") / trace_file, 'r')
        idx = np.where(h5['exam_id'][:] == id)[0][0]
        signal = h5['tracings'][idx].astype(np.float32)
        h5.close()
        return signal, 400.0
    
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_processed_500hz(dataset: str, id: int) -> np.ndarray:
    path = Path(f"data/processed/1d_signals_500hz/{dataset}/{id}.npy")
    if not path.exists():
        raise FileNotFoundError(f"500 Hz processed not found: {path}")
    return np.load(path)


def load_processed_100hz(dataset: str, id: int) -> np.ndarray:
    path = Path(f"data/processed/1d_signals_100hz/{dataset}/{id}.npy")
    if not path.exists():
        raise FileNotFoundError(f"100 Hz processed not found: {path}")
    return np.load(path)


def load_image(dataset: str, id: int) -> np.ndarray:
    path = Path(f"data/processed/2d_images/{dataset}/{id}_img.npy")
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return np.load(path)


def simulate_pipeline_steps(raw_signal: np.ndarray, raw_fs: float, dataset: str) -> dict:
    from src.preprocessing.baseline_removal import remove_baseline
    from src.preprocessing.resample import resample_ecg, pad_or_trim
    from src.preprocessing.normalization import zscore_per_lead

    # Baseline removal (dataset-specific)
    if dataset == 'ptbxl':
        baseline_removed = remove_baseline(raw_signal, raw_fs, 'bandpass', low_cut_hz=0.5, high_cut_hz=45.0, order=4)
    elif dataset == 'sami_trop':
        baseline_removed = remove_baseline(raw_signal, raw_fs, 'moving_average', window_seconds=0.2)
    else:  # code15 – no baseline
        baseline_removed = raw_signal

    # Resample to 500 Hz
    resampled_500, _ = resample_ecg(baseline_removed, raw_fs, TARGET_FS_IMAGE)
    fixed_500 = pad_or_trim(resampled_500, int(TARGET_DURATION_SEC * TARGET_FS_IMAGE))
    zscored_500 = zscore_per_lead(fixed_500)

    # Resample to 100 Hz
    resampled_100, _ = resample_ecg(baseline_removed, raw_fs, TARGET_FS_FM)
    fixed_100 = pad_or_trim(resampled_100, int(TARGET_DURATION_SEC * TARGET_FS_FM))

    return {
        'baseline_removed': baseline_removed,
        'resampled_500': resampled_500,
        'fixed_500': fixed_500,
        'zscored_500': zscored_500,
        'resampled_100': resampled_100,
        'fixed_100': fixed_100
    }


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
        plt.savefig(out_dir / f"{11 + ch}_image_channel{ch+1}.png")
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


def validate_single_ecg(dataset: str, ecg_id: int, ptbxl_raw_rel: str = None, row: int = 10):
    out_dir = OUT_DIR_BASE / "pipeline" / dataset / f"ecg_{ecg_id}"
    ensure_dir(out_dir)

    print(f"📂 Outputs: {out_dir}")

    # Load raw
    raw_signal, raw_fs = load_raw_signal(dataset, ecg_id, ptbxl_raw_rel)

    # Simulate pipeline
    steps = simulate_pipeline_steps(raw_signal, raw_fs, dataset)

    # Load actual processed
    processed_500 = load_processed_500hz(dataset, ecg_id)
    processed_100 = load_processed_100hz(dataset, ecg_id)

    # Load image
    image = load_image(dataset, ecg_id)

    # Plots
    plot_signal_comparison("Raw Lead I", [raw_signal], [raw_fs], ["Raw"], fname=out_dir / "01_raw_lead1.png")
    if 'baseline_removed' in steps:
        plot_signal_comparison("Baseline Removed Lead I", [steps['baseline_removed']], [raw_fs], ["Baseline Removed"], fname=out_dir / "02_baseline_removed_lead1.png")

    plot_signal_comparison("Resampled 500 Hz Lead I", [steps['resampled_500']], [TARGET_FS_IMAGE], ["Resampled 500 Hz"], fname=out_dir / "03_resampled_500hz_lead1.png")
    plot_signal_comparison("Fixed 10s 500 Hz Lead I", [steps['fixed_500']], [TARGET_FS_IMAGE], ["Fixed 500 Hz"], fname=out_dir / "04_fixed_10s_500hz_lead1.png")
    plot_signal_comparison("Z-scored 500 Hz Lead I", [steps['zscored_500']], [TARGET_FS_IMAGE], ["Z-scored 500 Hz"], fname=out_dir / "05_zscore_500hz_lead1.png")

    plot_signal_comparison("Resampled 100 Hz Lead I", [steps['resampled_100']], [TARGET_FS_FM], ["Resampled 100 Hz"], fname=out_dir / "06_resampled_100hz_lead1.png")
    plot_signal_comparison("Fixed 10s 100 Hz Lead I", [steps['fixed_100']], [TARGET_FS_FM], ["Fixed 100 Hz"], fname=out_dir / "07_fixed_10s_100hz_lead1.png")

    # Comparisons
    plot_signal_comparison("Raw vs Processed 500 Hz", [raw_signal, processed_500], [raw_fs, TARGET_FS_IMAGE], ["Raw", "Processed 500 Hz"], fname=out_dir / "08_raw_vs_500hz_lead1.png")
    plot_signal_comparison("Raw vs Processed 100 Hz", [raw_signal, processed_100], [raw_fs, TARGET_FS_FM], ["Raw", "Processed 100 Hz"], fname=out_dir / "09_raw_vs_100hz_lead1.png")
    plot_signal_comparison("Processed 500 Hz vs 100 Hz", [processed_500, processed_100], [TARGET_FS_IMAGE, TARGET_FS_FM], ["500 Hz", "100 Hz"], fname=out_dir / "10_500hz_vs_100hz_lead1.png")

    # Image channels
    plot_image_channels(image, "Image", out_dir)

    # Image row comparison
    plot_image_row_comparison(processed_500, image, row, TARGET_FS_IMAGE, out_dir / "14_image_row_comparison.png")

    # Spectrograms
    plot_spectrogram(raw_signal, raw_fs, "Raw Spectrogram", out_dir / "15_raw_spectrogram.png")
    plot_spectrogram(processed_500, TARGET_FS_IMAGE, "500 Hz Spectrogram", out_dir / "16_500hz_spectrogram.png")
    plot_spectrogram(processed_100, TARGET_FS_FM, "100 Hz Spectrogram", out_dir / "17_100hz_spectrogram.png")

    # 12-lead grids
    plot_12leads(raw_signal, raw_fs, "Raw 12-Leads", out_dir / "18_raw_12leads.png")
    plot_12leads(processed_500, TARGET_FS_IMAGE, "Processed 500 Hz 12-Leads", out_dir / "19_500hz_12leads.png")
    plot_12leads(processed_100, TARGET_FS_FM, "Processed 100 Hz 12-Leads", out_dir / "20_100hz_12leads.png")

    print(f"✅ Validation complete! Check {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ECG pipeline for a single record.")
    parser.add_argument("--dataset", required=True, choices=['ptbxl', 'sami_trop', 'code15'])
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--ptbxl-raw-rel", required=False)
    parser.add_argument("--row", type=int, default=10)

    args = parser.parse_args()
    validate_single_ecg(args.dataset, args.id, args.ptbxl_raw_rel, args.row)