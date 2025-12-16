"""
PTB-XL Loading and Standardized Preprocessing Interface.

This module provides a high-level generator-based loader for the PTB-XL
dataset using the official WFDB waveform files. It is responsible for
producing clean, uniformly processed ECG signals for downstream tasks.

Dataset Structure (Required):
    data/raw/ptbxl/
        ├── ptbxl_database.csv      # Metadata: IDs, file paths, demographics
        ├── scp_statements.csv       # SCP-ECG diagnostic taxonomy
        ├── records100/              # WFDB signals @ 100 Hz
        └── records500/              # WFDB signals @ 500 Hz

Processing Steps Per Record:
    1. Load WFDB waveform (12-lead ECG)
    2. Apply baseline/noise removal (high-pass, moving average, or band-pass)
    3. Resample to target sampling rate (e.g., 400 Hz)
    4. Pad/trim to fixed duration (e.g., 10 seconds)
    5. Return normalized float32 array (T, 12)

Output Schema:
    Each yielded record is a dictionary:
        {
            "ecg_id": int,
            "signal": np.ndarray (T, 12),   # preprocessed ECG
            "fs": float,                    # final sampling frequency
            "label": float,                 # Chagas label (always 0 for PTB-XL)
        }

Purpose:
    This loader acts as the backbone for:
        • 1D signal preprocessing (preprocess_ptbxl.py)
        • 2D ECG image generation
        • Foundation model pretraining datasets
        • Hybrid signal–image training pipelines

Notes:
    PTB-XL is treated as *fully negative* for Chagas disease due to its
    European origin, consistent with all Chagas detection literature.
"""

from pathlib import Path
from typing import Dict, Generator, Literal, Optional
import warnings

import numpy as np
import pandas as pd
import wfdb  # pip install wfdb

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead
from src.preprocessing.soft_labels import chagas_label_ptbxl


def _load_single_record(
    base_dir: Path,
    rel_path: str,
    fs_target: float,
    duration_sec: float,
    baseline_method: Literal["highpass", "moving_average", "bandpass"] = "highpass",
    baseline_kwargs: Optional[Dict] = None,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Dict:
    """
    Load a single PTB-XL record from WFDB, preprocess, and return dict.

    Parameters
    ----------
    base_dir : Path
        Base directory for PTB-XL (contains ptbxl_database.csv).
    rel_path : str
        Relative path to WFDB record (e.g. 'records100/00000/00001_lr').
    fs_target : float
        Target sampling rate.
    duration_sec : float
        Target duration in seconds.
    baseline_method : {"highpass", "moving_average", "bandpass"}
        Baseline removal method.
    baseline_kwargs : dict, optional
        Additional kwargs for baseline removal.
    normalize : bool
        Whether to apply z-score normalization.
    eps : float
        Small constant for numerical stability in normalization.

    Returns
    -------
    dict
        {
          "ecg_id": int,
          "signal": np.ndarray (T, 12),
          "fs": fs_target,
          "label": float
        }
    """
    if baseline_kwargs is None:
        baseline_kwargs = {}
    
    try:
        record_path = base_dir / rel_path
        # wfdb will handle .dat/.hea automatically
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal.astype(np.float32)  # (T, 12)
        fs_in = float(record.fs)
        
        if signal.shape[1] != 12:
            warnings.warn(f"Expected 12 leads, got {signal.shape[1]}")
            # Handle potentially missing leads by zero-padding
            if signal.shape[1] < 12:
                padded = np.zeros((signal.shape[0], 12), dtype=np.float32)
                padded[:, :signal.shape[1]] = signal
                signal = padded
        
    except Exception as e:
        raise IOError(f"Failed to load record {rel_path}: {e}")

    # Remove baseline
    signal = remove_baseline(signal, fs=fs_in, method=baseline_method, **baseline_kwargs)

    # Resample (now returns tuple)
    signal, fs_rs = resample_ecg(signal, fs_in=fs_in, fs_out=fs_target)

    # Pad/trim
    target_len = int(round(duration_sec * fs_rs))
    signal = pad_or_trim(signal, target_length=target_len)

    # Normalize if requested
    if normalize:
        signal = zscore_per_lead(signal, eps=eps)

    return {
        "signal": signal.astype(np.float32),
        "fs": fs_rs,
        "label": chagas_label_ptbxl(),
    }


def iter_ptbxl_records(
    ptbxl_root: str | Path,
    use_low_res: bool = True,
    fs_target: float = 400.0,
    duration_sec: float = 10.0,
    baseline_method: Literal["highpass", "moving_average", "bandpass"] = "highpass",
    baseline_kwargs: Optional[Dict] = None,
    normalize: bool = True,
    max_records: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Generator over PTB-XL records, yielding preprocessed ECG dicts.

    Parameters
    ----------
    ptbxl_root : str | Path
        Path to PTB-XL root directory (contains ptbxl_database.csv).
    use_low_res : bool
        If True, use 100 Hz records (`records100`); otherwise use 500 Hz.
    fs_target : float
        Target sampling frequency for preprocessing.
    duration_sec : float
        Target duration in seconds.
    baseline_method : {"highpass", "moving_average", "bandpass"}
        Baseline removal method.
    baseline_kwargs : dict, optional
        Additional kwargs for baseline removal.
    normalize : bool
        Whether to apply z-score normalization.
    max_records : int, optional
        Maximum number of records to yield (for testing).

    Yields
    ------
    dict
        Preprocessed ECG record with keys: ecg_id, signal, fs, label.
    """
    ptbxl_root = Path(ptbxl_root)
    df = pd.read_csv(ptbxl_root / "ptbxl_database.csv")
    
    # Use filename_lr (100 Hz) or filename_hr (500 Hz)
    filename_col = "filename_lr" if use_low_res else "filename_hr"
    
    if baseline_kwargs is None:
        baseline_kwargs = {}
    
    records_yielded = 0
    for _, row in df.iterrows():
        if max_records is not None and records_yielded >= max_records:
            break
            
        rel_path = row[filename_col]
        ecg_id = int(row["ecg_id"])
        
        try:
            rec = _load_single_record(
                base_dir=ptbxl_root,
                rel_path=rel_path,
                fs_target=fs_target,
                duration_sec=duration_sec,
                baseline_method=baseline_method,
                baseline_kwargs=baseline_kwargs,
                normalize=normalize,
            )
            rec["ecg_id"] = ecg_id
            yield rec
            records_yielded += 1
            
        except Exception as e:
            warnings.warn(f"Failed to process ECG {ecg_id} ({rel_path}): {e}")
            continue


def load_ptbxl_record(
    ptbxl_root: str | Path,
    ecg_id: int,
    use_low_res: bool = True,
    fs_target: float = 400.0,
    duration_sec: float = 10.0,
    baseline_method: Literal["highpass", "moving_average", "bandpass"] = "highpass",
    baseline_kwargs: Optional[Dict] = None,
    normalize: bool = True,
) -> Optional[Dict]:
    """
    Load a single PTB-XL record by ECG ID.
    
    Parameters
    ----------
    ptbxl_root : str | Path
        Path to PTB-XL root directory.
    ecg_id : int
        ECG ID to load.
    use_low_res : bool
        If True, use 100 Hz records.
    fs_target : float
        Target sampling frequency.
    duration_sec : float
        Target duration in seconds.
    baseline_method : str
        Baseline removal method.
    baseline_kwargs : dict, optional
        Additional kwargs for baseline removal.
    normalize : bool
        Whether to apply z-score normalization.
    
    Returns
    -------
    dict or None
        Preprocessed ECG record or None if not found/error.
    """
    ptbxl_root = Path(ptbxl_root)
    df = pd.read_csv(ptbxl_root / "ptbxl_database.csv")
    
    # Use filename_lr (100 Hz) or filename_hr (500 Hz)
    filename_col = "filename_lr" if use_low_res else "filename_hr"
    
    # Find the record
    matches = df[df["ecg_id"] == ecg_id]
    if len(matches) == 0:
        warnings.warn(f"ECG ID {ecg_id} not found in PTB-XL database")
        return None
    
    rel_path = matches.iloc[0][filename_col]
    
    try:
        rec = _load_single_record(
            base_dir=ptbxl_root,
            rel_path=rel_path,
            fs_target=fs_target,
            duration_sec=duration_sec,
            baseline_method=baseline_method,
            baseline_kwargs=baseline_kwargs,
            normalize=normalize,
        )
        rec["ecg_id"] = ecg_id
        return rec
    except Exception as e:
        warnings.warn(f"Failed to load ECG {ecg_id}: {e}")
        return None