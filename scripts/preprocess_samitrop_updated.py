# scripts/preprocess_samitrop_updated.py
"""Updated SaMi-Trop preprocessing with zero-padding removal."""

from pathlib import Path
import time
import warnings
import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample as scipy_resample
from tqdm import tqdm

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
SAMI_ROOT = Path("data/raw/sami_trop")
OUTPUT_DIR = Path("data/processed/1d_signals/sami_trop")
METADATA_OUTPUT = Path("data/processed/metadata/sami_trop_processed.csv")

ORIGINAL_FS = 400.0
TARGET_FS = 400.0
TARGET_DURATION_SEC = 10.0
TARGET_SAMPLES = int(TARGET_DURATION_SEC * TARGET_FS)

# Baseline removal - use GENTLE window for SaMi-Trop
BASELINE_METHOD = "moving_average"
BASELINE_KWARGS = {"window_seconds": 0.2}  # 200ms, not 800ms!

# Normalization configuration
NORMALIZATION_EPS = 1e-8

def remove_zero_padding(signal, threshold=0.01):
    """
    Remove zero-padding from SaMi-Trop signals.
    
    According to documentation:
    - 7s signals: 2800 samples → padded to 4096 with 648 zeros on each side
    - 10s signals: 4000 samples → padded to 4096 with 48 zeros on each side
    
    Parameters
    ----------
    signal : np.ndarray
        Raw signal of shape (4096, 12)
    threshold : float
        Threshold to detect non-zero values
        
    Returns
    -------
    np.ndarray
        Signal with zero-padding removed
    """
    # Find non-zero regions (signal above threshold in any lead)
    signal_abs = np.abs(signal)
    max_per_sample = np.max(signal_abs, axis=1)
    non_zero_mask = max_per_sample > threshold
    
    if not np.any(non_zero_mask):
        # If no signal found, return center 4000 samples
        return signal[48:4048, :]
    
    # Find first and last non-zero samples
    start_idx = np.argmax(non_zero_mask)
    end_idx = len(non_zero_mask) - np.argmax(non_zero_mask[::-1])
    
    # Determine original length
    original_length = end_idx - start_idx
    
    # Check if it's a 7s or 10s signal
    if abs(original_length - 2800) < 100:  # ~7s signal
        print(f"  Detected 7s signal (length: {original_length})")
        # Extract the actual signal (remove zeros)
        cleaned = signal[start_idx:end_idx, :]
        # Resample 2800 → 4000 samples
        cleaned = scipy_resample(cleaned, 4000, axis=0)
        return cleaned
    elif abs(original_length - 4000) < 100:  # ~10s signal
        print(f"  Detected 10s signal (length: {original_length})")
        # Extract the actual signal (remove zeros)
        return signal[start_idx:end_idx, :]
    else:
        # Fallback: use center 4000 samples
        print(f"  Warning: Unknown signal length {original_length}, using center crop")
        return signal[48:4048, :]

def preprocess_single_exam(
    signal: np.ndarray,
    fs_in: float = 400.0,
    fs_target: float = TARGET_FS,
    duration_sec: float = TARGET_DURATION_SEC,
    baseline_method: str = BASELINE_METHOD,
    baseline_kwargs: dict = None,
    eps: float = NORMALIZATION_EPS,
) -> np.ndarray:
    """
    Updated preprocessing for SaMi-Trop with zero-padding removal.
    """
    if baseline_kwargs is None:
        baseline_kwargs = BASELINE_KWARGS
        
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(
            f"Expected signal of shape (T, 12), got {signal.shape}"
        )

    # Ensure float32
    signal = signal.astype(np.float32)
    
    print(f"  Raw signal shape: {signal.shape}")
    print(f"  Raw range: {signal.min():.4f} to {signal.max():.4f}")

    # 1) Remove zero-padding (4096 → 4000 or 2800→4000)
    print("  Step 1: Removing zero-padding")
    signal = remove_zero_padding(signal)
    print(f"  After zero-pad removal: {signal.shape}")
    
    # 2) Check signal range - decide on baseline removal
    signal_range = np.max(signal) - np.min(signal)
    print(f"  Signal range: {signal_range:.4f}")
    
    # 3) Apply GENTLE baseline removal only if needed
    if signal_range > 3.0:  # Only if signal has substantial range
        print(f"  Step 2: Applying gentle baseline removal (0.2s window)")
        signal = remove_baseline(
            signal,
            fs=fs_in,
            method=baseline_method,
            **baseline_kwargs,
        )
    else:
        print(f"  Step 2: Skipping baseline removal (signal already clean)")
    
    # 4) Resample (if fs_in != fs_target, otherwise no-op)
    print(f"  Step 3: Resampling to {fs_target} Hz")
    signal, fs_rs = resample_ecg(signal, fs_in=fs_in, fs_out=fs_target)

    # 5) Pad/trim to fixed length (should be 4000 already)
    target_len = int(round(duration_sec * fs_rs))
    print(f"  Step 4: Padding/trimming to {target_len} samples")
    signal = pad_or_trim(signal, target_length=target_len)

    # 6) Z-score normalization
    print(f"  Step 5: Z-score normalization")
    signal = zscore_per_lead(signal, eps=eps)
    
    print(f"  Final shape: {signal.shape}, range: {signal.min():.4f} to {signal.max():.4f}")
    
    return signal.astype(np.float32)

def main():
    print("="*60)
    print("🔄 UPDATED SaMi-Trop Preprocessing with Zero-Pad Removal")
    print("="*60)
    
    # 1) Load metadata
    csv_path = SAMI_ROOT / "exams.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"SaMi-Trop CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📄 Loaded metadata: {len(df)} exams")
    
    # 2) Open HDF5
    h5_path = SAMI_ROOT / "exams.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found at {h5_path}")
    
    print(f"📂 Opening HDF5: {h5_path}")
    h5_file = h5py.File(h5_path, "r")
    tracings = h5_file["tracings"]  # (1631, 4096, 12)
    
    # 3) Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    # 4) Process each exam
    processed_records = []
    failed_records = []
    start_time = time.time()
    
    # Test with first 10 exams
    test_exams = min(10, len(df))
    print(f"\n🧪 Testing with first {test_exams} exams...")
    
    for idx in tqdm(range(test_exams), desc="Processing"):
        row = df.iloc[idx]
        exam_id = int(row["exam_id"])
        
        try:
            # Get raw signal
            raw_signal = tracings[idx, :, :]  # (4096, 12)
            
            # Preprocess
            processed = preprocess_single_exam(
                signal=raw_signal,
                fs_in=ORIGINAL_FS,
                fs_target=TARGET_FS,
                duration_sec=TARGET_DURATION_SEC,
                baseline_method=BASELINE_METHOD,
                baseline_kwargs=BASELINE_KWARGS,
                eps=NORMALIZATION_EPS,
            )
            
            # Save
            out_path = OUTPUT_DIR / f"{exam_id}.npy"
            np.save(out_path, processed)
            
            # Track metadata
            processed_records.append({
                "exam_id": exam_id,
                "processed_path": str(out_path),
                "signal_shape": processed.shape,
                "signal_dtype": str(processed.dtype),
                "fs": TARGET_FS,
                "duration_sec": TARGET_DURATION_SEC,
            })
            
            # Print first exam details
            if idx == 0:
                print(f"\n📊 First exam ({exam_id}) statistics:")
                print(f"   Raw: shape {raw_signal.shape}, range {raw_signal.min():.4f} to {raw_signal.max():.4f}")
                print(f"   Processed: shape {processed.shape}, range {processed.min():.4f} to {processed.max():.4f}")
                print(f"   Mean per lead: {processed.mean(axis=0)}")
                print(f"   Std per lead: {processed.std(axis=0)}")
                
        except Exception as e:
            print(f"❌ Failed exam {exam_id}: {e}")
            failed_records.append({
                "exam_id": exam_id,
                "error": str(e)
            })
            continue
    
    # 5) Close HDF5
    h5_file.close()
    
    # 6) Save metadata
    if processed_records:
        metadata_df = pd.DataFrame(processed_records)
        metadata_df.to_csv(METADATA_OUTPUT, index=False)
        print(f"\n💾 Saved metadata to {METADATA_OUTPUT}")
    
    # 7) Report
    elapsed = time.time() - start_time
    print(f"\n✅ Processed: {len(processed_records)} exams")
    print(f"❌ Failed: {len(failed_records)} exams")
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"📁 Output: {OUTPUT_DIR}")
    
    # 8) Create validation plot
    if processed_records:
        create_validation_plot()

def create_validation_plot():
    """Create validation plot for first processed exam."""
    try:
        import matplotlib.pyplot as plt
        
        # Load first processed exam
        df = pd.read_csv(SAMI_ROOT / "exams.csv")
        first_id = df.iloc[0]["exam_id"]
        processed_path = OUTPUT_DIR / f"{first_id}.npy"
        
        if processed_path.exists():
            signal = np.load(processed_path)
            
            fig, axes = plt.subplots(3, 4, figsize=(15, 8))
            axes = axes.flatten()
            
            time_axis = np.arange(signal.shape[0]) / 400.0
            
            for lead in range(12):
                axes[lead].plot(time_axis, signal[:, lead])
                axes[lead].set_title(f"Lead {lead+1}")
                axes[lead].set_xlabel("Time (s)")
                axes[lead].set_ylabel("Amplitude (z-score)")
                axes[lead].grid(True, alpha=0.3)
                axes[lead].set_ylim(-4, 4)
            
            plt.suptitle(f"SaMi-Trop Exam {first_id} - Processed 12-Lead ECG")
            plt.tight_layout()
            
            output_path = Path("notebooks/verification_outputs/samitrop_processed_check.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            print(f"\n📈 Validation plot saved: {output_path}")
            
    except Exception as e:
        print(f"  Could not create validation plot: {e}")

if __name__ == "__main__":
    main()