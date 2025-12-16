"""
Quick check of raw ECG data to see if it looks correct.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_ptbxl_sample():
    """Check a sample from PTB-XL."""
    try:
        import wfdb
        
        print("Checking PTB-XL sample...")
        ptb_dir = Path("data/raw/ptbxl")
        record_path = ptb_dir / "records100/00000/00001_lr"
        
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal.astype(np.float32)
        fs = float(record.fs)
        
        print(f"  Shape: {signal.shape}")
        print(f"  Sampling rate: {fs} Hz")
        print(f"  Lead I - Min: {signal[:, 0].min():.4f}, Max: {signal[:, 0].max():.4f}, Mean: {signal[:, 0].mean():.4f}")
        
        # Plot
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(len(signal))/fs, signal[:, 0])
        plt.title(f"PTB-XL Raw ECG (Lead I) - Record 00001")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        output_dir = Path(__file__).parent.parent / "notebooks" / "verification_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "ptbxl_raw_check.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {output_path}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    check_ptbxl_sample()
    print("\n✅ Raw data check completed!")