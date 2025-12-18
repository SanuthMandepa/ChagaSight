"""
Test script for resampling utilities (includes 400→500/100 Hz cases).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.resample import resample_ecg, pad_or_trim

def create_test_signal(fs=400, duration=10):
    t = np.linspace(0, duration, int(fs * duration))
    return np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t), fs, t

def test_resampling():
    print("Testing resampling...")
    
    signal, fs_in, t_in = create_test_signal()
    
    # Resample to 500 Hz
    resampled_500, fs_500 = resample_ecg(signal.reshape(-1, 1), fs_in, 500.0)  # (T, 1)
    resampled_500 = resampled_500.flatten()
    
    # Resample to 100 Hz
    resampled_100, fs_100 = resample_ecg(signal.reshape(-1, 1), fs_in, 100.0)
    resampled_100 = resampled_100.flatten()
    
    # Plots
    plt.figure(figsize=(12, 4))
    plt.plot(t_in, signal, label="Original (400 Hz)")
    t_500 = np.arange(len(resampled_500)) / 500.0
    plt.plot(t_500, resampled_500, label="Resampled 500 Hz")
    plt.legend()
    plt.grid(True)
    output_dir = Path("tests/verification_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "resample_400_to_500.png")
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(t_in, signal, label="Original (400 Hz)")
    t_100 = np.arange(len(resampled_100)) / 100.0
    plt.plot(t_100, resampled_100, label="Resampled 100 Hz")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "resample_400_to_100.png")
    plt.close()

    print("\n✅ All resample tests completed! Plots saved.")

if __name__ == "__main__":
    test_resampling()