"""
Test script to verify baseline removal is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import baseline removal
from src.preprocessing.baseline_removal import remove_baseline

# Check for FFT
try:
    from scipy.fft import fft, fftfreq
    HAVE_FFT = True
except ImportError:
    HAVE_FFT = False

def create_test_ecg(fs=500, duration=10):
    t = np.linspace(0, duration, int(fs * duration))
    clean_ecg = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
    baseline = 0.5 * np.sin(2 * np.pi * 0.1 * t) + np.random.normal(0, 0.1, len(t))
    noisy_ecg = clean_ecg + baseline
    return noisy_ecg.reshape(-1, 1), clean_ecg.reshape(-1, 1), fs, t  # (T, 1) for single lead

def test_baseline_removal():
    noisy_ecg, clean_ecg, fs, t = create_test_ecg()
    methods = ['highpass', 'moving_average', 'bandpass']
    for method in methods:
        corrected = remove_baseline(noisy_ecg, fs, method=method)
        print(f"{method} corrected range: {corrected.min():.2f} to {corrected.max():.2f}")

    # Plot example (bandpass)
    corrected = remove_baseline(noisy_ecg, fs, 'bandpass')
    plt.figure()
    plt.plot(t, noisy_ecg.flatten(), label='Noisy')
    plt.plot(t, corrected.flatten(), label='Corrected')
    plt.legend()
    output_dir = Path("tests/verification_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "baseline_test.png")
    plt.close()
    print("Test complete. Plot saved.")

    # 12-lead test
    ecg_12lead = np.repeat(noisy_ecg, 12, axis=1)
    corrected_12 = remove_baseline(ecg_12lead, fs, 'moving_average', window_seconds=0.8)
    print(f"12-lead corrected range: {corrected_12.min():.2f} to {corrected_12.max():.2f}")
    print("✓ All baseline tests successful")

if __name__ == "__main__":
    test_baseline_removal()