"""
Test the full preprocessing pipeline on a single PTB-XL record.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead

def test_ptbxl_pipeline():
    # We'll simulate a PTB-XL-like signal for testing
    fs_in = 100  # PTB-XL low res is 100 Hz
    duration = 10
    t = np.linspace(0, duration, int(fs_in * duration), endpoint=False)
    
    # Create a 12-lead synthetic ECG with baseline wander and noise
    signals = []
    for lead in range(12):
        # Each lead has a different ECG-like pattern and baseline
        heart_rate = 60 + lead * 5  # vary heart rate per lead
        rr_interval = 60 / heart_rate
        
        clean = np.zeros_like(t)
        for i in range(int(duration / rr_interval)):
            peak_idx = int(i * rr_interval * fs_in)
            if peak_idx + 100 < len(t):
                # QRS complex
                clean[peak_idx:peak_idx+20] -= 0.1 * (lead+1)
                clean[peak_idx+20:peak_idx+40] += 0.5 * (lead+1)
                clean[peak_idx+40:peak_idx+60] -= 0.2 * (lead+1)
                # T wave
                clean[peak_idx+80:peak_idx+100] += 0.3 * (lead+1)
        
        baseline = 0.2 * (lead+1) * np.sin(2 * np.pi * 0.1 * t)
        noise = 0.02 * np.random.randn(len(t))
        
        signals.append(clean + baseline + noise)
    
    # Stack to (T, 12)
    ecg = np.stack(signals, axis=1)
    
    print(f"Original signal shape: {ecg.shape}")
    print(f"Original range: {ecg.min():.2f} to {ecg.max():.2f}")
    
    # Step 1: Baseline removal (bandpass)
    ecg_filtered = remove_baseline(
        ecg, 
        fs=fs_in, 
        method="bandpass", 
        low_cut_hz=0.5, 
        high_cut_hz=45.0, 
        order=4
    )
    print(f"After bandpass: {ecg_filtered.min():.2f} to {ecg_filtered.max():.2f}")
    
    # Step 2: Resample to 400 Hz
    ecg_resampled, fs_out = resample_ecg(ecg_filtered, fs_in=fs_in, fs_out=400.0)
    print(f"After resampling to {fs_out} Hz: shape {ecg_resampled.shape}")
    
    # Step 3: Pad/trim to 10 seconds at 400 Hz (4000 samples)
    target_len = int(10 * fs_out)
    ecg_fixed = pad_or_trim(ecg_resampled, target_len)
    print(f"After fixing to {target_len} samples: shape {ecg_fixed.shape}")
    
    # Step 4: Z-score normalization
    ecg_normalized = zscore_per_lead(ecg_fixed)
    print(f"After z-score: mean={ecg_normalized.mean(axis=0)[0]:.2f}, std={ecg_normalized.std(axis=0)[0]:.2f}")
    
    # Plot the first lead at each stage
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    
    # Original
    axes[0].plot(t, ecg[:, 0])
    axes[0].set_title(f'Original (Lead 1) - {fs_in} Hz')
    axes[0].set_ylabel('Amplitude')
    
    # After bandpass
    t_filtered = np.arange(ecg_filtered.shape[0]) / fs_in
    axes[1].plot(t_filtered, ecg_filtered[:, 0])
    axes[1].set_title('After Bandpass (0.5-45 Hz)')
    axes[1].set_ylabel('Amplitude')
    
    # After resampling
    t_resampled = np.arange(ecg_resampled.shape[0]) / fs_out
    axes[2].plot(t_resampled, ecg_resampled[:, 0])
    axes[2].set_title(f'After Resampling to {fs_out} Hz')
    axes[2].set_ylabel('Amplitude')
    
    # After fixed length
    t_fixed = np.arange(ecg_fixed.shape[0]) / fs_out
    axes[3].plot(t_fixed, ecg_fixed[:, 0])
    axes[3].set_title(f'After Fixing to {target_len} samples')
    axes[3].set_ylabel('Amplitude')
    
    # After normalization
    axes[4].plot(t_fixed, ecg_normalized[:, 0])
    axes[4].set_title('After Z-score Normalization')
    axes[4].set_ylabel('Z-score')
    axes[4].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path(__file__).parent.parent / "notebooks" / "verification_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ptbxl_pipeline_test.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nPipeline test complete. Plot saved to {output_path}")

if __name__ == "__main__":
    test_ptbxl_pipeline()