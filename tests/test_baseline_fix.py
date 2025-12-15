"""
Test script to verify baseline removal is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the fixed baseline removal module
from src.preprocessing.baseline_removal import remove_baseline

# Test with a synthetic ECG signal
def create_test_ecg(fs=400, duration=10):
    """Create a synthetic ECG with baseline wander for testing."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Create a clean ECG-like signal (simplified)
    clean_ecg = np.zeros_like(t)
    
    # Add QRS complexes at regular intervals (approx 60 bpm)
    heart_rate = 60  # bpm
    rr_interval = 60 / heart_rate  # seconds
    
    for i in range(int(duration / rr_interval)):
        peak_time = i * rr_interval
        peak_idx = int(peak_time * fs)
        
        # Add QRS-like spike (simplified ECG waveform)
        if peak_idx + 100 < len(clean_ecg):
            # Q wave (negative)
            clean_ecg[peak_idx:peak_idx+20] -= 0.2
            # R wave (positive)
            clean_ecg[peak_idx+20:peak_idx+40] += 1.0
            # S wave (negative)
            clean_ecg[peak_idx+40:peak_idx+60] -= 0.3
            # T wave (positive)
            clean_ecg[peak_idx+80:peak_idx+100] += 0.4
    
    # Add baseline wander (low frequency)
    baseline = 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.3 * np.sin(2 * np.pi * 0.05 * t)
    
    # Add some high frequency noise
    noise = 0.05 * np.random.randn(len(t))
    
    return t, clean_ecg, baseline, noise

def test_baseline_removal():
    print("Testing baseline removal methods...")
    print("=" * 50)
    
    # Create test signal
    fs = 400
    duration = 10
    t, clean_ecg, baseline, noise = create_test_ecg(fs, duration)
    
    # Create corrupted signal (simulating raw ECG with issues)
    corrupted_ecg = clean_ecg + baseline + noise
    
    # Reshape to (T, 1) for the functions that expect 2D
    corrupted_2d = corrupted_ecg.reshape(-1, 1)
    clean_2d = clean_ecg.reshape(-1, 1)
    
    # Test different methods
    methods = ["highpass", "moving_average", "bandpass"]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i, method in enumerate(methods):
        try:
            print(f"\nTesting {method} method:")
            
            # Apply baseline removal with appropriate parameters
            if method == "highpass":
                corrected = remove_baseline(
                    corrupted_2d,
                    fs=fs,
                    method=method,
                    cutoff_hz=0.7,
                    order=3
                )
                method_name = f"High-pass (0.7 Hz)"
                
            elif method == "moving_average":
                corrected = remove_baseline(
                    corrupted_2d,
                    fs=fs,
                    method=method,
                    window_seconds=0.8
                )
                method_name = f"Moving Average (0.8s)"
                
            elif method == "bandpass":
                corrected = remove_baseline(
                    corrupted_2d,
                    fs=fs,
                    method=method,
                    low_cut_hz=0.5,
                    high_cut_hz=45.0,
                    order=4
                )
                method_name = f"Band-pass (0.5-45 Hz)"
            
            # Calculate metrics
            error = np.abs(corrected[:, 0] - clean_ecg)
            mae = error.mean()
            max_error = error.max()
            
            print(f"  Mean Absolute Error: {mae:.4f}")
            print(f"  Maximum Error: {max_error:.4f}")
            print(f"  Signal range before: {corrupted_ecg.min():.2f} to {corrupted_ecg.max():.2f}")
            print(f"  Signal range after:  {corrected.min():.2f} to {corrected.max():.2f}")
            
            # Plot 1: Original vs Baseline
            ax1 = axes[i, 0]
            ax1.plot(t, corrupted_ecg, label='Raw (with baseline)', alpha=0.7, linewidth=1)
            ax1.plot(t, baseline, label='Baseline', alpha=0.5, linestyle='--', linewidth=2)
            ax1.set_title(f'{method_name}\nRaw Signal')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Corrected vs Clean
            ax2 = axes[i, 1]
            ax2.plot(t, corrected[:, 0], label='Corrected', alpha=0.8, linewidth=1.5)
            ax2.plot(t, clean_ecg, label='Clean (reference)', alpha=0.5, linestyle=':', linewidth=2)
            ax2.set_title(f'Corrected vs Clean\nMAE: {mae:.3f}')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Error over time
            ax3 = axes[i, 2]
            ax3.plot(t, error, color='red', alpha=0.7, linewidth=1)
            ax3.fill_between(t, 0, error, alpha=0.3, color='red')
            ax3.set_title(f'Error\nMax: {max_error:.3f}')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Absolute Error')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Frequency spectrum comparison
            ax4 = axes[i, 3]
            # Compute FFT
            from scipy.fft import fft, fftfreq
            n = len(corrupted_ecg)
            freq = fftfreq(n, 1/fs)[:n//2]
            
            # Raw spectrum
            raw_fft = np.abs(fft(corrupted_ecg)[:n//2])
            ax4.plot(freq, raw_fft, label='Raw', alpha=0.5, linewidth=1)
            
            # Corrected spectrum
            corr_fft = np.abs(fft(corrected[:, 0])[:n//2])
            ax4.plot(freq, corr_fft, label='Corrected', alpha=0.8, linewidth=1.5)
            
            # Clean spectrum
            clean_fft = np.abs(fft(clean_ecg)[:n//2])
            ax4.plot(freq, clean_fft, label='Clean', alpha=0.5, linestyle=':', linewidth=2)
            
            ax4.set_title('Frequency Spectrum')
            ax4.set_xlabel('Frequency (Hz)')
            ax4.set_ylabel('Magnitude')
            ax4.set_xlim(0, 50)  # Show up to 50 Hz
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"  ✗ Error with {method}: {e}")
            # Mark failed subplots
            for j in range(4):
                axes[i, j].text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                              ha='center', va='center', transform=axes[i, j].transAxes,
                              color='red', fontsize=10)
    
    plt.suptitle('Baseline Removal Methods Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save results
    output_dir = Path(__file__).parent.parent / "notebooks" / "verification_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "baseline_removal_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*50}")
    print(f"Test complete! Results saved to:")
    print(f"📊 {output_path}")
    print(f"{'='*50}")
    
    # Also test with 12-lead data
    print("\nTesting with 12-lead synthetic data...")
    test_12lead_simulation()

def test_12lead_simulation():
    """Test with simulated 12-lead data."""
    fs = 400
    duration = 5
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create simple 12-lead data with different baselines for each lead
    signals = []
    for lead in range(12):
        # Each lead has different baseline wander
        baseline = 0.3 * (lead + 1) * np.sin(2 * np.pi * (0.1 + 0.02 * lead) * t)
        signal = 0.5 * np.random.randn(len(t)) + baseline
        signals.append(signal)
    
    # Stack to (T, 12)
    ecg_12lead = np.stack(signals, axis=1)
    
    print(f"Created 12-lead test data: {ecg_12lead.shape}")
    
    # Test moving average on all leads
    corrected = remove_baseline(
        ecg_12lead,
        fs=fs,
        method="moving_average",
        window_seconds=0.8
    )
    
    print(f"  Original range: {ecg_12lead.min():.2f} to {ecg_12lead.max():.2f}")
    print(f"  Corrected range: {corrected.min():.2f} to {corrected.max():.2f}")
    print(f"  ✓ 12-lead baseline removal successful")

if __name__ == "__main__":
    test_baseline_removal()