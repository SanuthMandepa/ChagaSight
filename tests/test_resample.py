"""
Test script for resampling utilities.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the resample module
from src.preprocessing.resample import resample_ecg, pad_or_trim


def test_resampling():
    print("Testing resampling functions...")
    print("=" * 50)
    
    # Create a test signal
    fs_in = 100  # Original sampling rate
    duration = 5  # seconds
    t_in = np.linspace(0, duration, int(fs_in * duration), endpoint=False)
    
    # Create a simple sine wave + noise
    signal = np.sin(2 * np.pi * 5 * t_in) + 0.1 * np.random.randn(len(t_in))
    
    # Add a second "lead" (just for testing)
    signal_2d = np.column_stack([signal, signal * 0.5])
    
    # Test different target sampling rates
    test_cases = [
        (100, 100, "Same rate (no change)"),
        (100, 200, "100 Hz → 200 Hz (upsample)"),
        (100, 50, "100 Hz → 50 Hz (downsample)"),
        (100, 400, "100 Hz → 400 Hz (4x upsample)"),
    ]
    
    fig, axes = plt.subplots(len(test_cases), 2, figsize=(12, 10))
    
    for idx, (fs_in_test, fs_out_test, title) in enumerate(test_cases):
        print(f"\nTest: {title}")
        
        try:
            # Resample
            resampled, fs_new = resample_ecg(signal_2d, fs_in=fs_in_test, fs_out=fs_out_test)
            
            # Create time axes
            t_in_test = np.arange(signal_2d.shape[0]) / fs_in_test
            t_out = np.arange(resampled.shape[0]) / fs_new
            
            print(f"  Input shape: {signal_2d.shape}, fs: {fs_in_test} Hz")
            print(f"  Output shape: {resampled.shape}, fs: {fs_new} Hz")
            print(f"  Duration: {t_in_test[-1]:.2f}s → {t_out[-1]:.2f}s")
            
            # Plot original
            ax1 = axes[idx, 0]
            ax1.plot(t_in_test, signal_2d[:, 0], 'b-', alpha=0.7, label='Original', linewidth=1)
            ax1.set_title(f'Original: {fs_in_test} Hz\n{title}')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot resampled
            ax2 = axes[idx, 1]
            ax2.plot(t_out, resampled[:, 0], 'r-', alpha=0.7, label='Resampled', linewidth=1)
            ax2.set_title(f'Resampled: {fs_out_test} Hz\nLength: {resampled.shape[0]} samples')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Check signal preservation
            if fs_in_test == fs_out_test:
                # Should be identical (within tolerance)
                diff = np.abs(signal_2d[:, 0] - resampled[:len(signal_2d), 0]).max()
                print(f"  Max difference (should be ~0): {diff:.6f}")
            else:
                # Check if peaks are preserved
                # Find peaks in original and resampled
                from scipy.signal import find_peaks
                peaks_orig, _ = find_peaks(signal_2d[:, 0], height=0.5)
                peaks_resamp, _ = find_peaks(resampled[:, 0], height=0.5)
                
                # Convert peak positions to time
                peak_times_orig = peaks_orig / fs_in_test
                peak_times_resamp = peaks_resamp / fs_new
                
                print(f"  Peaks found: {len(peaks_orig)} → {len(peaks_resamp)}")
                if len(peaks_orig) > 0 and len(peaks_resamp) > 0:
                    # Check first few peaks align
                    for i in range(min(3, len(peaks_orig), len(peaks_resamp))):
                        print(f"    Peak {i+1}: {peak_times_orig[i]:.3f}s → {peak_times_resamp[i]:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            axes[idx, 0].text(0.5, 0.5, f'Error:\n{str(e)}', 
                            ha='center', va='center', transform=axes[idx, 0].transAxes,
                            color='red', fontsize=10)
            axes[idx, 1].text(0.5, 0.5, f'Error:\n{str(e)}', 
                            ha='center', va='center', transform=axes[idx, 1].transAxes,
                            color='red', fontsize=10)
    
    plt.suptitle('Resampling Tests', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save results
    output_dir = Path(__file__).parent.parent / "notebooks" / "verification_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "resample_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*50}")
    print(f"Test complete! Results saved to:")
    print(f"📊 {output_path}")
    print(f"{'='*50}")
    
    # Test pad_or_trim
    print("\nTesting pad_or_trim function...")
    test_pad_trim()


def test_pad_trim():
    """Test the pad_or_trim function."""
    print("\n" + "="*30)
    print("Testing pad_or_trim")
    print("="*30)
    
    # Create test signal
    original_length = 3000
    test_signal = np.random.randn(original_length, 12)
    
    test_cases = [
        (4000, "Pad to 4000 (longer)"),
        (2000, "Trim to 2000 (shorter)"),
        (3000, "Same length (no change)"),
    ]
    
    fig, axes = plt.subplots(len(test_cases), 2, figsize=(12, 8))
    
    for idx, (target_len, description) in enumerate(test_cases):
        print(f"\nTest: {description}")
        print(f"  Original length: {original_length}")
        print(f"  Target length: {target_len}")
        
        try:
            # Apply pad_or_trim
            processed = pad_or_trim(test_signal, target_length=target_len)
            
            print(f"  Processed length: {processed.shape[0]}")
            
            # Plot original (lead 0)
            ax1 = axes[idx, 0]
            ax1.plot(test_signal[:, 0], 'b-', alpha=0.7, linewidth=0.5)
            ax1.axvline(x=original_length-1, color='r', linestyle='--', alpha=0.5)
            ax1.set_title(f'Original: {original_length} samples\n{description}')
            ax1.set_xlabel('Samples')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            
            # Plot processed
            ax2 = axes[idx, 1]
            ax2.plot(processed[:, 0], 'g-', alpha=0.7, linewidth=0.5)
            ax2.axvline(x=target_len-1, color='r', linestyle='--', alpha=0.5)
            ax2.set_title(f'Processed: {target_len} samples')
            ax2.set_xlabel('Samples')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)
            
            # Statistics
            if target_len > original_length:
                # Should be padded with zeros
                padding_start = (target_len - original_length) // 2
                padding_end = target_len - original_length - padding_start
                padded_part = np.concatenate([
                    processed[:padding_start, 0],
                    processed[target_len-padding_end:, 0]
                ])
                max_padding = np.abs(padded_part).max()
                print(f"  Padding max value (should be 0): {max_padding:.6f}")
            elif target_len < original_length:
                # Should be trimmed (center crop)
                crop_start = (original_length - target_len) // 2
                original_center = test_signal[crop_start:crop_start+target_len, 0]
                diff = np.abs(original_center - processed[:, 0]).max()
                print(f"  Max difference from center crop (should be 0): {diff:.6f}")
            else:
                # Should be identical
                diff = np.abs(test_signal[:, 0] - processed[:, 0]).max()
                print(f"  Max difference (should be 0): {diff:.6f}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            axes[idx, 0].text(0.5, 0.5, f'Error:\n{str(e)}', 
                            ha='center', va='center', transform=axes[idx, 0].transAxes,
                            color='red', fontsize=10)
            axes[idx, 1].text(0.5, 0.5, f'Error:\n{str(e)}', 
                            ha='center', va='center', transform=axes[idx, 1].transAxes,
                            color='red', fontsize=10)
    
    plt.suptitle('Pad/Trim Tests', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "notebooks" / "verification_outputs"
    output_path = output_dir / "pad_trim_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPad/Trim test complete! Saved to: {output_path}")


def test_realistic_ecg_resampling():
    """Test with a more realistic ECG-like signal."""
    print("\n" + "="*50)
    print("Testing with realistic ECG-like signal")
    print("="*50)
    
    # Create ECG-like signal
    fs_in = 100
    duration = 10
    t = np.linspace(0, duration, int(fs_in * duration))
    
    # Simulate ECG with QRS complexes
    ecg_signal = np.zeros_like(t)
    heart_rate = 72  # bpm
    rr_interval = 60 / heart_rate  # seconds
    
    for i in range(int(duration / rr_interval)):
        peak_time = i * rr_interval
        peak_idx = int(peak_time * fs_in)
        
        if peak_idx + 150 < len(t):
            # Add QRS complex
            ecg_signal[peak_idx:peak_idx+20] -= 0.2    # Q wave
            ecg_signal[peak_idx+20:peak_idx+40] += 1.0  # R wave
            ecg_signal[peak_idx+40:peak_idx+60] -= 0.3  # S wave
            # T wave
            ecg_signal[peak_idx+80:peak_idx+120] += 0.3
    
    # Add some baseline wander
    ecg_signal += 0.2 * np.sin(2 * np.pi * 0.1 * t)
    
    # Convert to 2D (12 leads)
    ecg_2d = np.column_stack([ecg_signal] * 12)
    
    # Test resampling to 400 Hz (common in your pipeline)
    print(f"\nResampling realistic ECG: {fs_in} Hz → 400 Hz")
    
    try:
        resampled, fs_out = resample_ecg(ecg_2d, fs_in=fs_in, fs_out=400)
        
        print(f"  Original: {ecg_2d.shape[0]} samples at {fs_in} Hz")
        print(f"  Resampled: {resampled.shape[0]} samples at {fs_out} Hz")
        
        # Check QRS complex preservation
        # Find R peaks in original
        from scipy.signal import find_peaks
        peaks_orig, props_orig = find_peaks(ecg_2d[:, 0], height=0.5, distance=fs_in*0.5)
        peaks_resamp, props_resamp = find_peaks(resampled[:, 0], height=0.5, distance=fs_out*0.5)
        
        print(f"  R-peaks found: {len(peaks_orig)} → {len(peaks_resamp)}")
        
        if len(peaks_orig) > 0 and len(peaks_resamp) > 0:
            # Plot comparison
            plt.figure(figsize=(12, 6))
            
            # Original
            plt.subplot(2, 1, 1)
            plt.plot(np.arange(len(ecg_2d))/fs_in, ecg_2d[:, 0], 'b-', alpha=0.7, label=f'Original ({fs_in} Hz)')
            plt.plot(peaks_orig/fs_in, props_orig['peak_heights'], 'ro', label='R-peaks')
            plt.title(f'Original ECG ({fs_in} Hz) - {len(peaks_orig)} R-peaks')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Resampled
            plt.subplot(2, 1, 2)
            plt.plot(np.arange(len(resampled))/fs_out, resampled[:, 0], 'g-', alpha=0.7, label=f'Resampled ({fs_out} Hz)')
            plt.plot(peaks_resamp/fs_out, props_resamp['peak_heights'], 'ro', label='R-peaks')
            plt.title(f'Resampled ECG ({fs_out} Hz) - {len(peaks_resamp)} R-peaks')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_dir = Path(__file__).parent.parent / "notebooks" / "verification_outputs"
            output_path = output_dir / "ecg_resample_test.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            print(f"  Plot saved to: {output_path}")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    test_resampling()
    test_realistic_ecg_resampling()
    print("\n✅ All resample tests completed!")