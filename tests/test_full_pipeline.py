"""
Test the full preprocessing pipeline on real data.
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
    """Test the exact PTB-XL preprocessing pipeline."""
    print("Testing PTB-XL preprocessing pipeline...")
    print("=" * 60)
    
    try:
        import wfdb
        
        # Load a PTB-XL record
        ptb_dir = Path("data/raw/ptbxl")
        record_path = ptb_dir / "records100/00000/00001_lr"
        record = wfdb.rdrecord(str(record_path))
        raw_signal = record.p_signal.astype(np.float32)
        fs_in = float(record.fs)
        
        print(f"Raw signal: {raw_signal.shape} at {fs_in} Hz")
        print(f"Lead I - Min: {raw_signal[:, 0].min():.4f}, Max: {raw_signal[:, 0].max():.4f}")
        
        # Step-by-step pipeline (matching preprocess_ptbxl.py)
        steps = []
        step_names = []
        
        # 1. Raw signal
        steps.append(raw_signal.copy())
        step_names.append("Raw")
        
        # 2. Baseline removal (bandpass as in PTB-XL script)
        print("\nStep 1: Baseline removal (bandpass 0.5-45 Hz)")
        baseline_removed = remove_baseline(
            raw_signal,
            fs=fs_in,
            method="bandpass",
            low_cut_hz=0.5,
            high_cut_hz=45.0,
            order=4
        )
        steps.append(baseline_removed.copy())
        step_names.append("Baseline\nRemoved")
        print(f"  Lead I range: {baseline_removed[:, 0].min():.4f} to {baseline_removed[:, 0].max():.4f}")
        
        # 3. Resample to 400 Hz
        print("\nStep 2: Resample to 400 Hz")
        resampled, fs_out = resample_ecg(
            baseline_removed,
            fs_in=fs_in,
            fs_out=400.0
        )
        steps.append(resampled.copy())
        step_names.append(f"Resampled\n{fs_out} Hz")
        print(f"  New shape: {resampled.shape}")
        print(f"  Lead I range: {resampled[:, 0].min():.4f} to {resampled[:, 0].max():.4f}")
        
        # 4. Pad/trim to 4000 samples (10s at 400 Hz)
        print("\nStep 3: Pad/trim to 4000 samples")
        target_len = 4000
        padded = pad_or_trim(resampled, target_length=target_len)
        steps.append(padded.copy())
        step_names.append(f"Fixed length\n{target_len} samples")
        print(f"  Final shape: {padded.shape}")
        print(f"  Lead I range: {padded[:, 0].min():.4f} to {padded[:, 0].max():.4f}")
        
        # 5. Z-score normalization
        print("\nStep 4: Z-score normalization")
        normalized = zscore_per_lead(padded)
        steps.append(normalized.copy())
        step_names.append("Z-score\nNormalized")
        print(f"  Lead I stats - Mean: {normalized[:, 0].mean():.4f}, Std: {normalized[:, 0].std():.4f}")
        print(f"  Lead I range: {normalized[:, 0].min():.4f} to {normalized[:, 0].max():.4f}")
        
        # Check z-score properties
        expected_mean = 0.0
        expected_std = 1.0
        actual_mean = normalized.mean(axis=0)
        actual_std = normalized.std(axis=0)
        
        print(f"\nZ-score verification (all leads):")
        print(f"  Mean (should be ~0): {actual_mean}")
        print(f"  Std (should be ~1): {actual_std}")
        
        # Visualize all steps
        fig, axes = plt.subplots(len(steps), 3, figsize=(15, 12))
        
        for i, (signal, name) in enumerate(zip(steps, step_names)):
            # Time axis for each step
            if i == 0:  # Raw - 100 Hz
                t = np.arange(len(signal)) / fs_in
            elif i == 1:  # After baseline - still 100 Hz
                t = np.arange(len(signal)) / fs_in
            else:  # After resampling - 400 Hz
                t = np.arange(len(signal)) / 400.0
            
            # Plot lead I
            ax1 = axes[i, 0]
            ax1.plot(t, signal[:, 0], 'b-', linewidth=1)
            ax1.set_title(f'Step {i+1}: {name}\nLead I')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            
            # Plot all leads (small view)
            ax2 = axes[i, 1]
            for lead in range(min(12, signal.shape[1])):
                ax2.plot(t, signal[:, lead], alpha=0.5, linewidth=0.5)
            ax2.set_title('All 12 Leads')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)
            
            # Histogram of lead I values
            ax3 = axes[i, 2]
            ax3.hist(signal[:, 0], bins=50, alpha=0.7, edgecolor='black')
            ax3.set_title(f'Distribution of Lead I')
            ax3.set_xlabel('Amplitude')
            ax3.set_ylabel('Count')
            ax3.grid(True, alpha=0.3)
            
            # Add stats to histogram
            stats_text = f"Min: {signal[:, 0].min():.3f}\nMax: {signal[:, 0].max():.3f}\nMean: {signal[:, 0].mean():.3f}\nStd: {signal[:, 0].std():.3f}"
            ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('PTB-XL Preprocessing Pipeline - Step by Step', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save results
        output_dir = Path(__file__).parent.parent / "notebooks" / "verification_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "full_pipeline_test.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n{'='*60}")
        print(f"Test complete! Results saved to:")
        print(f"📊 {output_path}")
        print(f"{'='*60}")
        
        # Now run the actual preprocess_ptbxl.py on this record and compare
        print("\n\nComparing with actual preprocess_ptbxl.py output...")
        compare_with_processed()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def compare_with_processed():
    """Compare our step-by-step with the saved processed file."""
    try:
        # Load the processed file (if it exists)
        processed_path = Path("data/processed/1d_signals/ptbxl/1.npy")
        
        if processed_path.exists():
            processed_signal = np.load(processed_path)
            print(f"Loaded processed signal: {processed_signal.shape}")
            
            # Compare with what we just computed
            # We need to compute the final normalized signal again
            raw_signal, fs_in = load_ptbxl_record()
            
            # Apply the pipeline
            baseline_removed = remove_baseline(
                raw_signal,
                fs=fs_in,
                method="bandpass",
                low_cut_hz=0.5,
                high_cut_hz=45.0,
                order=4
            )
            
            resampled, _ = resample_ecg(baseline_removed, fs_in=fs_in, fs_out=400.0)
            padded = pad_or_trim(resampled, target_length=4000)
            our_normalized = zscore_per_lead(padded)
            
            # Compare
            if our_normalized.shape == processed_signal.shape:
                diff = np.abs(our_normalized - processed_signal).max()
                mean_diff = np.abs(our_normalized - processed_signal).mean()
                
                print(f"\nComparison with saved processed file:")
                print(f"  Max absolute difference: {diff:.6f}")
                print(f"  Mean absolute difference: {mean_diff:.6f}")
                
                if diff < 0.01:
                    print("  ✅ Excellent match! Our pipeline matches saved file.")
                elif diff < 0.1:
                    print("  ⚠ Reasonable match (minor differences).")
                else:
                    print("  ⚠ Significant differences found!")
                    
                    # Plot comparison
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    
                    # Lead I comparison
                    axes[0, 0].plot(our_normalized[:, 0], 'b-', alpha=0.7, label='Our pipeline')
                    axes[0, 0].plot(processed_signal[:, 0], 'r--', alpha=0.7, label='Saved file')
                    axes[0, 0].set_title(f'Lead I Comparison\nMax diff: {diff:.4f}')
                    axes[0, 0].set_xlabel('Samples')
                    axes[0, 0].set_ylabel('Amplitude')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # Difference
                    axes[0, 1].plot(np.abs(our_normalized[:, 0] - processed_signal[:, 0]), 'g-')
                    axes[0, 1].set_title('Absolute Difference (Lead I)')
                    axes[0, 1].set_xlabel('Samples')
                    axes[0, 1].set_ylabel('Difference')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Scatter plot
                    axes[1, 0].scatter(our_normalized[:, 0], processed_signal[:, 0], alpha=0.5, s=1)
                    axes[1, 0].plot([-3, 3], [-3, 3], 'r--', alpha=0.5)  # y=x line
                    axes[1, 0].set_title('Scatter: Our vs Saved (Lead I)')
                    axes[1, 0].set_xlabel('Our pipeline')
                    axes[1, 0].set_ylabel('Saved file')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Histogram of differences
                    axes[1, 1].hist(np.abs(our_normalized - processed_signal).flatten(), bins=50, alpha=0.7)
                    axes[1, 1].set_title('Histogram of All Differences')
                    axes[1, 1].set_xlabel('Absolute Difference')
                    axes[1, 1].set_ylabel('Count')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    output_dir = Path(__file__).parent.parent / "notebooks" / "verification_outputs"
                    output_path = output_dir / "pipeline_comparison.png"
                    plt.savefig(output_path, dpi=150)
                    plt.close()
                    
                    print(f"  Comparison plot saved: {output_path}")
            else:
                print(f"  Shape mismatch: Our {our_normalized.shape} vs Saved {processed_signal.shape}")
        else:
            print("  Processed file not found. Run preprocess_ptbxl.py first.")
            
    except Exception as e:
        print(f"  Error during comparison: {e}")


def load_ptbxl_record():
    """Helper to load PTB-XL record."""
    import wfdb
    ptb_dir = Path("data/raw/ptbxl")
    record_path = ptb_dir / "records100/00000/00001_lr"
    record = wfdb.rdrecord(str(record_path))
    return record.p_signal.astype(np.float32), float(record.fs)


if __name__ == "__main__":
    test_ptbxl_pipeline()