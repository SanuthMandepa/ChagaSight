# tests/analyze_samitrop_signals.py
"""Analyze SaMi-Trop signal characteristics."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_samitrop_signal_characteristics():
    """Analyze SaMi-Trop signal characteristics."""
    print("🔍 Analyzing SaMi-Trop signal characteristics...")
    print("="*70)
    
    # Load data
    sami_dir = Path("data/raw/sami_trop")
    csv_path = sami_dir / "exams.csv"
    h5_path = sami_dir / "exams.hdf5"
    
    df = pd.read_csv(csv_path)
    
    with h5py.File(h5_path, 'r') as f:
        tracings = f['tracings']
        
        # Analyze first 100 exams
        sample_size = min(100, len(df))
        stats = {
            'min': [], 'max': [], 'range': [],
            'mean': [], 'std': [],
            'non_zero_length': [],
            'is_7s': [], 'is_10s': []
        }
        
        print(f"Analyzing {sample_size} random exams...")
        
        for i in range(sample_size):
            signal = tracings[i]  # (4096, 12)
            
            # Calculate statistics for lead I
            stats['min'].append(signal[:, 0].min())
            stats['max'].append(signal[:, 0].max())
            stats['range'].append(signal[:, 0].max() - signal[:, 0].min())
            stats['mean'].append(signal[:, 0].mean())
            stats['std'].append(signal[:, 0].std())
            
            # Find non-zero length
            threshold = 0.01
            signal_abs = np.abs(signal[:, 0])
            non_zero_mask = signal_abs > threshold
            if np.any(non_zero_mask):
                start = np.argmax(non_zero_mask)
                end = len(non_zero_mask) - np.argmax(non_zero_mask[::-1])
                length = end - start
                stats['non_zero_length'].append(length)
                
                # Classify as 7s or 10s
                if abs(length - 2800) < 100:
                    stats['is_7s'].append(True)
                    stats['is_10s'].append(False)
                elif abs(length - 4000) < 100:
                    stats['is_7s'].append(False)
                    stats['is_10s'].append(True)
                else:
                    stats['is_7s'].append(False)
                    stats['is_10s'].append(False)
            else:
                stats['non_zero_length'].append(0)
                stats['is_7s'].append(False)
                stats['is_10s'].append(False)
    
    print("\n📊 SaMi-Trop Signal Statistics:")
    print(f"   Min value: {np.mean(stats['min']):.4f} ± {np.std(stats['min']):.4f}")
    print(f"   Max value: {np.mean(stats['max']):.4f} ± {np.std(stats['max']):.4f}")
    print(f"   Range: {np.mean(stats['range']):.4f} ± {np.std(stats['range']):.4f}")
    print(f"   Mean: {np.mean(stats['mean']):.4f} ± {np.std(stats['mean']):.4f}")
    print(f"   Std: {np.mean(stats['std']):.4f} ± {np.std(stats['std']):.4f}")
    print(f"   Non-zero length: {np.mean(stats['non_zero_length']):.1f} ± {np.std(stats['non_zero_length']):.1f}")
    print(f"   7-second signals: {100*np.mean(stats['is_7s']):.1f}%")
    print(f"   10-second signals: {100*np.mean(stats['is_10s']):.1f}%")
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Histogram of signal ranges
    axes[0, 0].hist(stats['range'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Signal Range')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Signal Ranges')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram of non-zero lengths
    axes[0, 1].hist(stats['non_zero_length'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=2800, color='red', linestyle='--', label='7s (2800 samples)')
    axes[0, 1].axvline(x=4000, color='blue', linestyle='--', label='10s (4000 samples)')
    axes[0, 1].set_xlabel('Non-Zero Length (samples)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Signal Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter: mean vs std
    axes[0, 2].scatter(stats['mean'], stats['std'], alpha=0.5)
    axes[0, 2].set_xlabel('Mean')
    axes[0, 2].set_ylabel('Standard Deviation')
    axes[0, 2].set_title('Mean vs Standard Deviation')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Example 7s signal (if exists)
    seven_sec_indices = [i for i, is_7s in enumerate(stats['is_7s']) if is_7s]
    if seven_sec_indices:
        idx = seven_sec_indices[0]
        with h5py.File(h5_path, 'r') as f:
            signal = f['tracings'][idx]
        axes[1, 0].plot(signal[:, 0])
        axes[1, 0].set_xlabel('Samples')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].set_title(f'7-Second Signal Example (Lead I)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Example 10s signal (if exists)
    ten_sec_indices = [i for i, is_10s in enumerate(stats['is_10s']) if is_10s]
    if ten_sec_indices:
        idx = ten_sec_indices[0]
        with h5py.File(h5_path, 'r') as f:
            signal = f['tracings'][idx]
        axes[1, 1].plot(signal[:, 0])
        axes[1, 1].set_xlabel('Samples')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].set_title(f'10-Second Signal Example (Lead I)')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Distribution of signal types
    signal_types = ['7s', '10s', 'Other']
    counts = [np.sum(stats['is_7s']), np.sum(stats['is_10s']), 
              sample_size - np.sum(stats['is_7s']) - np.sum(stats['is_10s'])]
    axes[1, 2].bar(signal_types, counts, alpha=0.7)
    axes[1, 2].set_xlabel('Signal Type')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Distribution of Signal Types')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('SaMi-Trop Signal Characteristics Analysis', fontsize=16)
    plt.tight_layout()
    
    output_path = Path("notebooks/verification_outputs/samitrop_signal_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n📈 Analysis plot saved: {output_path}")
    print("="*70)

if __name__ == "__main__":
    analyze_samitrop_signal_characteristics()