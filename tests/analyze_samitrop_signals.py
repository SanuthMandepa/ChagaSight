# tests/analyze_samitrop_signals.py
"""Analyze SaMi-Trop signal characteristics (raw + processed)."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm  # ← ADD THIS IMPORT

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_samitrop_signal_characteristics():
    print("🔍 Analyzing SaMi-Trop signal characteristics...")
    print("="*70)
    
    # Load raw data
    sami_dir = Path("data/raw/sami_trop")
    csv_path = sami_dir / "exams.csv"
    df = pd.read_csv(csv_path)
    
    h5_path = sami_dir / "exams.hdf5"
    with h5py.File(h5_path, 'r') as h5:
        tracings = h5["tracings"]
        
        # Raw analysis
        lengths = []
        means = []
        stds = []
        types = []
        
        for i in tqdm(range(tracings.shape[0]), desc="Analyzing raw signals"):
            signal = tracings[i]
            lengths.append(signal.shape[0])
            means.append(np.mean(signal))
            stds.append(np.std(signal))
            
            if lengths[-1] < 3500:
                types.append('Short (~7s)')
            else:
                types.append('Full (~10s)')
        
        # Plots for raw
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes[0, 0].hist(lengths, bins=20)
        axes[0, 0].set_title('Raw Length Distribution')
        axes[0, 1].hist(means, bins=20)
        axes[0, 1].set_title('Raw Mean Distribution')
        axes[0, 2].hist(stds, bins=20)
        axes[0, 2].set_title('Raw Std Distribution')
        
        # Processed analysis (500 Hz and 100 Hz)
        proc_dir_500 = Path("data/processed/1d_signals_500hz/sami_trop")
        proc_dir_100 = Path("data/processed/1d_signals_100hz/sami_trop")
        
        proc_lengths_500 = []
        proc_means_500 = []
        proc_stds_500 = []
        proc_lengths_100 = []
        proc_means_100 = []
        proc_stds_100 = []
        
        npy_files_500 = list(proc_dir_500.glob("*.npy"))
        for npy in tqdm(npy_files_500, desc="Analyzing 500 Hz processed"):
            signal = np.load(npy)
            proc_lengths_500.append(signal.shape[0])
            proc_means_500.append(np.mean(signal))
            proc_stds_500.append(np.std(signal))
        
        npy_files_100 = list(proc_dir_100.glob("*.npy"))
        for npy in tqdm(npy_files_100, desc="Analyzing 100 Hz processed"):
            signal = np.load(npy)
            proc_lengths_100.append(signal.shape[0])
            proc_means_100.append(np.mean(signal))
            proc_stds_100.append(np.std(signal))
        
        # Plots for processed
        axes[1, 0].hist([proc_lengths_500, proc_lengths_100], bins=10, label=['500 Hz', '100 Hz'])
        axes[1, 0].set_title('Processed Length Distribution')
        axes[1, 0].legend()
        axes[1, 1].hist([proc_means_500, proc_means_100], bins=20, label=['500 Hz', '100 Hz'])
        axes[1, 1].set_title('Processed Mean Distribution')
        axes[1, 1].legend()
        axes[1, 2].hist([proc_stds_500, proc_stds_100], bins=20, label=['500 Hz', '100 Hz'])
        axes[1, 2].set_title('Processed Std Distribution')
        axes[1, 2].legend()
        
        plt.suptitle('SaMi-Trop Signal Analysis (Raw + Processed)', fontsize=16)
        plt.tight_layout()
        
        output_path = Path("tests/verification_outputs/samitrop_signal_analysis.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"\n📈 Analysis plot saved: {output_path}")
        print("="*70)

if __name__ == "__main__":
    analyze_samitrop_signal_characteristics()