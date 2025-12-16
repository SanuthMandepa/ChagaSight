"""
Test the preprocessing for SaMi-Trop dataset.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess_samitrop import preprocess_single_exam

def test_samitrop_preprocessing():
    # Load a single exam from the raw data
    sami_root = Path("data/raw/sami_trop")
    
    try:
        import h5py
        import pandas as pd
        
        csv_path = sami_root / "exams.csv"
        df = pd.read_csv(csv_path)
        exam_id = df.iloc[0]["exam_id"]
        
        h5_path = sami_root / "exams.hdf5"
        with h5py.File(h5_path, "r") as h5:
            tracings = h5["tracings"]
            # Get the first exam
            raw_signal = tracings[0, :, :]  # (4096, 12)
        
        print(f"Testing exam_id: {exam_id}")
        print(f"Raw signal shape: {raw_signal.shape}")
        print(f"Raw signal range: {raw_signal.min()} to {raw_signal.max()}")
        
        # Preprocess the signal
        processed = preprocess_single_exam(
            raw_signal,
            fs_in=400.0,
            fs_target=400.0,
            duration_sec=10.0,
            baseline_window_sec=0.8
        )
        
        print(f"Processed signal shape: {processed.shape}")
        print(f"Processed signal range: {processed.min()} to {processed.max()}")
        print(f"Processed signal mean (per lead): {processed.mean(axis=0)}")
        print(f"Processed signal std (per lead): {processed.std(axis=0)}")
        
        # Plot the raw and processed signals for lead I
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        # Raw signal (lead I)
        t_raw = np.arange(raw_signal.shape[0]) / 400.0
        axes[0].plot(t_raw, raw_signal[:, 0])
        axes[0].set_title(f"Raw ECG - Exam {exam_id} (Lead I)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        
        # Processed signal (lead I)
        t_proc = np.arange(processed.shape[0]) / 400.0
        axes[1].plot(t_proc, processed[:, 0])
        axes[1].set_title(f"Processed ECG (Lead I)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Amplitude (z-score)")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path(__file__).parent.parent / "notebooks" / "verification_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "samitrop_preprocessing_test.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"\nPlot saved to: {output_path}")
        
        # Also plot all leads for the processed signal to see if they look reasonable
        fig2, axes2 = plt.subplots(3, 4, figsize=(16, 8))
        axes2 = axes2.flatten()
        
        for lead in range(12):
            axes2[lead].plot(t_proc, processed[:, lead])
            axes2[lead].set_title(f"Lead {lead+1}")
            axes2[lead].set_xlabel("Time (s)")
            axes2[lead].set_ylabel("Amplitude")
            axes2[lead].grid(True, alpha=0.3)
        
        plt.suptitle(f"Processed ECG (All Leads) - Exam {exam_id}")
        plt.tight_layout()
        
        output_path2 = output_dir / "samitrop_preprocessing_all_leads.png"
        plt.savefig(output_path2, dpi=150)
        plt.close()
        
        print(f"All leads plot saved to: {output_path2}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_samitrop_preprocessing()