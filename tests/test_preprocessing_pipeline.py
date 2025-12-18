# tests/test_preprocessing_pipeline.py

"""
Test the preprocessing pipeline on sample IDs from all datasets (dual FS).
Uses known preprocessed IDs to avoid missing files.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from validate_single_ecg import load_raw_signal

# Use IDs that you KNOW are preprocessed
SAMPLE_IDS = {
    'ptbxl': 1,           # Always exists
    'sami_trop': 294669,  # From your successful run
    'code15': 4199464     # Your confirmed preprocessed CODE-15% ID
}

def test_preprocessing_pipeline():
    output_dir = Path("tests/verification_outputs/pipeline_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, id_used in SAMPLE_IDS.items():
        try:
            print(f"Testing {dataset.upper()} with ID: {id_used}")

            # Load raw
            raw, raw_fs = load_raw_signal(dataset, id_used)

            # Load processed (these now exist)
            proc_500 = np.load(Path(f"data/processed/1d_signals_500hz/{dataset}/{id_used}.npy"))
            proc_100 = np.load(Path(f"data/processed/1d_signals_100hz/{dataset}/{id_used}.npy"))

            # Plots (Lead I)
            lead = 0
            t_raw = np.arange(raw.shape[0]) / raw_fs
            t_500 = np.arange(proc_500.shape[0]) / 500.0
            t_100 = np.arange(proc_100.shape[0]) / 100.0

            # Raw vs 500 Hz
            plt.figure(figsize=(12, 4))
            plt.plot(t_raw, raw[:, lead], label="Raw", alpha=0.8)
            plt.plot(t_500, proc_500[:, lead], label="Processed 500 Hz (z-scored)")
            plt.title(f"{dataset.upper()} - Raw vs 500 Hz (ID {id_used}, Lead I)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / f"{dataset}_raw_vs_500hz.png", dpi=150)
            plt.close()

            # Raw vs 100 Hz
            plt.figure(figsize=(12, 4))
            plt.plot(t_raw, raw[:, lead], label="Raw", alpha=0.8)
            plt.plot(t_100, proc_100[:, lead], label="Processed 100 Hz (raw)")
            plt.title(f"{dataset.upper()} - Raw vs 100 Hz (ID {id_used}, Lead I)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / f"{dataset}_raw_vs_100hz.png", dpi=150)
            plt.close()

            print(f"   ✅ Plots saved for {dataset}")

        except Exception as e:
            print(f"   ❌ Error for {dataset} (ID {id_used}): {e}")

    print("\n✅ Dual-FS preprocessing pipeline test complete!")
    print(f"   All plots saved in: {output_dir}")


if __name__ == "__main__":
    test_preprocessing_pipeline()