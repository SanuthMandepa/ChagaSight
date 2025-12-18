"""
Test the preprocessing pipeline on sample IDs from all datasets (dual FS).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_IDS = {'ptbxl': 1, 'sami_trop': 94669, 'code15': 1169160}

def test_preprocessing_pipeline():
    output_dir = Path("tests/verification_outputs/pipeline_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, id in SAMPLE_IDS.items():
        # Load raw (reuse from validate_single_ecg)
        from validate_single_ecg import load_raw_signal
        raw, raw_fs = load_raw_signal(dataset, id)

        # Load processed
        proc_500 = np.load(Path(f"data/processed/1d_signals_500hz/{dataset}/{id}.npy"))
        proc_100 = np.load(Path(f"data/processed/1d_signals_100hz/{dataset}/{id}.npy"))

        # Plots
        lead = 0
        t_raw = np.arange(raw.shape[0]) / raw_fs
        t_500 = np.arange(proc_500.shape[0]) / 500.0
        t_100 = np.arange(proc_100.shape[0]) / 100.0

        plt.figure(figsize=(12, 4))
        plt.plot(t_raw, raw[:, lead], label="Raw")
        plt.plot(t_500, proc_500[:, lead], label="Processed 500 Hz")
        plt.title(f"{dataset.upper()} Pipeline Test: Raw vs 500 Hz (ID {id} Lead I)")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{dataset}_raw_vs_500hz.png")
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(t_raw, raw[:, lead], label="Raw")
        plt.plot(t_100, proc_100[:, lead], label="Processed 100 Hz")
        plt.title(f"{dataset.upper()} Pipeline Test: Raw vs 100 Hz (ID {id} Lead I)")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{dataset}_raw_vs_100hz.png")
        plt.close()

    print("\n✅ Pipeline tests complete for all datasets. Plots saved.")

if __name__ == "__main__":
    test_preprocessing_pipeline()