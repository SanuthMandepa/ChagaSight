"""
Test the preprocessing for SaMi-Trop dataset (dual FS).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_samitrop_preprocessing():
    sami_root = Path("data/raw/sami_trop")
    output_dir = Path("tests/verification_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sample raw
    csv_df = pd.read_csv(sami_root / "exams.csv")
    exam_id = csv_df.iloc[0]["exam_id"]
    h5_idx = csv_df[csv_df['exam_id'] == exam_id].index[0]
    h5 = h5py.File(sami_root / "exams.hdf5", 'r')
    raw_signal = h5['tracings'][h5_idx].astype(np.float32)
    raw_fs = 400.0
    h5.close()

    # Load processed
    proc_500 = np.load(Path(f"data/processed/1d_signals_500hz/sami_trop/{exam_id}.npy"))
    proc_100 = np.load(Path(f"data/processed/1d_signals_100hz/sami_trop/{exam_id}.npy"))

    # Plots
    lead = 0
    t_raw = np.arange(raw_signal.shape[0]) / raw_fs
    t_500 = np.arange(proc_500.shape[0]) / 500.0
    t_100 = np.arange(proc_100.shape[0]) / 100.0

    plt.figure(figsize=(12, 4))
    plt.plot(t_raw, raw_signal[:, lead], label="Raw")
    plt.plot(t_500, proc_500[:, lead], label="Processed 500 Hz")
    plt.title(f"SaMi-Trop Raw vs 500 Hz (Exam {exam_id} Lead I)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "samitrop_raw_vs_500hz.png")
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(t_raw, raw_signal[:, lead], label="Raw")
    plt.plot(t_100, proc_100[:, lead], label="Processed 100 Hz")
    plt.title(f"SaMi-Trop Raw vs 100 Hz (Exam {exam_id} Lead I)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "samitrop_raw_vs_100hz.png")
    plt.close()

    print("✅ SaMi-Trop preprocessing test complete. Plots saved.")

if __name__ == "__main__":
    test_samitrop_preprocessing()