# tests/check_raw_data.py

"""
Quick check of raw ECG data for all datasets.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import h5py
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_raw_data():
    output_dir = Path("tests/verification_outputs/raw_checks")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ['ptbxl', 'sami_trop', 'code15']

    for dataset in datasets:
        try:
            if dataset == 'ptbxl':
                path = Path("data/raw/ptbxl/records500/00000/00001_hr")
                record = wfdb.rdrecord(str(path))
                signal = record.p_signal.astype(np.float32)
                fs = record.fs
                used_id = "1 (high-res sample)"

            elif dataset == 'sami_trop':
                csv_path = Path("data/raw/sami_trop/exams.csv")
                csv_df = pd.read_csv(csv_path)
                if csv_df.empty:
                    raise ValueError("SaMi-Trop exams.csv is empty")
                # Use first valid exam_id from CSV
                exam_id = csv_df['exam_id'].iloc[0]
                h5_idx = csv_df[csv_df['exam_id'] == exam_id].index[0]
                h5 = h5py.File(Path("data/raw/sami_trop/exams.hdf5"), 'r')
                signal = h5['tracings'][h5_idx].astype(np.float32)
                fs = 400.0
                h5.close()
                used_id = exam_id

            elif dataset == 'code15':
                csv_df = pd.read_csv(Path("data/raw/code15/exams.csv"))
                row = csv_df.iloc[0]  # First row
                exam_id = row['exam_id']
                trace_file = row['trace_file']
                h5 = h5py.File(Path("data/raw/code15") / trace_file, 'r')
                idx = np.where(h5['exam_id'][:] == exam_id)[0][0]
                signal = h5['tracings'][idx].astype(np.float32)
                fs = 400.0
                h5.close()
                used_id = exam_id

            t = np.arange(signal.shape[0]) / fs
            plt.figure(figsize=(12, 4))
            plt.plot(t, signal[:, 0])
            plt.title(f"{dataset.upper()} Raw ECG (Lead I) - ID {used_id}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.savefig(output_dir / f"{dataset}_raw_check.png")
            plt.close()
            print(f"✅ Plot saved for {dataset} (ID: {used_id})")

        except Exception as e:
            print(f"❌ Error for {dataset}: {e}")

if __name__ == "__main__":
    check_raw_data()
    print("\n✅ Raw data checks completed for all datasets!")