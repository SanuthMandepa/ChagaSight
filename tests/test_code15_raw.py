# tests/test_code15_raw.py
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

def test_code15_raw():
    print("Starting CODE-15% raw data test...")
    
    raw_dir = Path("data/raw/code15")
    if not raw_dir.exists():
        print("❌ Raw directory does not exist.")
        return
    
    csv_path = raw_dir / "exams.csv"
    if not csv_path.exists():
        print("❌ exams.csv not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"🔢 Total exams: {len(df)}")
    
    # Check first shard
    sample_trace = df.iloc[0]['trace_file']
    h5_path = raw_dir / sample_trace
    if not h5_path.exists():
        print(f"❌ Sample HDF5 not found: {h5_path} (ensure unzipped)")
        return
    
    with h5py.File(h5_path, 'r') as h5:
        exam_ids = h5['exam_id'][:]
        tracings = h5['tracings']
        print(f"📊 Sample shard size: {tracings.shape[0]} exams")
        
        # Load and plot first signal
        signal = tracings[0].astype(np.float32)
        fs = 400.0
        t = np.arange(signal.shape[0]) / fs
        plt.figure(figsize=(12, 4))
        plt.plot(t, signal[:, 0])
        plt.title("CODE-15% Raw ECG (Lead I) - First Exam")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        output_dir = Path("tests/verification_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "code15_raw_test.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"✅ Plot saved: {output_path}")

if __name__ == "__main__":
    test_code15_raw()