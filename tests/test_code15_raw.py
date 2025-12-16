# tests/test_code15_raw_simple.py
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

def test_code15_raw_simple():
    print("Starting CODE-15% raw data test...")
    
    raw_dir = Path("data/raw/code15")
    print(f"Raw directory: {raw_dir.absolute()}")
    
    # Check if directory exists
    if not raw_dir.exists():
        print("❌ Raw directory does not exist.")
        return
    
    # Check exams.csv
    csv_path = raw_dir / "exams.csv"
    print(f"Checking exams.csv at: {csv_path}")
    if not csv_path.exists():
        print("❌ exams.csv not found.")
        return
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded exams.csv with {len(df)} rows.")
        print(f"   Columns: {list(df.columns)}")
        
        # Check if there is an exam_id column
        if 'exam_id' not in df.columns:
            print("❌ No 'exam_id' column in exams.csv")
            return
        
        # Get the first exam_id
        first_exam_id = df.iloc[0]['exam_id']
        print(f"   First exam_id: {first_exam_id}")
        
        # Check the trace_file for the first exam
        if 'trace_file' not in df.columns:
            print("❌ No 'trace_file' column in exams.csv")
            return
        
        trace_file = df.iloc[0]['trace_file']
        print(f"   First trace_file: {trace_file}")
        
        # Check if the HDF5 file exists
        h5_path = raw_dir / trace_file
        print(f"   HDF5 path: {h5_path}")
        if not h5_path.exists():
            print(f"❌ HDF5 file not found: {h5_path}")
            return
        
        # Try to open the HDF5 file
        print("   Opening HDF5 file...")
        with h5py.File(h5_path, 'r') as f:
            print(f"   ✅ HDF5 file opened successfully.")
            print(f"   Keys in HDF5 file: {list(f.keys())}")
            
            if 'exam_id' not in f.keys():
                print("❌ No 'exam_id' dataset in HDF5 file.")
                return
            
            if 'tracings' not in f.keys():
                print("❌ No 'tracings' dataset in HDF5 file.")
                return
            
            exam_ids = np.array(f['exam_id'])
            tracings = f['tracings']
            
            print(f"   Number of exams in HDF5: {len(exam_ids)}")
            print(f"   Tracings shape: {tracings.shape}")
            
            # Find the index of the first exam_id
            idx = np.where(exam_ids == first_exam_id)[0]
            if len(idx) == 0:
                print(f"❌ Exam {first_exam_id} not found in HDF5.")
                return
            
            idx = idx[0]
            print(f"   Index of exam {first_exam_id}: {idx}")
            
            # Load the tracing
            signal = tracings[idx]
            print(f"   Signal shape: {signal.shape}")
            print(f"   Signal dtype: {signal.dtype}")
            print(f"   Signal range: {signal.min()} to {signal.max()}")
            
            # Plot the first lead
            plt.figure(figsize=(10, 4))
            plt.plot(signal[:, 0])
            plt.title(f"CODE-15% Exam {first_exam_id} - Lead I")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            
            output_dir = Path("notebooks/verification_outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "code15_raw_simple_test.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            print(f"✅ Plot saved to: {output_path}")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_code15_raw_simple()