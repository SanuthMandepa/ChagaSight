from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_signal
from src.preprocessing.image_embedding import ecg_to_structured_image

import pandas as pd
import numpy as np
import wfdb
import os

def process_record(record_path, output_dir):
    signal, meta = wfdb.rdsamp(record_path)
    signal = remove_baseline(signal)
    signal = resample_signal(signal, target_fs=500)
    
    np.save(f"{output_dir}/{meta['record_name']}.npy", signal)

if __name__ == "__main__":
    # load PTB-XL metadata CSV
    df = pd.read_csv("data/raw/ptbxl/ptbxl_database.csv")
    
    out = "data/processed/1d_signals/ptbxl"
    os.makedirs(out, exist_ok=True)

    for i, row in df.iterrows():
        process_record(row['filename_hr'], out)
