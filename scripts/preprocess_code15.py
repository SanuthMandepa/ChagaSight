"""
Preprocess CODE-15% dataset into cleaned 1D ECG signals.

Pipeline:
- baseline removal (moving average)
- resampling to 400 Hz
- pad/trim to 10 seconds (4000 samples)
- per-lead z-score normalization

Output:
    data/processed/1d_signals/code15/{exam_id}.npy
"""

import os
import time
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Our preprocessing utilities
from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead


# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------
RAW_DIR = "data/raw/code15"
OUT_DIR = "data/processed/1d_signals/code15"

ORIGINAL_FS = 400.0      # CODE-15 HDF5 tracings are 400 Hz
TARGET_FS = 400.0        # must match PTB-XL & SaMi-Trop
TARGET_DURATION_SEC = 10
TARGET_SAMPLES = int(TARGET_DURATION_SEC * TARGET_FS)   # = 4000


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():

    print("⏱️ Starting: CODE-15% 1D preprocessing")
    t0 = time.time()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load metadata
    csv_path = os.path.join(RAW_DIR, "exams.csv")
    df = pd.read_csv(csv_path)

    print(f"📄 Loaded metadata: {csv_path}")
    print(f"🔢 Total exams listed: {len(df)}")

    # Group by HDF5 shard file
    part_groups = df.groupby("trace_file")

    total_processed = 0

    # ---------------------------------------------------
    # LOOP OVER EACH HDF5 PART FILE
    # ---------------------------------------------------
    for trace_file, group in part_groups:

        h5_path = os.path.join(RAW_DIR, trace_file)
        print(f"\n📂 Opening HDF5: {trace_file}  (rows: {len(group)})")

        with h5py.File(h5_path, "r") as h5:

            exam_ids = np.array(h5["exam_id"])
            tracings = h5["tracings"]      # shape = (N, 4096, 12)

            # FAST lookup map
            index_map = {int(exam_ids[i]): i for i in range(len(exam_ids))}

            # ---------------------------------------------------
            # PROCESS ALL ROWS IN THIS PART
            # ---------------------------------------------------
            for _, row in tqdm(group.iterrows(), total=len(group)):

                exam_id = int(row["exam_id"])

                # Skip if missing
                if exam_id not in index_map:
                    print(f"⚠️ exam_id {exam_id} missing in {trace_file}")
                    continue

                idx = index_map[exam_id]

                # FAST HDF5 read (avoids slowdown)
                raw = tracings[idx, :, :]          # (4096, 12)
                raw = np.asarray(raw)              # convert to numpy

                try:
                    # -------------------------
                    # 1. Baseline Removal
                    # -------------------------
                    cleaned = remove_baseline(
                        raw,
                        fs=ORIGINAL_FS,
                        method="moving_average",
                        window_seconds=0.8
                    )

                    # -------------------------
                    # 2. Resample to 400 Hz
                    # -------------------------
                    resampled, fs_rs = resample_ecg(
                        cleaned,
                        fs_in=ORIGINAL_FS,
                        fs_out=TARGET_FS
                    )

                    # -------------------------
                    # 3. Crop or pad to 4000 samples
                    # -------------------------
                    fixed = pad_or_trim(
                        resampled,
                        target_length=TARGET_SAMPLES
                    )

                    # -------------------------
                    # 4. Per-lead z-score normalization
                    # -------------------------
                    normed = zscore_per_lead(fixed)

                    # -------------------------
                    # 5. SAVE
                    # -------------------------
                    out_path = os.path.join(OUT_DIR, f"{exam_id}.npy")
                    np.save(out_path, normed)
                    total_processed += 1

                except Exception as e:
                    print(f"❌ Failed exam_id {exam_id}: {e}")
                    continue

    # ---------------------------------------------------
    # DONE
    # ---------------------------------------------------
    elapsed = time.time() - t0
    print("\n🎉 DONE: CODE-15% preprocessing")
    print(f"⏱️ Time: {elapsed:.2f} s  ({elapsed/60:.2f} min)")
    print(f"📦 Total processed: {total_processed}")
    print(f"💾 Saved to: {OUT_DIR}")


# ENTRY POINT
if __name__ == "__main__":
    main()
