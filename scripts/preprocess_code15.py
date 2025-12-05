"""
Preprocess CODE-15% dataset into standardized 1D ECG signals.

Features:
    - Optional --subset FRACTION   (e.g., 0.1 = 10% of CODE-15)
    - Avoids 50+ GB storage explosion
    - Tracks processing time per shard and total time
    - Efficient HDF5 access (loads only required rows)

Pipeline:
    1. Baseline removal (moving-average)
    2. Resample → 400 Hz
    3. Pad/trim → 10 seconds (4000 samples)
    4. Per-lead z-score normalization

Output:
    data/processed/1d_signals/code15/{exam_id}.npy
"""

import os
import time
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# preprocessing utilities
from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead


# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------
RAW_DIR = "data/raw/code15"
OUT_DIR = "data/processed/1d_signals/code15"

ORIGINAL_FS = 400.0
TARGET_FS = 400.0
TARGET_DURATION_SEC = 10
TARGET_SAMPLES = int(TARGET_DURATION_SEC * TARGET_FS)


# -------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------
def main(subset_fraction: float):
    print("===================================================")
    print("⏱️  Starting CODE-15% 1D Preprocessing Pipeline")
    print("===================================================\n")

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------------------------------
    # Load CSV metadata
    # ---------------------------------------
    csv_path = os.path.join(RAW_DIR, "exams.csv")
    df = pd.read_csv(csv_path)

    print(f"📄 Loaded metadata: {csv_path}")
    print(f"🔢 Total ECG exams found: {len(df)}")

    # ---------------------------------------
    # SUBSET MODE (prevents 50+ GB storage)
    # ---------------------------------------
    if subset_fraction < 1.0:
        df = df.sample(frac=subset_fraction, random_state=42)
        print(f"⚠️ SUBSET MODE ACTIVE → Using {len(df)} exams ({subset_fraction*100:.1f}%)")

    # Group rows by HDF5 part file
    part_groups = df.groupby("trace_file")
    total_processed = 0

    # ---------------------------------------------------
    # PROCESS EACH HDF5 SHARD
    # ---------------------------------------------------
    for trace_file, group in part_groups:

        shard_start = time.time()
        h5_path = os.path.join(RAW_DIR, trace_file)

        print(f"\n📂 Opening HDF5 file: {trace_file}  | Rows: {len(group)}")

        with h5py.File(h5_path, "r") as h5:

            exam_ids = np.array(h5["exam_id"])
            tracings = h5["tracings"]        # (N, 4096, 12)

            # Fast dictionary → exam_id → index
            index_map = {int(exam_ids[i]): i for i in range(len(exam_ids))}

            # -------------------------------------------
            # PROCESS EACH ECG IN THIS SHARD
            # -------------------------------------------
            for _, row in tqdm(group.iterrows(), total=len(group)):

                exam_id = int(row["exam_id"])

                if exam_id not in index_map:
                    print(f"⚠️ Missing exam_id {exam_id} in {trace_file}")
                    continue

                idx = index_map[exam_id]
                raw = np.asarray(tracings[idx, :, :])   # shape (4096,12)

                try:
                    # ------------------- 1. Baseline removal -------------------
                    cleaned = remove_baseline(
                        raw,
                        fs=ORIGINAL_FS,
                        method="moving_average",
                        window_seconds=0.8,
                    )

                    # ------------------- 2. Resample → 400 Hz -------------------
                    resampled, _ = resample_ecg(
                        cleaned,
                        fs_in=ORIGINAL_FS,
                        fs_out=TARGET_FS
                    )

                    # ------------------- 3. Pad/trim → 4000 samples -------------------
                    fixed = pad_or_trim(
                        resampled,
                        target_length=TARGET_SAMPLES
                    )

                    # ------------------- 4. Z-score per lead -------------------
                    normed = zscore_per_lead(fixed)

                    # ------------------- 5. SAVE -------------------
                    out_path = os.path.join(OUT_DIR, f"{exam_id}.npy")
                    np.save(out_path, normed)
                    total_processed += 1

                except Exception as e:
                    print(f"❌ ERROR processing exam_id {exam_id}: {e}")
                    continue

        shard_elapsed = time.time() - shard_start
        print(f"⏳ Shard completed in {shard_elapsed:.1f} sec ({shard_elapsed/60:.1f} min)")

    # ---------------------------------------------------
    # FINISHED
    # ---------------------------------------------------
    elapsed = time.time() - t0
    print("\n🎉 DONE: CODE-15% preprocessing complete!")
    print("===================================================")
    print(f"⏱️ Total time: {elapsed:.1f} sec ({elapsed/60:.1f} min)")
    print(f"📦 Total processed signals: {total_processed}")
    print(f"💾 Saved to: {OUT_DIR}")
    print("===================================================\n")


# -------------------------------------------------------
# CLI ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        type=float,
        default=0.1,
        help="Fraction of CODE-15 dataset to preprocess (default = 0.1 = 10%)"
    )

    args = parser.parse_args()
    main(args.subset)



# """
# Preprocess CODE-15% dataset into cleaned 1D ECG signals.

# Pipeline:
# - baseline removal (moving average)
# - resampling to 400 Hz
# - pad/trim to 10 seconds (4000 samples)
# - per-lead z-score normalization

# Output:
#     data/processed/1d_signals/code15/{exam_id}.npy
# """

# import os
# import time
# import h5py
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# # Our preprocessing utilities
# from src.preprocessing.baseline_removal import remove_baseline
# from src.preprocessing.resample import resample_ecg, pad_or_trim
# from src.preprocessing.normalization import zscore_per_lead


# # -------------------------------------------------------
# # CONSTANTS
# # -------------------------------------------------------
# RAW_DIR = "data/raw/code15"
# OUT_DIR = "data/processed/1d_signals/code15"

# ORIGINAL_FS = 400.0      # CODE-15 HDF5 tracings are 400 Hz
# TARGET_FS = 400.0        # must match PTB-XL & SaMi-Trop
# TARGET_DURATION_SEC = 10
# TARGET_SAMPLES = int(TARGET_DURATION_SEC * TARGET_FS)   # = 4000


# # -------------------------------------------------------
# # MAIN
# # -------------------------------------------------------
# def main():

#     print("⏱️ Starting: CODE-15% 1D preprocessing")
#     t0 = time.time()

#     os.makedirs(OUT_DIR, exist_ok=True)

#     # Load metadata
#     csv_path = os.path.join(RAW_DIR, "exams.csv")
#     df = pd.read_csv(csv_path)

#     print(f"📄 Loaded metadata: {csv_path}")
#     print(f"🔢 Total exams listed: {len(df)}")

#     # Group by HDF5 shard file
#     part_groups = df.groupby("trace_file")

#     total_processed = 0

#     # ---------------------------------------------------
#     # LOOP OVER EACH HDF5 PART FILE
#     # ---------------------------------------------------
#     for trace_file, group in part_groups:

#         h5_path = os.path.join(RAW_DIR, trace_file)
#         print(f"\n📂 Opening HDF5: {trace_file}  (rows: {len(group)})")

#         with h5py.File(h5_path, "r") as h5:

#             exam_ids = np.array(h5["exam_id"])
#             tracings = h5["tracings"]      # shape = (N, 4096, 12)

#             # FAST lookup map
#             index_map = {int(exam_ids[i]): i for i in range(len(exam_ids))}

#             # ---------------------------------------------------
#             # PROCESS ALL ROWS IN THIS PART
#             # ---------------------------------------------------
#             for _, row in tqdm(group.iterrows(), total=len(group)):

#                 exam_id = int(row["exam_id"])

#                 # Skip if missing
#                 if exam_id not in index_map:
#                     print(f"⚠️ exam_id {exam_id} missing in {trace_file}")
#                     continue

#                 idx = index_map[exam_id]

#                 # FAST HDF5 read (avoids slowdown)
#                 raw = tracings[idx, :, :]          # (4096, 12)
#                 raw = np.asarray(raw)              # convert to numpy

#                 try:
#                     # -------------------------
#                     # 1. Baseline Removal
#                     # -------------------------
#                     cleaned = remove_baseline(
#                         raw,
#                         fs=ORIGINAL_FS,
#                         method="moving_average",
#                         window_seconds=0.8
#                     )

#                     # -------------------------
#                     # 2. Resample to 400 Hz
#                     # -------------------------
#                     resampled, fs_rs = resample_ecg(
#                         cleaned,
#                         fs_in=ORIGINAL_FS,
#                         fs_out=TARGET_FS
#                     )

#                     # -------------------------
#                     # 3. Crop or pad to 4000 samples
#                     # -------------------------
#                     fixed = pad_or_trim(
#                         resampled,
#                         target_length=TARGET_SAMPLES
#                     )

#                     # -------------------------
#                     # 4. Per-lead z-score normalization
#                     # -------------------------
#                     normed = zscore_per_lead(fixed)

#                     # -------------------------
#                     # 5. SAVE
#                     # -------------------------
#                     out_path = os.path.join(OUT_DIR, f"{exam_id}.npy")
#                     np.save(out_path, normed)
#                     total_processed += 1

#                 except Exception as e:
#                     print(f"❌ Failed exam_id {exam_id}: {e}")
#                     continue

#     # ---------------------------------------------------
#     # DONE
#     # ---------------------------------------------------
#     elapsed = time.time() - t0
#     print("\n🎉 DONE: CODE-15% preprocessing")
#     print(f"⏱️ Time: {elapsed:.2f} s  ({elapsed/60:.2f} min)")
#     print(f"📦 Total processed: {total_processed}")
#     print(f"💾 Saved to: {OUT_DIR}")


# # ENTRY POINT
# if __name__ == "__main__":
#     main()
