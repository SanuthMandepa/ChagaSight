import glob
import pandas as pd
import numpy as np
import os

# 1. Count Samples
def count_images(dataset):
    path = f'data/processed/2d_images/{dataset}/*_img.npy'
    count = len(glob.glob(path))
    print(f"{dataset.upper()} images: {count}")
    return count

count_images('ptbxl')  # Expected ~21799
count_images('sami_trop')  # ~1631
count_images('code15')  # ~34578 for 0.1%

# 2. Inspect Sample Image (use PTB-XL ID 1)
img_path = 'data/processed/2d_images/ptbxl/1_img.npy'
if os.path.exists(img_path):
    img = np.load(img_path)
    print(f"Shape: {img.shape} (expected (3,24,2048))")
    print(f"Dtype: {img.dtype} (expected float32)")
    print(f"Min/Max: {img.min():.2f}/{img.max():.2f} (expected ~ -3/3)")
else:
    print("Error: Sample image missing—re-run build_images.py")

# 3. Labels Availability & Merge for CODE-15%
# PTB-XL and SaMi-Trop: Good (all 0.0/1.0)
ptb_df = pd.read_csv('data/processed/metadata/ptbxl_processed.csv')
sami_df = pd.read_csv('data/processed/metadata/sami_trop_processed.csv')
print(f"PTB-XL labels: {ptb_df['label'].unique()} (expected [0.0])")
print(f"SaMi-Trop labels: {sami_df['label'].unique()} (expected [1.0])")

# CODE-15%: Missing 'label'—major issue. Solve: Download Chagas labels from PhysioNet.
# Steps:
# a. Go to https://moody-challenge.physionet.org/2025/ (register if needed).
# b. Download 'Chagas labels for CODE-15%' (likely a CSV with exam_id and chagas_positive binary).
# c. Save as 'data/raw/code15/chagas_labels.csv' (assume columns: exam_id, chagas_positive [1/0]).
# d. Run below to merge soft labels (0.8 for positive, 0.2 for negative per soft_labels.py).

code_df = pd.read_csv('data/processed/metadata/code15_processed.csv')
if 'label' not in code_df.columns:
    # Load downloaded labels (replace path if different)
    labels_path = 'data/raw/code15/chagas_labels.csv'  # Download here
    if os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        # Merge on exam_id
        merged = pd.merge(code_df, labels_df, on='exam_id', how='left')
        # Apply soft labels
        merged['label'] = merged['chagas_positive'].apply(lambda x: 0.8 if x == 1 else 0.2)
        merged.drop('chagas_positive', axis=1, inplace=True)  # Clean up
        merged.to_csv('data/processed/metadata/code15_processed.csv', index=False)
        print("Labels merged! Check code15_processed.csv for 'label' column.")
    else:
        print("Error: Download chagas_labels.csv from https://moody-challenge.physionet.org/2025/ and place in data/raw/code15/")
else:
    print("Labels already present in CODE-15% CSV—good!")

# 4. Balance Check (run after merge)
all_dfs = {'ptbxl': ptb_df, 'sami_trop': sami_df, 'code15': pd.read_csv('data/processed/metadata/code15_processed.csv')}
for ds, df in all_dfs.items():
    if 'label' in df.columns:
        print(f"{ds.upper()} balance:\n{df['label'].value_counts(normalize=True)}")
    else:
        print(f"{ds.upper()}: No 'label' column")

# 5. Metadata Integrity (ID match)
for ds in ['ptbxl', 'sami_trop', 'code15']:
    df = all_dfs[ds]
    if 'label' in df.columns:
        sample_ids = df.iloc[:5][all_dfs[ds].columns[0]]  # First column is ID
        for id_val in sample_ids:
            img_path = f'data/processed/2d_images/{ds}/{id_val}_img.npy'
            if os.path.exists(img_path):
                print(f"{ds} ID {id_val}: Image exists")
            else:
                print(f"Error: {ds} ID {id_val}: Image missing—re-run build_images.py")