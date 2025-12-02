# 🧬 ChagaSight — ViT-Based ECG Image & Foundation Model Pipeline for Chagas Disease Detection

_A Multi-Dataset, Reproducible Research Framework using 2D Vision Transformers (ViT), 1D ECG Foundation Models (FM), and Hybrid Feature Alignment._

ChagaSight is a modular deep-learning framework for **Chagas disease detection** from **12-lead ECGs**, integrating:

- **2D Vision Transformer (ViT)** trained on physiologically structured ECG images
- **1D Vision Transformer Foundation Model (ECG-FM)** trained via masked self-supervision (ST-MEM)
- **Hybrid Alignment Model** aligning FM signal-level representations with ViT image embeddings

The pipeline draws from two modern research advancements:

- _Detecting Chagas Disease Using a Vision Transformer–Based ECG Foundation Model (2025)_
- _Embedding ECG Signals into 2D Images with Preserved Spatial Information (2025)_

---

# 📌 1. Project Overview

ChagaSight provides an end-to-end deep learning workflow for handling ECG signals and images:

### ✔ Multi-dataset preprocessing (PTB-XL, CODE-15%, SaMi-Trop)

### ✔ Two complementary modeling paths

- **2D ViT Image Classifier** (primary baseline model)
- **1D Foundation Model (FM)** for ECG signal representation

### ✔ Hybrid feature alignment (optional extension)

### ✔ Challenge-oriented evaluation

- Top-5% TPR (PhysioNet metric)
- AUROC, AUPRC, F1
- Grad-CAM for both signals & images

The framework is designed so that **ViT image-based modeling is implemented first**, and **the ECG Foundation Model can be integrated later** for improved robustness and transfer learning.

---

# 📊 2. Supported Datasets

ChagaSight supports the three open-access 12-lead ECG datasets used in the George Moody PhysioNet Challenge (2025):

| Dataset       | Description                          | Sample Rate | Chagas Label               |
| ------------- | ------------------------------------ | ----------- | -------------------------- |
| **PTB-XL**    | 21k European ECGs                    | 100/500 Hz  | All **negative (0)**       |
| **CODE-15%**  | 345k Brazilian ECGs                  | 400 Hz      | **Soft labels: 0.2 / 0.8** |
| **SaMi-Trop** | 1631 serology-confirmed Chagas cases | 400 Hz      | All **positive (1)**       |

### ✔ Label Policy

Consistent with state-of-the-art research:

| Dataset   | Confidence           | Assigned Label |
| --------- | -------------------- | -------------- |
| PTB-XL    | Very strong negative | **0**          |
| SaMi-Trop | Very strong positive | **1**          |
| CODE-15%  | Weak (self-reported) | **0.2 / 0.8**  |

---

# 📁 3. Folder Structure (2025 Architecture)

ChagaSight/
│
├── data/
│ ├── raw/
│ │ ├── ptbxl/
│ │ ├── code15/
│ │ └── sami_trop/
│ ├── processed/
│ │ ├── 1d_signals/ # baseline-corrected, resampled, z-scored
│ │ └── 2d_images/ # structured 3-channel ECG images (24×2048)
│ └── splits/ # patient-level train/val/test JSON
│
├── src/
│ ├── preprocessing/
│ │ ├── baseline_removal.py
│ │ ├── resample.py
│ │ ├── image_embedding.py # RA/LA/LL contour → 2D images
│ │ └── soft_labels.py
│ │
│ ├── foundation_model/
│ │ ├── vit_1d_encoder.py
│ │ ├── st_mem_pretraining.py
│ │ ├── aol_mixing.py
│ │ └── fm_feature_extractor.py
│ │
│ ├── image_model/
│ │ ├── vit_image_encoder.py
│ │ ├── projection_head.py
│ │ └── alignment_loss.py
│ │
│ ├── dataloaders/
│ │ ├── ptbxl_loader.py
│ │ ├── code15_loader.py
│ │ ├── sami_loader.py
│ │ ├── fm_signal_dataset.py
│ │ └── image_dataset.py
│ │
│ ├── training/
│ │ ├── train_fm.py # 1D FM pretraining (ST-MEM)
│ │ ├── train_image_model.py # 2D ViT classifier
│ │ ├── train_hybrid.py # FM + ViT alignment
│ │ ├── augmentations_1d.py
│ │ └── augmentations_2d.py
│ │
│ └── evaluation/
│ ├── metrics.py
│ ├── explainability.py
│ └── challenge_metric.py
│
├── scripts/ # <--- THIS IS THE FOLDER YOU ASKED FOR
│ ├── preprocess_ptbxl.py # run preprocessing for PTB-XL
│ ├── preprocess_code15.py # run preprocessing for CODE-15%
│ ├── preprocess_samitrop.py # run preprocessing for SaMi-Trop
│ ├── build_images.py # convert 1D → 2D ECG images
│ ├── create_splits.py # build patient-level splits
│ ├── train_vit_image.sh # shell script for ViT training
│ ├── train_fm.sh # shell script for FM training
│ └── train_hybrid.sh # shell script for hybrid alignment
│
├── notebooks/
│ ├── 01_preprocessing_1d.ipynb
│ ├── 02_image_embedding.ipynb
│ ├── 03_fm_pretraining.ipynb
│ ├── 04_cross_validation.ipynb
│ ├── 05_hybrid_alignment_training.ipynb
│ └── 06_evaluation.ipynb
│
├── docs/
│ ├── methodology/
│ │ ├── fm_architecture.md
│ │ ├── image_embedding_diagram.png
│ │ └── augmentation_strategy.md
│ ├── figures/
│ ├── reports/
│ └── diagrams/
│
├── experiments/
│ ├── fm_pretraining/
│ ├── image_baseline/
│ ├── hybrid_alignment/
│ └── logs/
│
├── results/ # gitignored
├── requirements.txt
└── README.md

---

# 🔧 4. Preprocessing Pipeline

ChagaSight adopts a **two-stage preprocessing strategy** inspired by recent ECG research.

## **Stage 1 — 1D Signal Preprocessing**

- Baseline drift removal
- Resampling to a unified frequency
- Trim/pad to a fixed duration
- Z-score normalization
- Save as `.npy` per record

## **Stage 2 — ECG → Structured 2D Image Conversion**

Based on physiologically meaningful lead contour maps:

- Construct 3 channels representing RA, LA, LL reference contours
- Clip signal to [−3, 3]
- Map amplitude to [0–255]
- Resize to **3 × 24 × 2048**
- Save as `.npy` or `.png`

This representation is **optimized for Vision Transformer input**.

---

# 🧠 5. Model Components

## **A. 2D ViT Image Encoder (Primary Model)**

- Input: structured ECG images
- Patch embeddings adapted for rectangular biomedical images
- Outputs:
  - Classification logits
  - Image-level latent embeddings

This is the **primary baseline model** and the recommended starting point for experimentation.

---

## **B. 1D Vision Transformer Foundation Model (Optional Extension)**

- 12-layer encoder
- Patch size = 50
- Self-supervised training: **ST-MEM masked ECG reconstruction**
- Multi-layer feature aggregation (AoL)

Provides robust signal-level ECG embeddings, especially useful in low-label or multi-dataset setups.

---

## **C. Hybrid FM + ViT Alignment Model (Optional Advanced Model)**

A REPA-inspired alignment loss encourages consistency between signal embeddings (1D FM) and image embeddings (2D ViT):

L_total = L_classification + λ \* cosine_similarity(FM_features, ViT_features)

This enhances invariance and robustness to confounders (e.g., sampling frequency, noise).

---

# 🧪 6. Training Workflow

### **1. Preprocess 1D ECG Signals**

notebooks/01_preprocessing_1d.ipynb

### **2. Convert Processed Signals to Images**

notebooks/02_image_embedding.ipynb

### **3. Train the ViT Image Baseline (recommended first model)**

python src/training/train_image_model.py

### **4. (Optional) Pretrain 1D ECG Foundation Model**

python src/training/train_fm.py

### **5. (Optional) Train Hybrid Alignment Model**

python src/training/train_hybrid.py

### **6. Evaluate**

python src/evaluation/metrics.py

---

# 📈 7. Evaluation Metrics

- PhysioNet Challenge Top-5% TPR
- AUROC / AUPRC
- F1-score
- Calibration metrics
- Confusion matrices
- Signal-level + Image-level Grad-CAM
- Dataset shift robustness tests

---

# 🔍 8. Key Contributions of This Pipeline

### **2D ViT Image Modeling**

- Enables direct use of modern vision architectures
- Uses physiologically accurate multi-channel ECG image construction

### **1D ECG Foundation Modeling**

- Captures deep, transferable ECG signal characteristics
- Based on ST-MEM masked self-supervised learning

### **Hybrid Alignment**

- Leverages both signal and image representations
- Improves generalization and robustness

### **Multi-Dataset Unified Framework**

- PTB-XL, CODE-15%, SaMi-Trop integrated under one consistent pipeline

---

# 🚀 9. Roadmap

- [ ] Complete preprocessing for all datasets
- [ ] Train ViT image baseline
- [ ] Train ECG-FM using ST-MEM
- [ ] Integrate hybrid alignment
- [ ] Perform 5-fold cross-validation
- [ ] Generate final evaluation report with visual explanations

---

# 📬 Contact

See the `docs/` directory for methodology, architecture notes, and experiment logs.
