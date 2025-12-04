🧬 **ChagaSight — A Vision Transformer–Based ECG Image Pipeline for Chagas Disease Detection**

A Final-Year Research Project using Physiologically Structured 2D ECG Images and Optional 1D ECG Foundation Models

ChagaSight is a modular deep-learning framework designed to detect Chagas disease from 12-lead ECGs.
This project focuses primarily on transforming ECG signals into physiologically structured 2D images and training a Vision Transformer (ViT) classifier on these images.

In addition, ChagaSight includes an optional extension exploring a 1D ECG Foundation Model (FM) based on masked self-supervised pretraining (ST-MEM), and an optional hybrid alignment model that links 1D signal embeddings with 2D image embeddings.

The approach is inspired by two modern research pipelines:

Physiologically Structured 2D ECG Image Embedding (2025)

Vision Transformer Foundation Model for ECGs (2025)

# 📌 1. Project Overview

ChagaSight provides an end-to-end workflow for multi-dataset ECG processing, image generation, model training, and evaluation.

✔ Multi-Dataset ECG Preprocessing
PTB-XL, CODE-15%, and SaMi-Trop are unified by cleaning, resampling, trimming, and normalizing all signals.

✔ 2D ECG Image Representation (PRIMARY METHOD)
ECGs are converted into structured 3-channel images using RA/LA/LL contour mapping, producing ViT-ready images (3 × 24 × 2048).

✔ Vision Transformer (ViT) Image Classifier (MAIN MODEL)
The primary dissertation model: a ViT trained on 2D ECG images to classify Chagas disease.

✔ Optional: ECG Foundation Model (1D ViT-FM)
A transformer encoder trained using ST-MEM masked reconstruction for advanced representation learning.

✔ Optional: Hybrid FM + ViT Alignment
A REPA-style cosine alignment loss linking 1D FM embeddings with 2D ViT image embeddings for robustness.
This structure enables a scalable, research-grade pipeline suitable for academic evaluation and future clinical studies.

---

# 📊 2. Supported Datasets

ChagaSight supports three widely used 12-lead ECG datasets in modern ECG AI research:

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
│ ├── raw/ # Original unmodified ECG datasets
│ │ ├── ptbxl/ # PTB-XL (100 Hz / 500 Hz WFDB files)
│ │ ├── code15/ # CODE-15% Brazil dataset (raw ECGs)
│ │ └── sami_trop/ # SaMi-Trop serology-confirmed Chagas dataset
│ │
│ ├── processed/
│ │ ├── 1d_signals/ # Cleaned, resampled ECG (1D numpy arrays)
│ │ │ # → Baseline-removed, resampled, normalized
│ │ └── 2d_images/ # 2D structured ECG images (3 × 24 × 2048)
│ │ # → Final input format for Vision Transformer
│ │
│ └── splits/ # Patient-level train/val/test splits (JSON)
│
├── src/ # All source code
│ ├── preprocessing/
│ │ ├── baseline_removal.py # High-pass / band-pass filtering
│ │ ├── resample.py # Resampling to 400 Hz + padding & trimming
│ │ ├── normalization.py # Per-lead z-score normalization utilities
│ │ ├── image_embedding.py # ECG → 2D image conversion (RA/LA/LL channels)
│ │ └── soft_labels.py # Soft-label generation for CODE-15% dataset
│ │
│ ├── foundation_model/ # OPTIONAL — For ECG Foundation Model (1D FM)
│ │ ├── vit_1d_encoder.py # 1D ViT backbone for ECG signals
│ │ ├── st_mem_pretraining.py # Masked self-supervised training (ST-MEM)
│ │ ├── aol_mixing.py # Aggregation of Layers (AoL) module
│ │ └── fm_feature_extractor.py # Extract FM embeddings for hybrid models
│ │
│ ├── image_model/ # MAIN MODEL — ViT image classifier
│ │ ├── vit_image_encoder.py # Vision Transformer backbone for ECG images
│ │ ├── projection_head.py # Linear head for classification
│ │ └── alignment_loss.py # Loss for hybrid FM + ViT alignment
│ │
│ ├── dataloaders/
│ │ ├── ptbxl_loader.py # PTB-XL dataloader (1D + 2D modes)
│ │ ├── code15_loader.py # CODE-15% loader with soft labels
│ │ ├── sami_loader.py # SaMi-Trop Chagas dataset loader
│ │ ├── fm_signal_dataset.py # Dataset for training 1D Foundation Model (FM)
│ │ └── image_dataset.py # Dataloader for ECG image-based ViT training
│ │
│ ├── training/
│ │ ├── train_image_model.py # MAIN TRAINER — Vision Transformer training
│ │ ├── train_fm.py # OPTIONAL — Training the ECG Foundation Model
│ │ ├── train_hybrid.py # OPTIONAL — FM + ViT hybrid alignment training
│ │ ├── augmentations_1d.py # 1D ECG augmentations for FM
│ │ └── augmentations_2d.py # 2D ECG image augmentations for ViT
│ │
│ └── evaluation/
│ ├── metrics.py # AUROC, AUPRC, F1, calibration metrics
│ ├── explainability.py # Grad-CAM for images + FM attention maps
│ └── challenge_metric.py # Top-K TPR scoring utilities
│
├── scripts/ # Executable scripts for full pipeline
│ ├── preprocess_ptbxl.py # Preprocess PTB-XL (Stage 1: 1D)
│ ├── preprocess_code15.py # Preprocess CODE-15% (Stage 1)
│ ├── preprocess_samitrop.py # Preprocess SaMi-Trop (Stage 1)
│ ├── build_images.py # Stage 2: ECG → 2D image creation
│ ├── create_splits.py # Build train/val/test splits
│ ├── train_vit_image.sh # Shell script: Train ViT model
│ ├── train_fm.sh # Shell script: Train FM model (optional)
│ └── train_hybrid.sh # Shell script: Hybrid alignment training
│
├── notebooks/ # Development + documentation notebooks
│ ├── 01_preprocessing_1d.ipynb # Preprocess ECG into 1D format
│ ├── 02_image_embedding.ipynb # Convert 1D → 2D images
│ ├── 03_fm_pretraining.ipynb # OPTIONAL — FM masked training
│ ├── 04_cross_validation.ipynb # Model validation experiments
│ ├── 05_hybrid_alignment_training.ipynb
│ └── 06_evaluation.ipynb # Visualisations + performance metrics
│
├── docs/
│ ├── methodology/ # Dissertation-ready documentation
│ │ ├── fm_architecture.md # 1D FM architecture explanation
│ │ ├── image_embedding_diagram.png # 2D ECG image pipeline visualisation
│ │ └── augmentation_strategy.md # Full augmentation design
│ │
│ ├── figures/ # Figures for thesis/report
│ ├── reports/ # Auto-generated experiment summaries
│ └── diagrams/ # Flowcharts, system diagrams, etc.
│
├── experiments/ # Saved experimental runs
│ ├── image_baseline/ # ViT image model results
│ ├── fm_pretraining/ # FM pretraining logs
│ └── hybrid_alignment/ # Hybrid model experiments
│
├── results/ # Outputs (excluded from Git)
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

# 🔧 4. Preprocessing Pipeline

ChagaSight adopts a **two-stage preprocessing strategy** inspired by recent ECG research.

## **Stage 1 — 1D Signal Preprocessing**

- Baseline drift removal
- Resampling to a unified frequency
- Padding/trimming to fixed duration (10s)
- Per-lead z-score normalization
- Saving as .npy 1D arrays

Output directory:
data/processed/1d_signals/

## **Stage 2 — ECG → Structured 2D Image Conversion**

Using physiologically meaningful RA/LA/LL contour mapping:

- Construct 3 channels representing RA, LA, LL contours
- Subtract reference lead (augmented limb lead)
- Clip amplitudes to [-3, 3]
- Scale to pixel range [0–255]
- Resize to 3 × 24 × 2048

Output directory:
data/processed/2d_images/

This format is optimized for Vision Transformers.

---

# 🧠 5. Model Components

## **A. Vision Transformer (MAIN MODEL)**

- Input: structured ECG images
- Patch embeddings adapted for rectangular biomedical images
- Output: disease probability + latent embeddings

## This is the primary deliverable of the final-year project.

## **B. ECG Foundation Model (OPTIONAL EXTENSION)**

A 1D Vision Transformer (12 layers) trained using:

- ST-MEM masked self-supervised learning
- Patch size = 50
- Aggregation of Layers (AoL)

Provides robust signal-level ECG embeddings, especially useful in low-label or multi-dataset setups.

---

## **C. Hybrid FM + ViT Alignment Model (Optional Advanced Model)**

A REPA-inspired loss encourages ViT and FM feature alignment:

L_total = L_classification + λ · cosine_similarity(FM_features, ViT_features)

## Used for robustness and dataset-shift resistance.

# 🧪 6. Training Workflow

### **1. Preprocess 1D ECG Signals**

notebooks/01_preprocessing_1d.ipynb

### **2. Convert 1D Signals to 2D Images**

notebooks/02_image_embedding.ipynb

### **3. Train Vision Transformer (Main Model)**

python src/training/train_image_model.py

### **4. Evaluate ViT Model**

    AUROC
    AUPRC
    F1
    Calibration
    Grad-CAM

### **5. (Optional) Train ECG Foundation Model**

python src/training/train_fm.py

### **6. (Optional) Train Hybrid Model**

python src/training/train_hybrid.py

---

# 📈 7. Evaluation Metrics

    AUROC
    AUPRC
    Accuracy & F1
    Top-K screening sensitivity
    Calibration curves
    Confusion matrices
    Grad-CAM (image & signal attention)

---

# 🔍 8. Key Contributions of This Pipeline

**✔ Physiologically Structured 2D ECG Image Pipeline**
A high-fidelity image representation built on limb-lead reference mapping.

**✔ Vision Transformer Baseline Model (Main Output)**
The primary focus of the dissertation.

**✔ Optional 1D Foundation Model (Advanced)**
Implements contemporary masked ECG self-supervision.

**✔ Optional Hybrid Feature Alignment**
Bridges image-based and signal-based features.

**✔ Unified Multi-Dataset Workflow**
A single preprocessing and training pipeline across PTB-XL, CODE-15%, SaMi-Trop.

---

# 🚀 9. Roadmap

[ ] Phase 1 — 1D ECG Preprocessing (Required)

    Clean and normalize all datasets
    Resample, trim, pad
    Save as .npy 1D signals

[ ] Phase 2 — Structured 2D Image Generation (Required)

    Produce 3-channel structured images
    Validate RA/LA/LL mapping
    Save to processed/2d_images/

[ ] Phase 3 — Train Vision Transformer (Required)

    Train ViT classifier
    Evaluate AUROC / AUPRC / F1

[ ] Phase 4 — ECG Foundation Model (Optional)

    ST-MEM masked pretraining
    Extract FM embeddings

[ ] Phase 5 — Hybrid Alignment (Optional)

    Train joint FM + ViT model
    Apply alignment loss

[ ] Phase 6 — Evaluation & Explainability (Required)

    Grad-CAM
    Calibration curves
    Dataset-shift analysis

[ ] Phase 7 — Dissertation Deliverables (Required)

    Write methodology chapter
    Include all diagrams
    Present results, comparison, limitations

---

# 📬 Contact

See the `docs/` directory for methodology, architecture notes, and experiment logs.
