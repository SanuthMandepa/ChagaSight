📌 1. Project Overview

ChagaSight is a modular deep-learning research framework designed to investigate whether Vision Transformers, when trained on physiologically meaningful ECG image representations, can effectively detect Chagas disease across multiple ECG datasets.

The project places primary emphasis on:

Transforming raw 12-lead ECG signals into structured 2D images and preparing a Vision Transformer classifier for disease detection.

In addition, the architecture is designed to support future extensions, including 1D ECG foundation models and hybrid alignment strategies. These extensions are explicitly marked as optional and non-essential for the core dissertation contribution.

📚 Research Inspiration

The design of ChagaSight is informed by two contemporary ECG-AI research directions:

Physiologically Structured 2D ECG Image Embedding (2025)
— converting ECG signals into spatially structured image representations.

Vision Transformer–Based ECG Foundation Models (2025)
— applying transformer architectures to ECG signals using self-supervised learning.

These works inspire architectural choices, but the implementation focuses on a practical, verifiable, and reproducible pipeline suitable for academic evaluation.

📊 2. Supported Datasets

ChagaSight currently supports three widely used 12-lead ECG datasets in modern ECG research.

Dataset Description Sample Rate Chagas Label
PTB-XL European clinical ECG dataset (~21k recordings) 100 / 500 Hz All negative (0)
CODE-15% Brazilian population ECG cohort 400 Hz Soft labels: 0.2 / 0.8
SaMi-Trop Serology-confirmed Chagas cohort 400 Hz All positive (1)
✔ Label Policy

The following labeling strategy is adopted in line with state-of-the-art ECG-AI research:

Dataset Confidence Label Assignment
PTB-XL Very strong negative 0
SaMi-Trop Very strong positive 1
CODE-15% Weak / self-reported Soft labels (0.2 / 0.8)
📁 3. Project Folder Structure

The structure below represents the logical system architecture.
Large datasets and virtual-environment internals are intentionally excluded.

ChagaSight/
├── .git/
├── .venv/ # Python virtual environment (ignored in version control)
│
├── data/
│ ├── raw/ # Original unmodified ECG datasets
│ │ ├── ptbxl/
│ │ ├── code15/
│ │ └── sami_trop/
│ │
│ ├── processed/
│ │ ├── 1d_signals_100hz/ # FM-compatible 1D ECG signals
│ │ ├── 1d_signals_500hz/ # High-resolution signals for image embedding
│ │ ├── 2d_images/ # Structured ECG image representations
│ │ └── metadata/ # Processed dataset metadata (CSV)
│ │
│ └── splits/ # Train / validation / test splits
│
├── notebooks/ # Exploratory analysis and development notebooks
│
├── scripts/ # Dataset-level preprocessing scripts
│ ├── preprocess_ptbxl.py
│ ├── preprocess_code15.py
│ ├── preprocess_code15_corrected.py
│ ├── preprocess_samitrop.py
│ ├── preprocess_samitrop_updated.py
│ ├── build_images.py
│ └── validate_single_ecg.py
│
├── src/
│ ├── preprocessing/ # Core ECG signal processing modules
│ │ ├── baseline_removal.py
│ │ ├── resample.py
│ │ ├── normalization.py
│ │ ├── image_embedding.py
│ │ └── soft_labels.py
│ │
│ ├── dataloaders/
│ │ └── ptbxl_loader.py
│ │
│ ├── image_model/ # Image-based model components (planned)
│ ├── foundation_model/ # Optional FM architecture (design stage)
│ ├── training/ # Training orchestration (planned)
│ └── evaluation/ # Evaluation utilities (planned)
│
├── tests/ # Scientific verification & validation suite
│ ├── test_baseline.py
│ ├── test_resample.py
│ ├── test_preprocessing_pipeline.py
│ ├── test_samitrop_preprocessing.py
│ ├── test_code15_raw.py
│ ├── analyze_samitrop_signals.py
│ ├── check_raw_data.py
│ ├── validate_single_ecg.py
│ └── verification_outputs/
│
├── requirements.txt
├── .gitignore
└── README.md

🔧 4. ECG Preprocessing Pipeline

ChagaSight adopts a two-stage preprocessing strategy, fully implemented and verified through scripts and tests.

Stage 1 — 1D ECG Signal Preprocessing (Implemented)

The following steps are applied in a dataset-aware manner:

Baseline drift removal

Resampling to a unified frequency

Padding or trimming to a fixed duration (10 seconds)

Per-lead z-score normalization

Saving processed signals as .npy arrays

Dataset-specific baseline handling:

PTB-XL → band-pass filtering (0.5–45 Hz)

SaMi-Trop → moving-average baseline removal

CODE-15% → no baseline removal (pre-filtered data)

Output directories:

data/processed/1d_signals_500hz/
data/processed/1d_signals_100hz/

Stage 2 — ECG → Structured 2D Image Conversion (Implemented)

Each ECG is converted into a physiologically structured image using limb-lead reference mapping:

Construction of three channels representing RA, LA, and LL contours

Subtraction of augmented limb-lead reference

Amplitude clipping to [-3, 3]

Linear scaling to [0, 255]

Resizing to 3 × 24 × 2048

Output directory:

data/processed/2d_images/

This representation is optimised for Vision Transformer input.

🧠 5. Model Scope
A. Vision Transformer (Primary Dissertation Model)

Input: structured 2D ECG images

Architecture: Vision Transformer adapted for rectangular biomedical images

Output: probability of Chagas disease

Status:
Model training and evaluation constitute the next planned project phase.

B. ECG Foundation Model (Optional Research Extension)

An optional architectural extension is designed for a 1D ECG Foundation Model, inspired by masked self-supervised learning (e.g. ST-MEM).

Status:
Design exploration only.
Not required for core dissertation results.

C. Hybrid FM + ViT Alignment (Future Research Direction)

A proposed hybrid model aims to align 1D FM embeddings with 2D ViT embeddings using a cosine-similarity–based objective.

Status:
Conceptual design only.

🧪 6. Training & Validation Workflow
Implemented

1D ECG preprocessing (scripts/preprocess\_\*.py)

ECG → 2D image generation (scripts/build_images.py)

Pipeline validation (tests/validate_single_ecg.py)

Validation includes:

Signal integrity checks

Frequency-domain inspection

Lead-wise consistency analysis

1D ↔ 2D correspondence

Planned

Vision Transformer training

Model evaluation and explainability

📈 7. Evaluation Metrics (Planned)

AUROC

AUPRC

Accuracy

F1-score

Calibration curves

Confusion matrices

Explainability techniques (e.g. Grad-CAM) are planned for future evaluation.

🔍 8. Key Contributions
✔ Implemented

Physiologically structured 2D ECG image pipeline

Unified multi-dataset preprocessing

Comprehensive verification and validation suite

◻ Planned

Vision Transformer classifier training

Performance evaluation and explainability

Optional foundation-model experiments

🚀 9. Roadmap

1D ECG preprocessing

Structured 2D image generation

Validation & verification

Vision Transformer training

Model evaluation & explainability

Dissertation writing & submission

📦 Reproducibility

All dependencies are defined in:

requirements.txt

A Python virtual environment (.venv/) is used locally and excluded from version control.

📬 Contact

Refer to the tests/ and notebooks/ directories for validation outputs, diagnostic plots, and exploratory analyses supporting the methodology.
