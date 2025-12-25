# ChagaSight — Detecting Chagas Disease from 12-Lead ECGs Using Physiologically Structured 2D Images and Vision Transformer

A Final-Year Research Project inspired by the George B. Moody PhysioNet Challenge 2025

## Overview

ChagaSight is a modular, reproducible deep learning framework for detecting Chagas disease from standard 12-lead ECG recordings. The project innovatively combines the strengths of two top-performing approaches from the 2025 PhysioNet Challenge:

**Primary Innovation**: Physiologically structured 2D contour images (Kim et al., 2025) fed into a Vision Transformer architecture inspired by Van Santvliet et al. (2025), the Challenge winners.

**Core Idea**: Transform raw ECG signals into spatial-preserving 2D embeddings (RA/LA/LL contours) as input to a powerful ViT. This achieves superior representation of both temporal dynamics and inter-lead relationships, potentially outperforming pure 1D approaches.

This hybrid approach is original — neither reference paper combined structured 2D images with a ViT designed for ECG sequences. The framework is built on top of the official PhysioNet Challenge 2025 code base for full reproducibility and submission compatibility.

## Project Objectives

- Reproduce and extend Kim et al.'s 2D contour embedding with preserved spatial information.
- Apply the resulting 2D images to a ViT architecture inspired by Van Santvliet et al.'s ECG-pretrained foundation model.
- Integrate official PhysioNet Challenge data processing and evaluation tools for standardization.
- Achieve strong performance on the Challenge metric (fraction of true positives in top 5% ranked cohort), AUROC, AUPRC, etc.
- Provide explainability via Grad-CAM visualizations and ablation studies.
- Validate the full pipeline through extensive testing and comparisons.

## Data Sources (PhysioNet Challenge 2025 Training Set)

| Dataset   | Records | Sampling Rate | Origin | Chagas Labels                                                  |
| --------- | ------- | ------------- | ------ | -------------------------------------------------------------- |
| PTB-XL    | 21,799  | 500 Hz        | Europe | All negative (strong geographic label)                         |
| SaMi-Trop | 1,631   | 400 Hz        | Brazil | All positive (serologically validated)                         |
| CODE-15%  | ~39,798 | 400 Hz        | Brazil | Mixed weak self-reported (~2% positive), soft labeling applied |

All raw data is converted to the official WFDB format using PhysioNet-provided scripts, which embed Chagas labels in `.hea` headers.

## Pipeline Overview

Raw Data (HDF5 / original WFDB)
↓
Official PhysioNet Conversion (prepare\_.py) → WFDB (.hea + .dat) with embedded labels
↓
Main Processing Script (build_2d_images_wfdb.py):
• Load via wfdb.rdsamp
• Dataset-specific baseline removal
• Resample to 500 Hz (for images) and 100 Hz (for FM)
• Per-lead z-score normalization + clipping [-3, 3] (500 Hz only)
• Generate 2D RA/LA/LL contour images (Kim et al.)
↓
Outputs:
• data/processed/2d_images//ID_img.npy → Primary input for ViT training
• data/processed/1d_signals_100hz/\*/ID.npy → For future 1D FM pretraining / hybrid alignment
↓
Vision Transformer Training (train_vit.ipynb)
• Input: 2D contour images
• ViT-Base architecture (inspired by Van Santvliet et al.)
• Soft labels + BCE loss
• Future: Cosine alignment with 1D FM embeddings
↓
Evaluation (evaluate_model.py from PhysioNet)
• Official Challenge score, AUROC, AUPRC
• Grad-CAM visualizations for explainability

## Project Structure

ChagaSight/
├── data/
│ ├── raw/ # Original unchanged datasets (do not modify)
│ │ ├── ptbxl/
│ │ ├── sami_trop/
│ │ └── code15/
│ └── official_wfdb/ # Converted WFDB files (from PhysioNet scripts)
│ ├── ptbxl/
│ ├── sami_trop/
│ └── code15/
├── data/processed/
│ ├── 2d_images/ # 2D contour images (3×24×2048 .npy) — Primary training data
│ │ ├── ptbxl/
│ │ ├── sami_trop/
│ │ └── code15/
│ └── 1d_signals_100hz/ # Raw 100 Hz signals — For future FM/hybrid work
│ ├── ptbxl/
│ ├── sami_trop/
│ └── code15/
├── external/
│ └── official_2025/ # PhysioNet Challenge 2025 example code (BSD 2-Clause)
│ ├── prepare_ptbxl_data.py
│ ├── prepare_samitrop_data.py
│ ├── prepare_code15_data.py
│ ├── evaluate_model.py
│ ├── helper_code.py
│ ├── team_code.py
│ ├── train_model.py
│ ├── run_model.py
│ ├── Dockerfile
│ └── requirements.txt
├── src/
│ └── preprocessing/
│ ├── baseline_removal.py # Bandpass / moving-average filters
│ ├── resample.py # Polyphase resampling
│ ├── normalization.py # Per-lead z-score
│ ├── image_embedding.py # RA/LA/LL contour mapping (Kim et al.)
│ └── soft_labels.py # Strong/soft label assignment
├── scripts/
│ └── build_2d_images_wfdb.py # Main script: WFDB → 2D images + 100 Hz signals (parallelized)
├── tests/
│ ├── test_data_integrity.py # Checks WFDB vs processed counts/shapes
│ ├── test_full_pipeline.py # Simulates pipeline on sample records
│ ├── test_image_embedding.py # Tests contour embedding
│ ├── test_soft_labels.py # Tests label assignment
│ ├── test_preprocessing.py # Component tests (baseline, resample, normalize)
│ ├── test_raw_data.py # Raw data loading checks
│ ├── test_wfdb_data.py # WFDB loading checks
│ └── validate_single_ecg.py # Detailed visual validation for single records
├── notebooks/
│ └── train_vit.ipynb # ViT training, evaluation, visualizations
├── outputs/ # Model weights, logs, plots
├── configs/
│ └── vit_config.yaml # Training hyperparameters
├── README.md # This file
└── requirements.txt # Project dependencies (includes PhysioNet + torch, transformers)
text

## Setup & Usage

1. **Create virtual environment**:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   # source .venv/bin/activate  # Linux/macOS

   ```

2. **Install dependencies**
   pip install -r requirements.txt

3. **Convert raw data to official WFDB (run once)**
   python external/official_2025/prepare_ptbxl_data.py -i data/raw/ptbxl -d data/raw/ptbxl/ptbxl_database.csv -o data/official_wfdb/ptbxl
   python external/official_2025/prepare_samitrop_data.py -i data/raw/sami_trop/exams.hdf5 -d data/raw/sami_trop/exams.csv -o data/official_wfdb/sami_trop
   python external/official_2025/prepare_code15_data.py -i data/raw/code15 -o data/official_wfdb/code15

4. **Generate 2D images and 100 Hz signals (main processing step)**

# Full run (recommended: run per-dataset if RAM-limited)

python -m scripts.build_2d_images_wfdb --subset 1.0

# Or run one dataset at a time

python -m scripts.build_2d_images_wfdb --dataset ptbxl
python -m scripts.build_2d_images_wfdb --dataset sami_trop
python -m scripts.build_2d_images_wfdb --dataset code15

5. **Run tests:**
   python -m unittest discover tests

6. **Train the ViT:**
   Open notebooks/train_vit.ipynb in Jupyter and run all cells.

7.Validate single record (visual debugging):
python -m tests.validate_single_ecg --dataset ptbxl --id 1
python -m tests.validate_single_ecg --all

Current Status (December 25, 2025)

Phase,Status,Notes
Data Acquisition,Complete,All raw datasets downloaded
WFDB Conversion,Complete,Using official PhysioNet scripts
2D Image Generation,Complete,"Clean, validated output"
Pipeline Validation,Complete,All tests passing
ViT Baseline Training,In Progress,Ready to start
Hybrid FM–ViT Alignment,Planned,Next major step
Full Evaluation,Planned,Challenge metrics + explainability

Reproducibility & Submission

All code is self-contained and uses seeded random states where applicable.
Compatible with PhysioNet submission format (team_code.py, train_model.py, run_model.py).
Dockerfile included for containerized submission.

References

Kim et al. (2025). "Embedding ECG Signals into 2D Image with Preserved Spatial Information for Chagas Disease Classification"
Van Santvliet et al. (2025). "Detecting Chagas Disease Using a Vision Transformer–based ECG Foundation Model"
George B. Moody PhysioNet Challenge 2025: https://physionetchallenges.org/2025/
Official example code: https://github.com/physionetchallenges/python-example-2025

License
Custom code © 2025 [Your Name].
Official PhysioNet components under BSD 2-Clause (see external/official_2025/LICENSE).
