ChagaSight — Detecting Chagas Disease from 12-Lead ECGs Using Physiologically Structured 2D Images and Vision Transformer
A Final-Year Research Project inspired by the George B. Moody PhysioNet Challenge 2025
Overview
ChagaSight is a modular, reproducible deep learning framework for detecting Chagas disease from standard 12-lead ECG recordings. The project innovatively combines the strengths of two top-performing approaches from the 2025 PhysioNet Challenge:

Primary Innovation: Physiologically structured 2D contour images (Kim et al., 2025) fed into a Vision Transformer architecture inspired by Van Santvliet et al. (2025), the Challenge winners.
Core Idea: Transform raw ECG signals into spatial-preserving 2D embeddings (RA/LA/LL contours) as input to a powerful ECG-domain ViT. This achieves superior representation of both temporal dynamics and inter-lead relationships, potentially outperforming the winning 1D ViT foundation model.

This hybrid approach is original — neither paper combined structured 2D images with a ViT designed for ECG sequences. The framework is built on top of the official PhysioNet Challenge 2025 code base for full reproducibility and submission compatibility.
Project Objectives

Reproduce and extend Kim et al.'s 2D contour embedding with preserved spatial information.
Apply the resulting 2D images to a ViT architecture inspired by Van Santvliet et al.'s ECG-pretrained foundation model.
Integrate official PhysioNet Challenge data processing and evaluation tools for standardization.
Achieve strong performance on the Challenge metric (fraction of true positives in top 5% ranked cohort), AUROC, AUPRC, etc.
Provide explainability via Grad-CAM visualizations and ablation studies (e.g., 1D vs. 2D input).
Validate the full pipeline through extensive testing and comparisons.

Data Sources (PhysioNet Challenge 2025 Training Set)

PTB-XL (21,799 records, 500 Hz, Europe): All Chagas-negative (strong geographic labels).
SaMi-Trop (1,631 records, 400 Hz, Brazil): All Chagas-positive (strong, serologically validated).
CODE-15% (~345,000 records; current subset ~34k, 400 Hz, Brazil): Mixed weak self-reported labels (~2% positive), with soft labeling applied during training to handle noise.

All raw data is converted to the official WFDB format using PhysioNet-provided scripts (prepare*ptbxl_data.py, prepare_samitrop_data.py, prepare_code15_data.py), which embed Chagas labels in .hea headers.
Pipeline Overview
textRaw Data (HDF5 / WFDB Files)
↓
Official PhysioNet Preprocessing (prepare*\*.py) → WFDB (.hea + .dat) with embedded labels
↓
Custom Preprocessing (build_2d_images_wfdb.py):
• Load via wfdb.rdsamp
• Baseline wander removal (bandpass filter 0.5-45 Hz)
• Resample to 500 Hz (polyphase, anti-aliased)
• Per-lead z-score normalization
• Clip to [-3, 3] std dev
↓
2D Contour Image Embedding (image_embedding.py, from Kim et al., 2025)
• RA/LA/LL referenced channels
• Output: 3 × 24 × 2048 float32 image (preserves spatial lead relationships)
↓
Vision Transformer Classification (train_vit.ipynb, inspired by Van Santvliet et al., 2025)
• 2D patch embedding on contour image (resized to 224x224)
• ViT-Base (12 transformer blocks + [CLS] token)
• Soft labels applied via soft_labels.py (strong for PTB-XL/SaMi-Trop, soft 0.2/0.8 for CODE-15%)
• Cosine alignment loss to 1D FM (future extension)
• Fine-tuned for binary Chagas probability
↓
Evaluation (evaluate_model.py from PhysioNet)
• Official Challenge score (top 5% prioritization)
• AUROC, AUPRC, F1, Accuracy
• Grad-CAM visualizations for explainability
Project Structure
textChagaSight/
├── data/
│ ├── raw/ # Original unchanged datasets
│ │ ├── ptbxl/ # PTB-XL raw WFDB files
│ │ ├── sami_trop/ # SaMi-Trop HDF5 and CSV
│ │ └── code15/ # CODE-15% HDF5 shards and exams.csv + code15_chagas_labels.csv
│ └── official_wfdb/ # Official WFDB output from PhysioNet scripts
│ ├── ptbxl/ # PTB-XL .hea/.dat with embedded labels
│ ├── sami_trop/ # SaMi-Trop .hea/.dat
│ └── code15/ # CODE-15% .hea/.dat
├── data/processed/
│ └── 2d_images/ # Generated 2D contour images (3×24×2048 .npy)
│ ├── ptbxl/
│ ├── sami_trop/
│ └── code15/
├── external/
│ └── official_2025/ # Cloned PhysioNet Challenge example repo[](https://github.com/physionetchallenges/python-example-2025)
│ ├── prepare_ptbxl_data.py # PhysioNet script: Converts PTB-XL to WFDB with labels
│ ├── prepare_samitrop_data.py # PhysioNet script: Converts SaMi-Trop to WFDB with labels
│ ├── prepare_code15_data.py # PhysioNet script: Converts CODE-15% to WFDB with labels
│ ├── evaluate_model.py # PhysioNet script: Computes Challenge score, AUROC, etc.
│ ├── helper_code.py # PhysioNet helper: Utility functions (e.g., load signals)
│ ├── team_code.py # PhysioNet template: For custom training/inference (adapt for ViT)
│ ├── train_model.py # PhysioNet wrapper: Calls team_code.train_model
│ ├── run_model.py # PhysioNet wrapper: Calls team_code.run_model
│ ├── Dockerfile # For PhysioNet submission
│ ├── LICENSE # BSD 2-Clause
│ ├── README.md # PhysioNet example README
│ └── requirements.txt # PhysioNet dependencies
├── src/
│ └── preprocessing/
│ ├── baseline_removal.py # Bandpass/highpass filters for wander removal
│ ├── resample.py # Polyphase resampling to 500/100 Hz
│ ├── normalization.py # Z-score normalization per lead
│ ├── image_embedding.py # RA/LA/LL contour mapping (Kim et al.)
│ └── soft_labels.py # Dataset-specific Chagas labels (strong/soft)
├── scripts/
│ ├── custom_old/ # Archived early custom preprocessing scripts
│ │ ├── preprocess_ptbxl.py
│ │ ├── preprocess_samitrop.py
│ │ ├── preprocess_samitrop_updated.py
│ │ ├── preprocess_code15.py
│ │ └── preprocess_code15_corrected.py
│ └── build_2d_images_wfdb.py # Current: Full pipeline from WFDB to 2D images (parallelized)
├── tests/
│ ├── old/ # Archived early validation scripts
│ │ ├── test_baseline.py # Tests baseline removal
│ │ ├── test_resample.py # Tests resampling
│ │ ├── test_preprocessing_pipeline.py # Old dual-FS pipeline test
│ │ ├── test_code15_raw.py # Old CODE-15% raw check
│ │ ├── test_samitrop_preprocessing.py # Old SaMi-Trop test
│ │ ├── analyze_samitrop_signals.py # Old SaMi-Trop analysis
│ │ ├── check_raw_data.py # Old raw data check
│ │ └── validate_single_ecg.py # Old single ECG validation
│ ├── test_wfdb_data.py # New: Checks WFDB data presence/integrity
│ ├── test_baseline.py # (Kept) Component test for baseline
│ ├── test_resample.py # (Kept) Component test for resample
│ ├── test_image_embedding.py # New: Tests contour embedding
│ ├── test_soft_labels.py # New: Tests soft label assignment
│ ├── test_full_pipeline.py # New: Tests full WFDB → 2D pipeline
│ ├── check_processed_data.py # New: Checks processed images (counts, samples)
│ ├── verify_official_data.py # (Kept) WFDB verification
│ └── check_data.py # (Kept) Legacy integrity check
├── notebooks/
│ └── train_vit.ipynb # Interactive ViT training, visualization, evaluation
├── outputs/ # Model weights, metrics, plots
├── configs/
│ └── vit_config.yaml # ViT hyperparameters
├── README.md # This file
└── requirements.txt # Dependencies (includes PhysioNet + custom for ViT: torch, transformers)
Current Status & Roadmap

PhaseStatusDescription

Data Acquisition | Complete | Raw datasets + official Chagas labels downloaded.
Official WFDB Conversion | Complete | Used PhysioNet scripts (prepare\_\*.py) for standardization.
2D Contour Image Generation | Complete | From WFDB → custom pipeline → 3×24×2048 images (build_2d_images_wfdb.py).
Hybrid ViT Training | In Progress | 2D images → Van Santvliet-inspired ViT (train_vit.ipynb).
Evaluation & Explainability | Planned | Challenge score (evaluate_model.py), Grad-CAM, ablations.
Full Dataset Scaling | Post-Demo | Complete CODE-15% processing.

Reproducibility
All dependencies are defined in requirements.txt (includes PhysioNet's wfdb, numpy, etc., plus custom for ViT: torch, transformers).
Activate virtual environment:
text.venv\Scripts\activate # Windows
Install dependencies:
textpip install -r requirements.txt
Run WFDB conversion (PhysioNet scripts in external/official_2025/):
textpython external/official_2025/prepare_samitrop_data.py -i data/raw/sami_trop/exams.hdf5 -d data/raw/sami_trop/exams.csv -o data/official_wfdb/sami_trop

# Similar for PTB-XL and CODE-15%

Generate 2D images:
textpython -m scripts.build_2d_images_wfdb --subset 1.0
Train ViT:

Open notebooks/train_vit.ipynb in Jupyter.

Run tests:
textpython -m unittest discover tests
Outputs (plots, summaries) in tests/verification_outputs_new/ and outputs/.
Tests & Verification
Use tests/ folder for validation:

test_wfdb_data.py: WFDB data checks (counts, labels, plots).
test_baseline.py: Baseline removal test.
test_resample.py: Resampling test.
test_image_embedding.py: Contour embedding test.
test_soft_labels.py: Soft label assignment test.
test_full_pipeline.py: Full WFDB → 2D test.
check_processed_data.py: Processed images checks (counts, samples, plots).
verify_official_data.py & check_data.py: Legacy WFDB/integrity checks.

All tests output to tests/verification_outputs_new/ for comparisons (e.g., raw vs processed plots).
References

Kim et al. (2025). "Embedding ECG Signals into 2D Image with Preserved Spatial Information for Chagas Disease Classification"
Van Santvliet et al. (2025). "Detecting Chagas Disease Using a Vision Transformer–based ECG Foundation Model"
George B. Moody PhysioNet Challenge 2025: https://physionetchallenges.org/2025/
Official example code: https://github.com/physionetchallenges/python-example-2025

License
Custom code © 2025 [Your Name]. Official PhysioNet components under BSD 2-Clause (as per LICENSE in external/official_2025/).
This README provides a complete overview, setup instructions, and detailed structure. Save it as README.md. For the full script updates (RAM-safe, per-dataset run, batching), use the previous response's code with N_JOBS=2 for CODE-15%. Run per-dataset to avoid crashes:
Bashpython -m scripts.build_2d_images_wfdb --subset 1.0 # For SaMi-Trop/PTB-XL

# Then change DATASETS = ["code15"] and run again
