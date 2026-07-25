# ChagaSight

**A Self-Supervised Dual-Pathway Vision Transformer Ensemble for Chagas Disease Detection from 12-Lead ECG**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](requirements_deploy.txt)
[![React 18](https://img.shields.io/badge/frontend-React%2018-61DAFB.svg)](frontend/package.json)

University final year project (FYP) — an end-to-end machine learning system that screens standard 12-lead ECG recordings for Chagas disease, from raw signal preprocessing through to a deployed web application.

**Live demo:** [chagasight.vercel.app](https://chagasight.vercel.app/) · **API:** [sanuthmandepa-chagasight-api.hf.space](https://sanuthmandepa-chagasight-api.hf.space)

---

## Overview

Chagas disease is a parasitic infection endemic to Latin America that can cause silent, progressive cardiac damage over decades. Standard 12-lead ECG is cheap and widely available, but subtle Chagas-related changes (e.g. right bundle branch block, axis deviation) are easy to miss on manual reading, and confirmatory serological testing is not scalable for population-level screening.

This project analyses, designs, develops, and evaluates a self-supervised dual-pathway Vision Transformer ensemble for automated Chagas detection from ECG signals, addressing two core challenges in the underlying data: severe class imbalance (~2.2% positive prevalence) and heterogeneity across multiple ECG data sources.

## Architecture

ChagaSight represents each ECG recording in two complementary forms and fuses them:

```
1D signal (12, 1000) @ 100Hz + demographics ──► 1D ViT (ST-MEM pretrained)  ──► 768-d
                                                                                    │
2D contour image (3, 24, 2048) @ 500Hz ────────► 2D ViT (MAE pretrained)  ──► 768-d │
                                                          │                        │
                                                    REPA alignment                 │
                                                          ▼                        ▼
                                                   [ Aligned 2D  |  1D FM features ] → 1536-d
                                                                  │
                                                          3-layer MLP classifier
                                                                  │
                                                            Chagas probability
```

| Component | Description |
|---|---|
| **1D pathway** ([`src/models/vit_1d_fm.py`](src/models/vit_1d_fm.py)) | Per-lead Conv1D patch embedding + transformer, pretrained with ST-MEM (spatial-temporal masked ECG modelling). Age/sex injected via FiLM modulation. |
| **2D pathway** ([`src/models/vit_2d.py`](src/models/vit_2d.py)) | Vision Transformer over a 3-channel Wilson Central Terminal (WCT) re-referenced ECG image, pretrained with a Masked Autoencoder (MAE). |
| **Cross-modal alignment** ([`src/models/repa_alignment.py`](src/models/repa_alignment.py)) | REPA projection aligns 2D features into the 1D feature space via a cosine-similarity loss, so the 2D pathway learns representations consistent with the (stronger, pretrained) 1D pathway. |
| **Fusion** ([`src/models/hybrid_model.py`](src/models/hybrid_model.py)) | Aligned 2D + 1D features are concatenated and passed through an MLP classifier for the final binary prediction. |
| **Ensemble** | Five models are trained under 5-fold cross-validation (173,570,817 parameters each) and their predicted probabilities averaged at inference time. |

Full methodology — preprocessing, self-supervised pretraining, the two-phase fine-tuning strategy, loss functions, and augmentations — is documented in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Results

Five-fold cross-validated ensemble, evaluated on pooled held-out predictions across all three datasets (N=386,981; 8,579 positive, 2.22% prevalence). Confidence intervals are bootstrapped over 1,000 resamples.

| Metric | Value | 95% CI |
|---|---|---|
| AUROC | 0.871 | 0.867 to 0.875 |
| AUPRC | 0.259 | 0.249 to 0.269 |

At the Youden's J operating point (τ = 0.706):

| Metric | Value |
|---|---|
| Sensitivity | 0.777 |
| Specificity | 0.801 |
| NPV | 0.994 |
| Precision (PPV) | 0.081 |
| MCC | 0.208 |
| Accuracy | 0.801 |
| Number needed to screen | 4.5 |

Precision is low by construction at 2.22% prevalence, and AUPRC should be read against the 0.022 random baseline (0.259 is roughly a 12x lift). The intended role is triage rather than diagnosis: a high NPV of 0.994 rules out the large negative majority, leaving confirmatory serology for the flagged minority.

Per-fold AUROC:

| Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Ensemble |
|---|---|---|---|---|---|
| 0.850 | 0.761 | 0.800 | 0.822 | 0.848 | **0.871** |

The ensemble scores above its strongest individual fold, indicating the folds make partly decorrelated errors. Per-dataset ROC is reported for CODE-15% alone (AUROC 0.864, AUPRC 0.215); PTB-XL contributes no positive cases and SaMi-Trop no negative cases within this evaluation, so a per-dataset ROC is undefined for both.

Raw per-record predictions and full breakdowns: [`Checkpoints/ensemble_predictions.csv`](Checkpoints/ensemble_predictions.csv), [`Checkpoints/per_fold_metrics.csv`](Checkpoints/per_fold_metrics.csv), [`Checkpoints/per_dataset_metrics.csv`](Checkpoints/per_dataset_metrics.csv).

## Tech Stack

- **Modelling:** Python 3.11, PyTorch, scikit-learn
- **Backend:** Flask, wfdb (ECG I/O), served via Docker
- **Frontend:** React 18, Vite, Tailwind CSS
- **Deployment:** Docker → Hugging Face Spaces (API), Vercel (frontend); model weights hosted on the Hugging Face Hub and downloaded at container start

## Datasets

Trained on three public PhysioNet-hosted ECG datasets (not redistributed in this repository — see each dataset's own access terms):

- [PTB-XL](https://physionet.org/content/ptb-xl/) — general 12-lead ECG database (Germany)
- [CODE-15%](https://zenodo.org/records/4916206) — large clinical ECG database (Brazil), weak/algorithmic labels
- [SaMi-Trop](https://physionet.org/content/samitrop-chagas/) — Chagas-enriched cohort (Brazil)

## Project Structure

```
ChagaSight/
├── app.py                     # Flask backend — inference API
├── Dockerfile                 # Backend container for HF Spaces
├── requirements_deploy.txt    # Backend runtime dependencies
├── PROJECT_PLAN.md            # Full methodology & design-decision log
├── src/
│   ├── preprocessing/         # Filtering, resampling, normalisation, WCT image construction, soft labels
│   ├── models/                # 1D ViT-FM, 2D ViT, REPA alignment, hybrid fusion model
│   └── training/               # Dataset, losses, metrics, trainer (2-phase fine-tuning)
├── scripts/                   # Data-build, split-creation, pretraining entry points
├── notebooks/                 # Training & evaluation notebooks
├── tests/                     # Unit / integration tests
├── frontend/                  # React + Vite + Tailwind web app
├── deploy/                    # HF Space README/config
└── external/official_2025/    # Vendored official PhysioNet/CinC 2025 Challenge starter code
```

`data/`, `models/*.pt`, `checkpoints/` and other large generated artifacts are excluded via `.gitignore`; model weights are fetched from the Hugging Face Hub at runtime instead of being committed.

## Getting Started

### Backend

```bash
pip install -r requirements_deploy.txt
python app.py
```

The server downloads model weights from the Hugging Face Hub on first run and listens on `http://127.0.0.1:5050`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` in `frontend/.env` to point at your backend if not running on the default local port.

### API

```bash
# Health check
curl https://sanuthmandepa-chagasight-api.hf.space/api/health

# Prediction — upload a WFDB record (.hea + .dat/.mat)
curl -X POST https://sanuthmandepa-chagasight-api.hf.space/api/predict \
  -F "files=@record.hea" \
  -F "files=@record.dat" \
  -F "model_type=hybrid"
```

### Training pipeline

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full stage-by-stage pipeline (preprocessing → self-supervised pretraining → two-phase fine-tuning → evaluation), and [`notebooks/`](notebooks/) for the executable pipeline notebooks.

## Acknowledgements

- [PhysioNet/Computing in Cardiology Challenge 2025](https://physionetchallenges.org/2025/) organisers, whose official starter code is vendored under [`external/official_2025/`](external/official_2025/) under its own license.
- ST-MEM pretraining approach following Van Santvliet et al. (2025).
- 2D WCT-image representation and REPA cross-modal alignment approach following Kim et al. (2025).

## License

This project is licensed under the [MIT License](LICENSE) — see the file for details. Code vendored under `external/official_2025/` retains its own original license.

## Author

Sanuth Mandepa — University final year project.
