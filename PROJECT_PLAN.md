# ChagaSight: Full Project Plan & Methodology

## Project Overview

**Goal:** Develop an AI system to detect Chagas disease from standard 12-lead ECG recordings, enabling scalable screening in resource-limited endemic regions.

**Evaluation Challenge:** PhysioNet/CinC 2025 Challenge — the primary metric is **TPR@5%**: "If a hospital can only afford to screen 5% of patients, what fraction of true Chagas cases does the model correctly flag?"

**Datasets:**
- PTB-XL (Germany, general ECG database, ~21k records)
- CODE-15% (Brazil, large clinical ECG database, ~345k records)
- SaMi-Trop (Brazil, Chagas-enriched cohort, ~1.6k records)

**Total: ~366,000 ECG recordings across 3 heterogeneous sources.**

**Class Imbalance:** ~2.24% positive Chagas prevalence in the combined dataset.

---

## Stage 1 — Data Preparation & Preprocessing

### 1.1 Raw ECG Loading

All recordings are in **WFDB format** (standard for PhysioNet datasets). Each file contains 12-lead ECG signals at varying sampling rates (100 Hz, 500 Hz) and varying lengths. Leads are reordered to the standard 12-lead anatomical arrangement using PhysioNet helper utilities. Metadata (age, sex, label) is extracted and normalised across the three differently-formatted datasets into a unified CSV.

### 1.2 Baseline Removal

**Method:** Butterworth bandpass filter, 0.5–40 Hz, order 4, applied via `scipy.signal.filtfilt` (zero-phase).

**Justification:**
- Sub-0.5 Hz content is baseline wander (respiration, patient movement) — clinically irrelevant and harmful for morphology-based features.
- Above 40 Hz is muscle noise (EMG artefact). Standard clinical ECG bandwidth is 0.05–150 Hz for diagnostic quality, but for automated analysis 0.5–40 Hz is the accepted range.
- Zero-phase filtering (`filtfilt`) ensures no phase distortion, preserving waveform timing (PR interval, QRS duration).

### 1.3 Depadding

SaMi-Trop and CODE-15 recordings sometimes contain trailing zero-padding. These are stripped before resampling to avoid introducing artificial signal boundaries.

### 1.4 Resampling

All recordings are resampled to two target frequencies:

| Target | Frequency | Length | Purpose |
|--------|-----------|--------|---------|
| 1D signal | 100 Hz | 1000 samples (10 sec) | 1D ViT input |
| 2D intermediate | 500 Hz | 2048 samples | 2D image construction |

**Justification for 100 Hz for 1D:**
- ECG diagnostically relevant bandwidth is ≤40 Hz. By Nyquist, 80 Hz is sufficient; 100 Hz provides headroom.
- Reduces sequence length (1000 tokens after patching), making it computationally tractable on a 6 GB GPU.
- Consistent with Van Santvliet et al. (top PhysioNet 2025 team), who use 100 Hz for the FM backbone.

**Justification for 500 Hz for 2D:**
- Higher temporal resolution preserves morphology detail for image rendering (fine QRS structure).
- 2048 samples maps cleanly to the patch grid without remainder.

### 1.5 Per-Lead Z-Score Normalisation

Applied per lead independently: `y = (x − μ) / (σ + ε)` where ε = 1e-8.

**For the 1D pathway:** no clipping. Preserves amplitude dynamics between leads (important for vector-based features like axis deviation).

**For the 2D image intermediate:** hard-clip at ±3σ before normalisation. Prevents extreme amplitude artefacts from dominating the visual representation; images encode relative morphology, not absolute amplitude.

**Justification:** Different leads have very different amplitude scales (e.g. V5 >> aVR). Without per-lead normalisation, small-amplitude leads would be swamped in the attention mechanism.

---

## Stage 2 — Dual Representation Construction

### 2.1 Why Two Representations?

ECG signals carry information at different scales and in different formats:

- **1D signals** are the raw temporal waveform — they carry precise timing information (RR intervals, PR/QT duration), morphological patterns (ST elevation/depression, T-wave inversions), and amplitude-based features (QRS voltage). These are naturally processed by sequence models.

- **2D images** allow the model to leverage spatial pattern recognition. Chagas disease produces diffuse fibrotic changes that manifest as complex multi-lead patterns. Representing all leads simultaneously as a 2D spatial map allows convolutional or vision-based models to identify cross-lead spatial co-occurrences that are non-trivial to encode in a purely sequential 1D model.

**Combined:** The hybrid model exploits complementary information. The 1D pathway captures fine temporal dynamics per lead; the 2D pathway captures spatial cross-lead patterns.

### 2.2 The 2D ECG Image Construction (WCT Re-referencing)

**Method:** Wilson's Central Terminal (WCT) re-referencing.

```
WCT = (V_RA + V_LA + V_LL) / 3
```

Three re-referenced views are constructed from the standard 12-lead arrangement, each referenced to a different limb electrode (RA, LA, LL). This produces 3 channels, each containing all 12 leads re-expressed from a different spatial reference point.

Each channel's 12 leads are stacked into rows, then duplicated to height 24 (to form a spatially coherent 2D grid). The result is **(3, 24, 2048)** — a 3-channel "image" where width is time and height encodes lead identity.

**Justification:**
- WCT is the standard clinical reference for unipolar chest leads. Re-referencing systematically generates distinct spatial perspectives on the electrical field, encoding physiologically meaningful variation across channels.
- The 3-channel format matches standard RGB image conventions, enabling direct use of pre-trained 2D Vision Transformers (which expect 3-channel inputs).
- This approach follows Kim et al. (2025), whose 2D ECG representations achieved strong performance in the same PhysioNet challenge.

Normalised to uint8 [0–255] by mapping the ±3σ-clipped signal linearly to [0, 255].

---

## Stage 3 — Dataset Splitting (5-Fold Stratified Cross-Validation)

**Method:** Stratified 5-fold split, stratified jointly by `(dataset, label_hard)`.

**Justification:**
- Joint stratification on dataset × label ensures every fold has proportional representation from all three data sources AND maintains class balance per fold.
- Without dataset stratification, a fold might accidentally contain mostly SaMi-Trop positives (an enriched cohort), inflating apparent performance.
- 5-fold CV provides 5 independent train/val splits, enabling ensemble averaging and reliable uncertainty estimation via fold-wise variance.

**Result:** ~16,626 validation samples per fold. Positive class: ~2.24% (~373 positives per fold).

---

## Stage 4 — Self-Supervised Pretraining

Before any Chagas-specific training, both ViT branches are pretrained on the full 366k-record dataset in a purely self-supervised (label-free) manner. This is critical given the extreme label scarcity (only ~2.24% of records are positive, and labelled data for Chagas is rare globally).

### 4.1 MAE Pretraining for 2D-ViT

**Method:** Masked Autoencoder (MAE; He et al., 2022).

A random subset of 2D ECG image patches (typically 75%) is masked, and the model is trained to reconstruct the original pixel values in the masked regions using only the visible patches. No labels are used.

**Justification:**
- Forces the 2D-ViT to learn meaningful, generalisable representations of ECG morphology — the only way to reconstruct masked regions is to understand the underlying structure of ECG waveforms.
- MAE has demonstrated strong transfer performance in medical imaging with limited labelled data.
- Pretraining on 366k unlabelled records gives the 2D pathway far more signal than the ~8k positive-labelled records available for supervised fine-tuning.

**Output:** Pretrained 2D-ViT weights saved as `mae_2d_pretrained.pt`.

### 4.2 ST-MEM Pretraining for 1D-ViT

**Method:** Spatial-Temporal Masked ECG Modelling (ST-MEM; Van Santvliet et al., 2025).

ST-MEM extends MAE to 1D multi-lead ECG. It jointly masks patches across both the temporal dimension (within-lead) and the spatial dimension (across leads), forcing the model to learn both within-lead temporal dynamics and cross-lead spatial dependencies simultaneously.

**Justification:**
- Standard MAE masking applied to 1D ECG would only mask time segments. Chagas-related changes (right bundle branch block, axis deviation) are fundamentally cross-lead phenomena — ST-MEM's spatial masking ensures the model learns cross-lead relationships during pretraining.
- Van Santvliet et al. achieved the top score in the PhysioNet 2025 challenge using this pretraining strategy. This provides strong empirical validation.
- Using the same pretraining scheme as the challenge winner allows the project to build directly on validated methodology.

**Output:** Pretrained 1D-ViT weights saved as `stmem_1d_pretrained.pt`.

> **Note on Pretraining Data Leakage:** The pretraining used the full 366k dataset before fold splits were applied, meaning validation fold samples were seen by both ViTs in an unsupervised context. This is standard practice in self-supervised learning (no label information is exposed) but will be explicitly disclosed as a limitation in the thesis.

---

## Stage 5 — Model Architecture: HybridChagasModel

### 5.1 Architecture Overview

```
1D signal (12, 1000) ──────────────────────────────────────► 1D ViT (ST-MEM)
                                                                     │
                                                                     ▼
                                                              FM features (768d)
                                                                     │
2D image (3, 24, 2048) ──────────────────────────────────────► 2D ViT (MAE)
                                                                     │
                                                                     ▼
                                                          2D features (768d)
                                                                     │
                                                             REPA alignment
                                                                     │
                                                          Aligned 2D (768d)
                                                                     │
                                   [Aligned 2D | FM features] → (1536d)
                                                                     │
                                                        3-layer MLP classifier
                                                                     │
                                                             Logit (binary)
```

**Total parameters: ~173 million.**

### 5.2 1D Vision Transformer (with Demographics)

- Patches each lead independently: Conv1d(1→768, kernel=50, stride=50) → 20 patches per lead × 12 leads = **240 tokens**.
- Lead embeddings added per-lead to distinguish anatomical position (aVR vs V1 etc.).
- Learned positional embeddings added per temporal position within each lead.
- **Demographics modulation:** A small MLP maps (age, sex) → (γ, β) via FiLM (Feature-wise Linear Modulation), applied after the transformer. This allows the model to adjust its features based on patient demographics without requiring demographics to be tokenised and processed through the full attention stack.
- AoL (Aggregation of Layers): outputs from all 12 transformer layers are mean-pooled, then averaged across layers → (B, 768).

**Justification for 1D ViT:**
- Attention mechanisms are well-suited to ECG: long-range dependencies (e.g. P-wave to QRS to T-wave, 200–400ms apart) are easily captured by global self-attention, unlike CNNs which are limited by receptive field.
- ViT scales well with pretraining, unlike RNNs which are harder to pretrain with masked modelling.
- Per-lead patching respects the physiological independence of each lead before cross-lead information is merged in the transformer layers.

**Justification for demographics modulation:**
- Age and sex are strong confounders for Chagas ECG findings. Right bundle branch block prevalence increases with age independently of Chagas. FiLM modulation allows demographic conditioning without hard-coding a one-size-fits-all decision boundary.

### 5.3 2D Vision Transformer

- Patches the 2D ECG image: Conv2d(3→768, kernel=8×64, stride=8×64) → 3×32 = **96 patches**.
- Standard transformer (12 layers, 12 heads, embed=768).
- AoL aggregation.

**Justification for 2D pathway:**
- Spatial attention over the 2D representation allows the model to learn which combinations of leads (rows) at which time positions (columns) are most informative — a form of learned cross-lead temporal alignment.
- The 2D image preserves relative lead amplitude relationships across all 12 leads simultaneously in a format amenable to spatial convolution and attention.

### 5.4 REPA Cross-Modal Alignment

REPA (Representation Alignment) is a lightweight projection module (depthwise Conv → SiLU → Linear) that projects 2D features into the space of 1D FM features.

During training, a cosine-similarity alignment loss encourages the 2D representation to align with the 1D representation:

```
L_align = 1 − cosine_similarity(aligned_2D, FM_features.detach())
```

The FM features are **detached** (treated as a fixed target), so only the 2D pathway and the projection are trained against this loss.

**Justification:**
- Forces the 2D pathway to learn representations consistent with what the 1D pathway (pretrained with ST-MEM) has already learned to capture.
- Acts as a regulariser for the 2D pathway, preventing it from drifting to purely texture-based features.
- Cross-modal alignment is known to improve robustness and generalisation in multimodal medical imaging.
- This is the REPA approach from Kim et al. (2025), adapted here for ECG.

---

## Stage 6 — Fine-Tuning Strategy (Per-Fold, 2-Phase)

### 6.1 Phase 1 — Warm-Up (2,000 iterations)

- **Freeze:** 1D-ViT (FM backbone) completely.
- **Train:** 2D-ViT + REPA + MLP classifier.
- **LR:** 2e-4 with 200-step linear warmup.
- **Effective batch size:** 16 × 4 (gradient accumulation) = 64.

**Justification:**
- The 1D-ViT is the larger, more critical backbone. Freezing it initially prevents the pretrained FM representations from being corrupted early in training when the classifier is random and produces large, noisy gradients.
- The 2D-ViT is first aligned to the FM representation space via REPA before the FM is unfrozen, ensuring the cross-modal interface is stable before joint optimisation.
- This mirrors the standard progressive unfreezing strategy used in transfer learning for NLP (ULMFiT) and computer vision.

### 6.2 Phase 2 — Full Fine-Tuning (24,000 iterations)

- **Unfreeze:** All parameters, including 1D-ViT (FM).
- **Differential learning rates:** High LR (2e-4) for classifier and REPA; Low LR (2e-5) for both ViT backbones.
- **Effective batch size:** 16 × 2 = 32.

**Justification:**
- Differential LR (also called discriminative fine-tuning) prevents the pretrained backbone weights from being overwritten too quickly. The classifier is randomly initialised and needs a higher LR; the backbone has useful pretrained features that should be updated gently.
- The factor-of-10 LR difference follows established practice in vision fine-tuning (FastAI, BiT, etc.).
- Gradient clipping (max_norm=1.0) stabilises training when both pathways are jointly optimised with different LRs.

### 6.3 Loss Function: Asymmetric BCE

```
L = −w_pos × (1−p)^γ⁺ × y × log(p)  −  p^γ⁻ × (1−y) × log(1−p)

γ⁺ = 0    (no focusing on positives — they are already upweighted)
γ⁻ = 2    (focus on hard negatives — suppress easy negatives)
w_pos = 10 (explicit class weight for 2.24% prevalence)
```

**Justification:**
- Standard BCE with no weighting would give the model an easy path to >97% accuracy by predicting everything negative. The positive weight (×10) directly counters the class imbalance.
- The asymmetric focusing exponent on negatives (γ⁻=2) is the key insight from ASL (Ridnik et al., 2021): in highly imbalanced settings, the loss is dominated by easy negatives (confidently predicted as negative). Focal loss on negatives suppresses these, forcing the model to focus on genuinely ambiguous and hard cases.
- γ⁺=0 (no focusing on positives) because positive examples are rare and all of them are "hard" in the sense that they are underrepresented — we do not want to discount any of them.

### 6.4 Training Hardware & Practicalities

- **GPU:** NVIDIA RTX 3050 6 GB VRAM.
- **AMP (fp16):** ~2–3× memory and speed improvement; allows batch size 16 which would OOM in fp32.
- **Gradient accumulation:** Simulates larger effective batch without increasing memory.
- **Weighted sampling:** 5× oversampling of positives so the model sees sufficient positive examples per batch despite 2.24% prevalence.
- **Time:** ~11 hours per fold on the RTX 3050 → ~55 hours total for 5 folds.

---

## Stage 7 — Augmentations

Applied **only at training time** on the raw 1D signal (before image construction):

| Augmentation | Probability | Justification |
|---|---|---|
| Lead mixup (α=0.2) | 30% | Regularisation; ECG morphology varies by patient — interpolating between patients forces learning of invariant features |
| Powerline noise (50/60 Hz ± harmonics, SNR 15–30 dB) | 50% | Realistic clinical artefact; model must be robust to mains interference present in real hospital recordings |
| Random temporal shift (±100 samples) | 50% | ECG onset position varies; forces temporal invariance |
| Amplitude scaling (×0.8–1.2) | 30% | ECG amplitude varies with electrode placement; calibrated gain uncertainty |
| Baseline wander (0.1–0.5 Hz) | 20% | Residual low-frequency noise after filtering; real-world robustness |

**Powerline noise is added identically across all 12 leads** — a critical detail (confirmed by Van Santvliet et al.) because real mains interference affects all leads simultaneously via common-mode coupling.

**No augmentation is applied to the 2D images directly** — the image is constructed from the augmented 1D signal, so augmentations propagate naturally.

---

## Stage 8 — Soft Labels

For CODE-15% specifically (the large Brazilian dataset), label quality is uncertain: labels were assigned algorithmically from clinical reports, not from serology. To reflect this uncertainty:

```
Hard label 0 → soft label 0.2
Hard label 1 → soft label 0.8
```

**Justification:**
- Label smoothing is known to improve model calibration and prevent overconfident predictions.
- The 0.8/0.2 range (rather than the more aggressive 0.9/0.1) reflects the genuinely uncertain provenance of CODE-15% labels.
- PTB-XL and SaMi-Trop retain hard labels (0/1) as their labels are more reliably verified.

---

## Stage 9 — Evaluation

### 9.1 Primary Metric: TPR@5%

**Definition:** Sort all patients by predicted score (descending). Take the top 5% of patients. What fraction of true Chagas-positive patients appear in that top 5%?

```
capacity = floor(0.05 × N)
TPR@5% = |{true positives in top 'capacity' predictions}| / |{total true positives}|
```

**Justification:** This is the official PhysioNet 2025 challenge metric. It directly models the **clinical deployment scenario**: a hospital AI system flags the highest-risk patients for confirmatory testing, but can only process a fixed capacity. It rewards precision at the top of the risk ranking, which is the actionable regime.

Computed using the official PhysioNet permutation test (10,000 permutations for full evaluation, 1,000 for mid-training checks).

### 9.2 Secondary Metrics

| Metric | Purpose |
|---|---|
| AUROC | Discrimination across all thresholds; benchmark against prior work |
| AUPRC | More informative than AUROC for imbalanced datasets |
| MCC | Balanced accuracy measure for binary classification |
| Sensitivity / Specificity / F1 | Standard clinical reporting |
| NPV | Negative Predictive Value — confidence that a negative prediction is truly negative |
| NNS (Number Needed to Screen) | "How many patients screened per true case identified?" — directly interpretable by clinicians |

**95% Bootstrap Confidence Intervals** (1,000 resamples) reported for AUROC and AUPRC.

### 9.3 5-Fold Ensemble

After training one model per fold, all 5 models are loaded and run on each validation patient. The final probability is the **mean across all 5 fold predictions**. Ensemble averaging reduces variance and typically outperforms any individual fold model.

Per-fold scores (mean ± std across folds) demonstrate stability and prevent the reported performance from being a lucky outlier.

### 9.4 Per-Dataset Breakdown

Results are reported separately for PTB-XL, SaMi-Trop, and CODE-15 to assess:
- Generalisation across heterogeneous populations (European vs South American)
- Model behaviour on enriched (SaMi-Trop) vs incidental (PTB-XL / CODE-15) cohorts

### 9.5 Benchmark Targets

| System | TPR@5% |
|---|---|
| Random baseline | 0.050 |
| Phase 1 only (frozen FM) | 0.138 |
| Kim et al. (2025) | 0.369 |
| Van Santvliet et al. (2025, top team) | 0.445 |
| **ChagaSight target** | **≥ 0.420** |

---

## Design Decision Summary

| Decision | Choice | Justification |
|---|---|---|
| ECG representation | Dual: 1D signal + 2D image | 1D captures temporal dynamics; 2D captures spatial cross-lead patterns |
| Why 1D ViT | ViT with per-lead patching | Global attention for long-range ECG dependencies; pretrain-compatible |
| Why 2D ViT | ViT on WCT re-referenced image | Spatial cross-lead co-occurrence; leverages pretrained MAE |
| Why spatial (2D image) | WCT re-referencing → 3 views | Encodes 3D cardiac electrical field from 3 limb reference points |
| Pretraining | MAE (2D) + ST-MEM (1D) | Label-free learning from 366k records; directly validated by top challenge team |
| Cross-modal alignment | REPA | Forces 2D features to be consistent with the stronger 1D FM |
| Training strategy | 2-phase progressive unfreezing | Protects pretrained backbone from random-init gradient corruption |
| Loss function | Asymmetric BCE | Handles 2.24% class imbalance + suppresses easy negatives |
| Evaluation metric | TPR@5% (official challenge) | Directly models the clinical "triage at capacity" deployment scenario |
| Cross-validation | Stratified 5-fold | Reliable performance estimate; enables ensemble; accounts for dataset heterogeneity |
| Demographics | FiLM modulation (age, sex) | Conditions predictions on known confounders without complex tokenisation |

---

## Current Status

| Stage | Status |
|---|---|
| Preprocessing pipeline | Complete |
| 5-fold stratified splits | Complete |
| MAE pretraining (2D-ViT) | Complete |
| ST-MEM pretraining (1D-ViT) | Complete |
| Fine-tuning — Fold 0 | Complete |
| Fine-tuning — Fold 1 | Complete |
| Fine-tuning — Fold 2 | Complete |
| Fine-tuning — Fold 3 | Complete |
| Fine-tuning — Fold 4 | In progress |
| Final ensemble evaluation | Pending |
| Thesis write-up | Pending |
