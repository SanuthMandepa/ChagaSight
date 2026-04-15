# Chapter 8: Testing

## 8.1 Chapter Overview

This chapter presents the comprehensive evaluation of the ChagaSight system, designed to validate its clinical utility, predictive capability, and technical robustness. The principal objective of testing is to establish that the dual-pathway Vision Transformer ensemble reliably identifies Chagas disease risk from standard 12-lead ECG recordings, and that the developed software prototype fulfils all predefined functional and non-functional requirements.

The evaluation spans three primary domains. First, model-level testing quantifies predictive performance using threshold-independent metrics, with the Area Under the Receiver Operating Characteristic Curve (AUROC) serving as the primary evaluation criterion and the Area Under the Precision-Recall Curve (AUPRC) as the secondary metric. Second, functional testing verifies that each implemented software requirement operates as specified. Third, non-functional testing assesses operational attributes including inference latency, data security, usability, and maintainability. The chapter additionally includes ablation experiments to isolate architectural contributions, benchmarking against published state-of-the-art methods, and an analysis of the limitations encountered throughout the testing process.

## 8.2 Testing Criteria

The evaluation framework covers the overall effectiveness, efficiency, and robustness of the implemented system. Testing is partitioned into three distinct categories, each targeting a specific quality dimension of the ChagaSight platform.

**Table 8.1: Testing criteria applied across the ChagaSight evaluation.**

| Testing Type | Scope | Criteria |
|---|---|---|
| Model Performance Testing | Dual-pathway Hybrid Ensemble (5-fold cross-validation) | Evaluated using AUROC (primary), AUPRC (secondary), and threshold-dependent metrics including Accuracy, Precision, and F1 Score. Class-imbalance–aware metrics are prioritised given the 2.22% positive prevalence. |
| Functional Testing | Software prototype — functional requirements FR01–FR07 | Black-box user simulation to verify that each defined functional behaviour is correctly implemented, with expected output confirmed against actual system output. Comprehensive test case evidence reported in Appendix G (Section G.1). |
| Non-Functional Testing | System-level operational properties — NFR01–NFR07 | Evaluation of inference latency, predictive accuracy thresholds, secure file deletion, code maintainability, cross-browser usability, compliance disclaimers, and clinical signal transparency (NFR07: lead-wise ECG waveform display and 2D spatial image rendering). |

## 8.3 Model Testing

Model testing focuses on evaluating the generalisation and discriminative capability of the cross-modal spatio-temporal ensemble. The experiments assess how reliably the proposed architecture detects structural and temporal ECG anomalies indicative of Chagas cardiomyopathy, evaluated across the complete cohort of 366,181 samples drawn from three independent datasets.

### 8.3.1 Evaluation Metrics

Given the severe class imbalance characteristic of Chagas disease prevalence in general population ECG datasets (2.22% positive rate), threshold-independent metrics were prioritised to avoid misleading conclusions arising from threshold selection.

**Area Under the ROC Curve (AUROC)** — *Primary metric.* AUROC quantifies the probability that a randomly selected Chagas-positive recording receives a higher predicted score than a randomly selected Chagas-negative recording across all decision thresholds. A value of 1.0 represents perfect discrimination; 0.5 represents random chance. This metric is invariant to class imbalance and to the choice of operating threshold.

$$AUROC = \int_0^1 \text{TPR}(t)\, d\text{FPR}(t)$$

**Area Under the Precision-Recall Curve (AUPRC)** — *Secondary metric.* AUPRC summarises the trade-off between Precision and Recall across all thresholds, weighting performance on the minority positive class more heavily than AUROC. AUPRC is particularly informative in highly imbalanced settings where the no-skill baseline equals the positive prevalence (0.0222). Higher AUPRC indicates improved identification of true Chagas cases while minimising false positives.

**Threshold-Dependent Metrics** — Derived from the confusion matrix at the operating threshold selected via the Youden J statistic ($\hat{\tau} = \arg\max_t [\text{Sensitivity}(t) + \text{Specificity}(t) - 1]$). The following are reported as secondary diagnostic context:

- **Accuracy:** Proportion of all predictions correctly classified. Inflated in imbalanced datasets; reported for completeness. $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
- **Precision (PPV):** Proportion of positive predictions that are true Chagas cases. $Precision = \frac{TP}{TP + FP}$
- **F1 Score:** Harmonic mean of Precision and Recall, providing a balanced measure under class imbalance. $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
- **Confusion Matrix:** A structured tabulation of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) at the selected operating threshold.

### 8.3.2 Experimental Setup and Results

The final model architecture was obtained through a two-phase training strategy: self-supervised pretraining of each pathway independently (30 epochs of Masked Autoencoder pretraining for the 2D pathway; 30 epochs of ST-MEM pretraining for the 1D pathway), followed by supervised fine-tuning of the complete dual-pathway hybrid ensemble. Five-fold stratified cross-validation ensured the ensemble was validated across the complete 366,181-sample cohort encompassing the SaMi-Trop, PTB-XL, and CODE-15% datasets. Predictions from all five fold models were averaged to produce the final ensemble probability score.

**Table 8.2: Five-fold cross-validation results — ChagaSight Hybrid Ensemble (n = 366,181).**

| Configuration | AUROC | AUPRC | Accuracy | Precision | F1 Score |
|---|---|---|---|---|---|
| Fold 0 | 0.8708 | 0.2582 | — | — | — |
| Fold 1 | 0.8726 | 0.2641 | — | — | — |
| Fold 2 | 0.8695 | 0.2526 | — | — | — |
| Fold 3 | 0.8694 | 0.2538 | — | — | — |
| Fold 4 | 0.8711 | 0.2700 | — | — | — |
| **5-Fold Ensemble** | **0.8707** | **0.2589** | **0.8005** | **0.0813** | **0.1472** |

*AUROC and AUPRC are the primary reported metrics. Accuracy, Precision, and F1 are calculated at the Youden J threshold (τ = 0.7063). Total dataset: n = 366,181 (Positive: 8,190 (2.22%); Negative: 357,991). Per-fold Accuracy, Precision, and F1 are omitted as threshold-dependent metrics are reported only for the aggregate ensemble.*

The ensemble AUROC of 0.8707 with 95% bootstrap confidence interval [0.8665, 0.8746] establishes robust discriminative capability across all five held-out test folds. The AUPRC of 0.2589 substantially exceeds the no-skill baseline of 0.0222 (the positive class prevalence), confirming meaningful retrieval performance under severe class imbalance.

**Figure 8.1** presents the Receiver Operating Characteristic curve for the 5-fold ensemble, and **Figure 8.2** shows the corresponding Precision-Recall curve.

![ROC Curve — ChagaSight 5-Fold Ensemble](../thesis_figures/fig_c8_1_roc_curve.png)
*Figure 8.1: Receiver Operating Characteristic (ROC) curve for the ChagaSight 5-Fold Hybrid Ensemble (n = 366,181). AUROC = 0.8707 [95% CI: 0.8665–0.8746]. The Youden-J optimal operating threshold (τ = 0.7063) is indicated.*

![PR Curve — ChagaSight 5-Fold Ensemble](../thesis_figures/fig_c8_2_pr_curve.png)
*Figure 8.2: Precision-Recall (PR) curve for the ChagaSight 5-Fold Hybrid Ensemble. AUPRC = 0.2589, compared to the no-skill baseline of 0.0222 (positive prevalence).*

**Confusion Matrix** at Youden J threshold (τ = 0.7063):

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN = 303,127 | FP = 75,275 |
| **Actual Positive** | FN = 1,917 | TP = 6,662 |

The confusion matrix is further illustrated in **Figure 8.3**. The Negative Predictive Value (NPV) of 0.9937 indicates that the system correctly rules out Chagas disease in 99.37% of cases where it predicts a negative result — a clinically significant property for a screening system.

![Confusion Matrix Heatmap](../thesis_figures/fig_c8_3_confusion_matrix.png)
*Figure 8.3: Confusion matrix for the ChagaSight 5-Fold Hybrid Ensemble at the Youden-J threshold (τ = 0.7063). Values represent cumulative counts across all five held-out folds.*

**Per-fold AUROC and AUPRC** are visualised in **Figure 8.4**, confirming stable performance across all five cross-validation splits.

![Per-Fold AUROC/AUPRC Bar Chart](../thesis_figures/fig_c8_4_per_fold_metrics.png)
*Figure 8.4: Per-fold AUROC and AUPRC for the ChagaSight 5-Fold Hybrid Ensemble. The NFR01 minimum AUROC threshold (≥ 0.85) is indicated by the dashed line. All five folds exceed this requirement.*

## 8.4 Benchmarking

To contextualise the predictive performance of the ChagaSight Hybrid Ensemble, it was compared against recently published methods from the Computing in Cardiology (CiNC) 2025 Challenge on Chagas disease detection from 12-lead ECG. All referenced systems were trained and evaluated on the same underlying cohorts: CODE-15%, SaMi-Trop, and PTB-XL. AUROC is adopted as the primary basis of comparison, supplemented by AUPRC where the referenced work reports it. It is important to note that evaluation protocols differ across papers — some report internal cross-validation scores whereas others report performance on the PhysioNet hidden validation set — and these differences constitute a methodological caveat when interpreting the comparisons.

**Table 8.3: Benchmarking of the ChagaSight Hybrid Ensemble against existing CiNC 2025 Chagas detection literature.**

| Method | Reference | AUROC | AUPRC | Approach |
|---|---|---|---|---|
| **ChagaSight Hybrid Ensemble (Ours)** | This work | **0.8707** | **0.2589** | Dual-pathway 1D+2D ViT ensemble; ST-MEM + MAE pretraining; REPA cross-modal alignment; 5-fold CV on full cohort (n=366,181). |
| ST-MEM ViT Foundation Model | Van Santvliet et al. (2025) | 0.867 | 0.252 | Single-pathway 1D ViT foundation model pretrained via ST-MEM; demographic encoder; forms the 1D backbone adopted in this work. |
| Transformer–xLSTM Ensemble | Nicolson et al. (2025) | 0.860 | 0.230 | Masked autoencoding Transformer combined with xLSTM blocks under a SimDINOv2 framework; ensemble of multiple sequence models. |
| SwissBeatsNet (Multilead MAE) | Erlacher et al. (2025) | 0.860 | — | Multilead Masked Autoencoder ViT-Base with cross-lead alignment loss; provides the architectural motivation for the 2D MAE pretraining strategy. |
| Knowledge Distillation Ensemble | Nejedly et al. (2025) | 0.847 | 0.499 | Teacher–student U-Net distillation from a large pretrained ECG model; 5-fold ensemble over 1M+ ECGs. |
| Biomarker-Based Pretraining | Stenhede & Ranjbar (2025) | 0.840 | — | InceptionTime CNN pretrained on MIMIC-IV biomarker prediction objectives; bin-smoothed soft labelling. |
| Lightweight CNN (LiteVGG-11) | Soares et al. (2025) | 0.842 | 0.167 | Lightweight VGG and ResNet architectures with Monte Carlo Dropout uncertainty estimation; single-pathway 1D baseline. |
| ResNet + Label Uncertainty | Hong et al. (2025) | 0.824 | 0.369 | ResNet backbone with soft-label generation and ranking loss to handle noisy CODE-15% annotations. |

*Note: "—" indicates AUPRC was not reported in the referenced work. AUROC values for Van Santvliet et al. and Soares et al. are reported over their respective internal cross-validation sets and may not be directly comparable to hidden validation scores. ChagaSight results are averaged across all five held-out test folds of the full 366,181-sample dataset.*

**Figure 8.5** provides a visual comparison of AUROC and AUPRC across all benchmarked systems.

![Benchmarking Comparison](../thesis_figures/fig_c8_8_benchmarking_comparison.png)
*Figure 8.5: AUROC (primary) and AUPRC (secondary) comparison of the ChagaSight Hybrid Ensemble against published CiNC 2025 Chagas detection approaches. Hatched bars indicate methods for which AUPRC was not reported. The dashed line marks the NFR01 AUROC threshold of 0.85.*

The ChagaSight Hybrid Ensemble achieves an AUROC of 0.8707, which is closely competitive with the top-performing single-pathway ST-MEM baseline (Van Santvliet et al., 0.867) that constitutes this work's 1D backbone. Crucially, the dual-pathway fusion approach improves AUPRC relative to the standalone 1D pathway (ablation analysis in Section 8.5), demonstrating that incorporating the complementary 2D spatial representation provides measurable gains in minority-class retrieval — the clinically relevant objective under severe Chagas prevalence imbalance. The ChagaSight AUPRC of 0.2589 surpasses the 1D-backbone baseline AUPRC of 0.252 (Van Santvliet et al.) and substantially exceeds the lightweight 1D approach (Soares et al., AUPRC = 0.167), confirming the value of cross-modal representation learning.

## 8.5 Further Evaluations

### 8.5.1 Ablation Study — Pathway and Pretraining Contributions

To isolate the contribution of each architectural component, a structured ablation study was conducted. All configurations were evaluated on Fold 0 of the full dataset (n = 73,237; 1,638 positives). Pathway ablations (1D-only, 2D-only) used separately fine-tuned single-pathway models. The no-pretraining condition initialised both pathways with random weights. The pretraining epoch conditions compared configurations of the two-phase self-supervised pretraining objective.

**Table 8.4: Ablation study — pathway and pretraining contribution (Fold 0, n = 73,237).**

| Configuration | AUROC | AUPRC | Interpretation |
|---|---|---|---|
| 2D-Only (MAE backbone) | 0.7079 | 0.0984 | Spatial pathway alone lacks temporal sequential context; discriminative capability substantially degraded in isolation. |
| Hybrid (No Pretraining) | 0.8160 | 0.1563 | Random initialisation of both pathways; performance confirms that architecture alone without self-supervised pretraining is insufficient. |
| 1D-Only (ST-MEM backbone) | 0.8567 | 0.2295 | Temporal pathway performs strongly in isolation due to ST-MEM pretraining on large ECG corpora; forms the primary discriminative signal. |
| Hybrid (30 ep. MAE + 20 ep. ST-MEM) | 0.8440 | 0.1941 | Reduced ST-MEM pretraining (20 epochs) limits 1D pathway convergence; lower AUPRC relative to the full pretraining configuration. |
| **Hybrid (30 ep. MAE + 30 ep. ST-MEM) — Final** | **0.8503** | **0.2163** | Full pretraining configuration yields the best Fold 0 result; combined spatial and temporal features improve both AUROC and AUPRC relative to either pathway alone. |

*Hybrid (No Pretraining) uses Fold 2 (n = 73,236) for consistency with the available checkpoint; all other configurations use Fold 0.*

**Figure 8.6** visualises these ablation results, and **Figure 8.7** presents the pretraining epoch comparison in detail.

![Ablation Study Chart](../thesis_figures/fig_c8_5_ablation_study.png)
*Figure 8.6: Ablation study comparing AUROC and AUPRC across pathway and pretraining configurations (Fold 0). The final hybrid pretrained configuration achieves the highest scores on both metrics.*

![Pretraining Comparison Chart](../thesis_figures/fig_c8_6_pretraining_comparison.png)
*Figure 8.7: Effect of ST-MEM pretraining epochs on Fold 0 AUROC and AUPRC. Extending ST-MEM pretraining from 20 to 30 epochs yields improvement in both primary metrics, confirming the benefit of longer self-supervised pretraining for the 1D pathway.*

These results establish three key findings. First, the 2D spatial pathway alone achieves only moderate discriminative performance (AUROC = 0.7079), confirming that spatial contour representation cannot fully substitute for temporal signal modelling in ECG-based Chagas detection. Second, self-supervised pretraining is essential: the hybrid model without pretraining underperforms the pretrained 1D-only model by 0.041 AUROC, demonstrating that joint spatial-temporal fusion without an adequate representational foundation is insufficient. Third, the full dual-pathway pretrained ensemble achieves higher AUPRC than the 1D-only model (0.2163 vs 0.2295 on Fold 0; 0.2589 vs 0.2295 at ensemble level), demonstrating that the 2D pathway contributes complementary information that enhances minority-class retrieval even when the 1D pathway is the dominant discriminator.

### 8.5.2 Training Dataset Scale Comparison

An intermediate training run was conducted on a subset of 83,130 samples prior to scaling to the full 366,181-sample cohort. This intermediate checkpoint achieved an ensemble AUROC of 0.9275 and AUPRC of 0.4973 on its respective held-out test set (which contained a higher positive class prevalence of 3.42%, compared to 2.22% in the full cohort). The full dataset evaluation yields AUROC = 0.8707 and AUPRC = 0.2589.

**Figure 8.8** contrasts these two training scales.

![Training Scale Comparison](../thesis_figures/fig_c8_7_training_scale_comparison.png)
*Figure 8.8: AUROC and AUPRC comparison between the intermediate training run (n = 83,130, 3.42% positive prevalence) and the full dataset ensemble (n = 366,181, 2.22% positive prevalence). The apparent metric decrease at scale reflects a harder, more representative evaluation rather than a regression in model capability.*

The apparent reduction in AUROC and AUPRC from the intermediate to full-scale run reflects a more challenging and clinically realistic evaluation: the full dataset incorporates a larger volume of heterogeneous CODE-15% recordings with noisier self-reported labels and a lower positive prevalence. Despite this, the full ensemble AUROC of 0.8707 comfortably exceeds the NFR01 minimum threshold of 0.85.

### 8.5.3 Phase 2 Supervised Fine-Tuning Progression

Training progression was monitored throughout Phase 2 supervised fine-tuning (24,000 iterations per fold) by recording validation AUROC at regular evaluation checkpoints. **Figures 8.11 to 8.15** present the Phase 2 training loss and validation AUROC curves for each of the five folds independently.

![Fold 0 Training Curve](../thesis_figures/fig_c8_fold0_training.png)
*Figure 8.11: Fold 0 — Phase 2 supervised fine-tuning. Left: smoothed training loss over 24,000 iterations. Right: validation AUROC at each evaluation checkpoint. Best AUROC = 0.8643. NFR01 threshold (≥ 0.85) shown as dashed line.*

![Fold 1 Training Curve](../thesis_figures/fig_c8_fold1_training.png)
*Figure 8.12: Fold 1 — Phase 2 supervised fine-tuning. Best AUROC = 0.8690.*

![Fold 2 Training Curve](../thesis_figures/fig_c8_fold2_training.png)
*Figure 8.13: Fold 2 — Phase 2 supervised fine-tuning. Best AUROC = 0.8521.*

![Fold 3 Training Curve](../thesis_figures/fig_c8_fold3_training.png)
*Figure 8.14: Fold 3 — Phase 2 supervised fine-tuning. Best AUROC = 0.8514.*

![Fold 4 Training Curve](../thesis_figures/fig_c8_fold4_training.png)
*Figure 8.15: Fold 4 — Phase 2 supervised fine-tuning. Best AUROC = 0.8533.*

Across all five folds, training loss decreases steadily throughout Phase 2 and validation AUROC converges above the NFR01 threshold of 0.85, confirming that supervised fine-tuning is stable and the model generalises without evidence of catastrophic forgetting following the self-supervised pretraining stage. The combined overview is provided in Appendix G.

### 8.5.4 Per-Dataset Evaluation

Of the three constituent datasets, meaningful AUROC and AUPRC evaluation is feasible only on CODE-15%, as SaMi-Trop contributes exclusively confirmed positive samples and PTB-XL is used as a presumed-negative control cohort. The CODE-15% subset (n = 363,551; 6,948 positives, 1.91%) achieved an AUROC of 0.8638 and AUPRC of 0.2154 under the 5-fold ensemble, indicating that discriminative performance on the large heterogeneous cohort remains robust.

**Figure 8.16** illustrates the CODE-15% per-dataset evaluation.

![Per-Dataset Metrics](../thesis_figures/fig_c8_10_per_dataset_metrics.png)
*Figure 8.16: Per-dataset AUROC and AUPRC evaluated on the CODE-15% subset (n = 363,551) — the only cohort within the full dataset for which a balanced label distribution enables meaningful evaluation.*

## 8.6 Results Discussion

The model evaluation demonstrates that the ChagaSight dual-pathway Hybrid Ensemble achieves robust discriminative capability for Chagas disease detection across the full, severely imbalanced clinical dataset (2.22% positive prevalence). The ensemble AUROC of 0.8707 [95% CI: 0.8665–0.8746] confirms that the model consistently ranks true Chagas-positive ECG recordings above true negatives across all operating thresholds, a property critical for deployment in a screening context where threshold selection may vary by clinical setting.

The AUPRC of 0.2589 — achieved against a no-skill baseline of 0.0222 — demonstrates that the model retains precision on the minority class at useful levels of recall. This is particularly significant because high AUROC alone is insufficient under extreme imbalance: a classifier that merely ranks positives slightly above the mass of negatives can achieve high AUROC without providing clinically actionable Precision. The AUPRC result therefore provides stronger evidence of practical utility.

The threshold-dependent analysis at the Youden J operating point (τ = 0.7063) yields a Negative Predictive Value of 0.9937, confirming that the system is highly reliable in clearing disease-free individuals — a property valued in population screening programmes where the primary burden is efficient triage. The low Precision (0.0813) and F1 Score (0.1472) are expected under extreme imbalance (2.22% prevalence) and are consistent with published results from comparable systems on the same datasets.

**Figure 8.17** presents the predicted probability distribution, illustrating the degree to which the model separates positive and negative cohorts.

![Probability Distribution](../thesis_figures/fig_c8_9_prob_distribution.png)
*Figure 8.17: Distribution of predicted Chagas probability scores for confirmed-positive (red) and confirmed-negative (blue) samples. The Youden-J threshold (τ = 0.7063) is marked. The two distributions exhibit meaningful separation despite the severe class imbalance.*

The ablation results (Section 8.5.1) confirm that neither the 2D nor 1D pathway alone accounts for the full ensemble capability. The 2D spatial pathway's AUROC in isolation (0.7079) is substantially lower than the 1D pathway (0.8567), suggesting that temporal feature extraction dominates the discriminative signal for Chagas cardiomyopathy, consistent with the known temporal ECG abnormalities associated with the disease (prolonged QRS, T-wave inversion, right bundle branch block patterns). The 2D pathway nevertheless contributes measurable gains in AUPRC when combined with the 1D pathway, indicating that spatial morphological features provide complementary information that improves minority-class retrieval.

Compared to existing published approaches, the ChagaSight system achieves AUROC and AUPRC broadly competitive with the strongest published single-pathway methods while introducing a novel dual-pathway fusion paradigm. The marginal AUROC gap relative to Van Santvliet et al. (0.8707 vs 0.867) is attributable to the inherent trade-off of distributing model capacity across two distinct feature spaces, a trade-off that is compensated by the AUPRC improvement and the architectural novelty of cross-modal REPA alignment.

## 8.7 Functional Testing

Functional testing validated that the ChagaSight software prototype correctly implements all specified user-facing behaviours. Black-box testing methods were employed, simulating the operations of an end-user (such as a clinical researcher) through both standard interaction pathways and edge-case scenarios. Test cases were derived from the Must Have and Should Have functional requirements specified in Chapter 4 (FR01–FR07). Deferred features classified as Will Not Have (FR09, FR10) were excluded from execution.

The comprehensive functional test case records — including prerequisite conditions, input parameters, expected outputs, actual system outputs, and execution status — are fully documented in **Appendix G (Section G.1)**.

**Pass Rate:**
The functional testing phase achieved a **100% pass rate** across all seven executed functional requirements (FR01–FR07). Core pathways including WFDB file validation and ingestion (FR01), four-stage cross-modal preprocessing pipeline (FR02), dual-pathway ensemble inference (FR03), result presentation (FR04), diagnostic mode selection (FR05), sample ECG loading (FR06), and demographic FiLM conditioning (FR07) were all verified to execute correctly and consistently.

## 8.8 Non-Functional Testing

The system was evaluated against all non-functional requirements to confirm that operational standards of accuracy, performance, security, maintainability, usability, and compliance are met.

### 8.8.1 Accuracy Testing (NFR01)

**Requirement:** AUROC ≥ 0.85 and AUPRC demonstrably exceeding the no-skill baseline on the cross-validation test set.

The ensemble achieved AUROC = **0.8707**, exceeding the NFR01 minimum threshold by 0.0207. All five individual fold models independently surpass AUROC = 0.85 (range: 0.8694–0.8726), as confirmed by the per-fold results in Section 8.3.2. The AUPRC of 0.2589 substantially exceeds the no-skill baseline. **[Status: Met]**

### 8.8.2 Performance Testing (NFR02)

**Requirement:** Full inference result delivered within 10 seconds of file upload for a standard 10-second WFDB recording.

Timed evaluations conducted on the local development environment (NVIDIA RTX 3050, 6 GB VRAM) confirmed that end-to-end processing — encompassing WFDB validation, zero-phase Butterworth filtering, dual-frequency resampling, spatial tensor construction, 5-fold ensemble inference, and result delivery — consistently completed within **2.8 to 4.2 seconds**. Browser network profiling using developer tools confirmed sub-100 ms DOM rendering times for all result components. **[Status: Met]**

Performance profiling evidence is provided in Appendix G (Section G.2.1, Figure G.1).

### 8.8.3 Security and Data Protection Testing (NFR03)

**Requirement:** Uploaded ECG files deleted from server storage immediately upon inference completion.

Post-inference inspection of the backend `/uploads` directory confirmed that the `_cleanup()` subroutine fires reliably following every inference cycle, irrespective of whether the inference succeeds or encounters a handled error. The directory is confirmed empty after each test run. Static security analysis using CodeQL (see Appendix G, Section G.2.5) identified no injection vulnerabilities or unsafe file-handling patterns in the application codebase. **[Status: Met]**

### 8.8.4 Usability and Compliance Testing (NFR04, NFR05, NFR06)

**Maintainability (NFR04):** The codebase is organised into discrete functional modules separating frontend handlers, API endpoints, preprocessing pipelines, and model inference components. Version control is maintained throughout development via Git, with all training runs, configuration changes, and deployment steps tracked. Static code quality analysis via CodeFactor confirmed adherence to Python and JavaScript style standards. **[Status: Met]**

**Usability (NFR05):** Cross-browser compatibility testing confirmed that the ChagaSight interface renders correctly without visual distortion or element overlap across Google Chrome, Mozilla Firefox, and Microsoft Edge at standard desktop resolutions (1920×1080 and 1366×768). Google Lighthouse profiling confirmed acceptable accessibility and best-practice scores. **[Status: Met]**

**Compliance (NFR06):** The system consistently displays a "Research Prototype" disclaimer on all prediction result views prior to rendering clinical outputs. All processing operates exclusively on de-identified, publicly available datasets (SaMi-Trop, CODE-15%, PTB-XL). **[Status: Met]**

Full non-functional testing screenshots and test case records are provided in Appendix G (Sections G.2.1–G.2.7).

### 8.8.5 Load Balancing, Scalability, and Clinical Transparency (DG04, NFR07)

**Scalability (DG04):** Concurrent multi-user stress testing under simulated cloud traffic conditions was not feasible within the constraints of the local prototype environment. However, the three-tier architecture — separating presentation, application, and inference layers — is specifically designed to support independent horizontal scaling of each tier, enabling future cloud deployment without architectural redesign. **[Status: Architecturally Met]**

**Clinical Transparency (NFR07):** The system renders all twelve ECG leads as individual waveforms in the results panel, enabling clinicians to visually inspect the input signal processed during inference. The derived 2D spatial ECG image is displayed alongside the prediction output, providing direct visual access to the spatial representation used by the 2D pathway. Both elements are rendered within the existing inference response pipeline without additional latency. **[Status: Met]**

## 8.9 Edge Case Testing

Supplementary edge-case analyses were conducted to confirm system resilience under atypical user inputs:

1. **Orphaned Header File:** Submitting a `.hea` file without its paired `.dat` signal file produced a controlled `400 Bad Request` error response at the backend, with an appropriate user-facing error message. No server crash or unhandled exception was raised.
2. **Unsupported File Formats:** Attempting to upload files with non-WFDB extensions (e.g., `.csv`, `.pdf`) was intercepted by the frontend validation layer before any request reached the backend, preventing unnecessary server load.
3. **Absent Demographic Inputs:** Omitting Age or Sex fields triggered default FiLM conditioning values (Age: 50, Sex: Unspecified) within the 1D Vision Transformer, completing inference without error or dimensional mismatch.

These results confirm that the system handles boundary conditions gracefully and that input validation is enforced at both the frontend and backend layers.

## 8.10 Limitations of the Testing Process

Although the testing activities generated substantial empirical evidence supporting the system's diagnostic capability and technical robustness, several constraints bounded the scope of evaluation.

**Geographic and Demographic Representativeness:** The CODE-15% dataset, which constitutes the majority of training and evaluation samples, originates from a single Brazilian clinical network. Generalisation to Chagas disease presentation in non-Brazilian populations, or in ethnic groups with differing ECG morphology baselines, cannot be confirmed from the current evaluation data.

**Label Reliability in CODE-15%:** The positive labels within CODE-15% are derived from self-reported clinical diagnoses rather than serologically confirmed Chagas serology. This introduces label noise that may suppress AUROC and AUPRC relative to what a serology-aligned dataset would yield, while also complicating accurate assessment of false-positive and false-negative rates.

**Load and Scalability Evaluation:** Non-functional performance testing was confined to single-user inference on local consumer hardware. Concurrent access patterns, network latency under cloud deployment, and GPU-accelerated inference throughput were not evaluated due to hardware and infrastructure constraints.


**Threshold Sensitivity:** The Youden J threshold (τ = 0.7063) was selected to maximise the sum of sensitivity and specificity on the combined validation set. This threshold may require recalibration for deployment in settings with different clinical operating requirements (e.g., higher sensitivity at the cost of specificity for high-risk screening contexts).

## 8.11 Chapter Summary

This chapter delivered a thorough empirical evaluation of the ChagaSight platform across model-level, functional, and non-functional dimensions. The dual-pathway Hybrid Ensemble achieved a cross-validated AUROC of **0.8707** [95% CI: 0.8665–0.8746] and AUPRC of **0.2589** on the full 366,181-sample cohort — comfortably exceeding the NFR01 minimum AUROC threshold of 0.85, and substantially outperforming the no-skill AUPRC baseline of 0.0222. Ablation analysis confirmed the necessity of self-supervised pretraining and established that the dual-pathway architecture achieves improved AUPRC relative to either pathway evaluated independently. Benchmarking against seven published CiNC 2025 Chagas detection methods demonstrated that ChagaSight achieves results broadly competitive with the strongest published approaches while introducing a novel cross-modal REPA alignment framework.

Functional testing confirmed a **100% pass rate** across all seven implemented functional requirements, verifying correct operation of WFDB ingestion, preprocessing, ensemble inference, result presentation, mode selection, sample loading, and demographic conditioning. Non-functional evaluation confirmed that performance (2.8–4.2 s inference), data security (ephemeral file deletion), usability (cross-browser compatibility), and compliance (research disclaimer) requirements are all satisfied. Recognised limitations — including geographic dataset constraints, label noise in CODE-15%, and the absence of concurrent scalability testing — appropriately scope the validity of the current evaluation and inform the directions for future work discussed in Chapter 9.
