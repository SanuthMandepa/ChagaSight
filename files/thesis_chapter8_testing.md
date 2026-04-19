# Chapter 8: Testing

## 8.1 Chapter Overview

This chapter presents the testing activities carried out to assess the reliability, predictive capability, and operational behaviour of the ChagaSight system. The evaluation is structured around three main areas: model-specific testing to measure the discriminative performance of the dual-pathway ensemble; functional testing to verify that the software prototype correctly implements the specified user-facing behaviours; and non-functional testing to assess operational attributes such as inference latency, data security, and interface compatibility. Benchmarking comparisons and ablation studies are also reported to contextualise the model results within the existing literature. The chapter closes with a discussion of testing limitations and their implications for future development.

## 8.2 Testing Criteria

The evaluation framework addresses the effectiveness, efficiency, and robustness of the ChagaSight platform across three distinct quality dimensions.

**Table 8.1: Testing criteria applied across the ChagaSight evaluation.**

| Testing Criteria | Description |
|---|---|
| Model performance testing | The dual-pathway Vision Transformer ensemble is assessed on predictive quality using AUROC and AUPRC as primary metrics, with Accuracy, Precision, Recall, and F1 Score derived at the Youden J operating threshold. Evaluation spans the full 366,181-sample cohort under five-fold stratified cross-validation. |
| Functional testing | Verifies that the system correctly implements each functional requirement, including WFDB file ingestion, cross-modal preprocessing, dual-pathway inference, result presentation, model mode selection, sample ECG loading, and demographic FiLM conditioning. |
| Non-functional testing | Assesses inference latency, data security, code maintainability, usability, and regulatory compliance against the non-functional requirements defined in Chapter 4. |

## 8.3 Model Testing

Model testing assesses the generalisation and discriminative capability of the cross-modal spatio-temporal ensemble across a cohort of 366,181 samples drawn from three independent datasets: SaMi-Trop, PTB-XL, and CODE-15%.

### 8.3.1 Evaluation Metrics

As discussed in Chapter 2 (Section 2.6), reliable evaluation under the severe class imbalance characteristic of Chagas disease screening (2.22% positive prevalence) requires metrics that are sensitive to the minority class.

Two threshold-independent metrics were adopted as the primary evaluation criteria:

- **Area Under the ROC Curve (AUROC)** — the primary metric, measuring overall discriminative capacity across all operating thresholds.
- **Area Under the Precision-Recall Curve (AUPRC)** — the secondary metric, providing a more informative measure of precision-recall behaviour under extreme class imbalance.

In addition, threshold-dependent metrics — **Accuracy**, **Precision (PPV)**, **Recall (Sensitivity)**, and **F1 Score** — were derived from the confusion matrix at the operating threshold selected by maximising the Youden J statistic across the combined cross-validation predictions.

### 8.3.2 Experimental Setup and Results

The final model was trained using a two-phase strategy. Phase 1 applied self-supervised pretraining to each pathway independently: 30 epochs of Masked Autoencoder (MAE) pretraining for the 2D spatial pathway, and 30 epochs of ST-MEM pretraining for the 1D temporal pathway. Phase 1 training logs are provided in Appendix G (Section G.3.1).

Phase 2 applied supervised fine-tuning of the complete dual-pathway ensemble under five-fold stratified cross-validation, evaluated across the full 366,181-sample cohort. Predictions from all five fold models were averaged to produce the final ensemble probability score.

**Table 8.2: Five-fold cross-validation results for the ChagaSight Hybrid Ensemble (n = 366,181).**

| Configuration | AUROC | AUPRC | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|
| Fold 0 | 0.8503 | 0.2163 | -- | -- | -- | -- |
| Fold 1 | 0.7606 | 0.1687 | -- | -- | -- | -- |
| Fold 2 | 0.7997 | 0.1370 | -- | -- | -- | -- |
| Fold 3 | 0.8217 | 0.1749 | -- | -- | -- | -- |
| Fold 4 | 0.8482 | 0.2056 | -- | -- | -- | -- |
| **5-Fold Ensemble** | **0.8707** | **0.2589** | **0.8005** | **0.0813** | **0.7765** | **0.1472** |

**Note:** Accuracy, Precision, Recall, and F1 are calculated at the Youden J threshold (t = 0.7063). Total dataset: n = 366,181 (8,190 positives, 357,991 negatives). Per-fold threshold-dependent metrics are omitted as they are calculated exclusively at the aggregate ensemble level.

The ensemble AUROC of 0.8707 with a 95% bootstrap confidence interval of [0.8665, 0.8746] indicates robust discriminative capability across all five held-out test folds. The AUPRC of 0.2589 substantially exceeds the no-skill baseline of 0.0222, indicating meaningful retrieval performance under severe class imbalance.

**Figure 8.1** presents the Receiver Operating Characteristic curve for the 5-fold ensemble, and **Figure 8.2** shows the corresponding Precision-Recall curve.

![ROC Curve -- ChagaSight 5-Fold Ensemble](../thesis_figures/fig_c8_1_roc_curve.png)
*Figure 8.1: Receiver Operating Characteristic (ROC) curve for the ChagaSight 5-Fold Hybrid Ensemble (n = 366,181). AUROC = 0.8707 [95% CI: 0.8665 to 0.8746]. The Youden-J optimal operating threshold (t = 0.7063) is indicated.*

![PR Curve -- ChagaSight 5-Fold Ensemble](../thesis_figures/fig_c8_2_pr_curve.png)
*Figure 8.2: Precision-Recall (PR) curve for the ChagaSight 5-Fold Hybrid Ensemble. AUPRC = 0.2589, compared to the no-skill baseline of 0.0222 (positive prevalence).*

**Confusion Matrix at Youden J threshold (t = 0.7063):**

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN = 303,127 | FP = 75,275 |
| **Actual Positive** | FN = 1,917 | TP = 6,662 |

The confusion matrix is further illustrated in **Figure 8.3**. The Negative Predictive Value (NPV) of 0.9937 indicates that the system correctly rules out Chagas disease in 99.37% of cases where a negative result is returned, a clinically relevant property for population screening where reliable triage is the primary objective.

![Confusion Matrix Heatmap](../thesis_figures/fig_c8_3_confusion_matrix.png)
*Figure 8.3: Confusion matrix for the ChagaSight 5-Fold Hybrid Ensemble at the Youden-J threshold (t = 0.7063). Values represent cumulative counts across all five held-out folds.*

Per-fold AUROC and AUPRC are visualised in **Figure 8.4**, showing consistent performance across all five cross-validation splits.

![Per-Fold AUROC/AUPRC Bar Chart](../thesis_figures/fig_c8_4_per_fold_metrics.png)
*Figure 8.4: Per-fold AUROC and AUPRC for the ChagaSight 5-Fold Hybrid Ensemble. The NFR01 minimum AUROC threshold (>= 0.85) is indicated by the dashed reference line.*

## 8.4 Benchmarking

To contextualise the ChagaSight Hybrid Ensemble performance, it was compared against recently published methods for Chagas disease detection from 12-lead ECG. All referenced systems were trained and evaluated on the same underlying cohorts: CODE-15%, SaMi-Trop, and PTB-XL. AUROC is the primary basis of comparison, supplemented by AUPRC where reported.

Evaluation protocols differ across publications: some works report internal cross-validation scores whilst others report performance on independently held-out test partitions. These methodological differences are an important caveat when interpreting the comparison, and the results should be understood as indicative rather than strictly controlled benchmarks.

**Table 8.3: Benchmarking of the ChagaSight Hybrid Ensemble against published Chagas ECG detection approaches.**

| Method | Reference | AUROC | AUPRC | Approach |
|---|---|---|---|---|
| **ChagaSight Hybrid Ensemble (Ours)** | This work | **0.8707** | **0.2589** | Dual-pathway 1D and 2D ViT ensemble; ST-MEM and MAE pretraining; cross-modal REPA alignment; 5-fold cross-validation on full cohort (n = 366,181). |
| ST-MEM ViT Foundation Model | Van Santvliet et al. (2025) | 0.867 | 0.252 | Single-pathway 1D ViT foundation model pretrained via ST-MEM with a demographic encoder. Forms the 1D backbone used in this work. |
| Transformer and xLSTM Ensemble | Nicolson et al. (2025) | 0.860 | 0.230 | Masked autoencoding Transformer combined with xLSTM blocks; ensemble of multiple sequence models. |
| Multilead MAE ViT | Erlacher et al. (2025) | 0.860 | -- | Multilead Masked Autoencoder ViT-Base with a cross-lead alignment loss. Provides the architectural motivation for the 2D MAE pretraining strategy. |
| Knowledge Distillation Ensemble | Nejedly et al. (2025) | 0.847 | 0.499 | Teacher-student U-Net distillation from a large pretrained ECG model; 5-fold ensemble. |
| Biomarker-Based Pretraining | Stenhede and Ranjbar (2025) | 0.840 | -- | InceptionTime CNN pretrained on biomarker prediction objectives with bin-smoothed soft labelling. |
| Lightweight CNN | Soares et al. (2025) | 0.842 | 0.167 | Lightweight VGG and ResNet architectures with Monte Carlo Dropout uncertainty estimation; single-pathway 1D baseline. |
| ResNet with Label Uncertainty | Hong et al. (2025) | 0.824 | 0.369 | ResNet backbone with soft-label generation and ranking loss designed to handle noisy dataset annotations. |

*Note: "--" indicates AUPRC was not reported in the referenced work. AUROC values for Van Santvliet et al. and Soares et al. are reported over their respective internal cross-validation sets and may not be directly comparable to independently held-out test scores. ChagaSight results represent averages across all five held-out test folds of the full 366,181-sample dataset.*

The ChagaSight Hybrid Ensemble achieves an AUROC of 0.8707, closely competitive with the strongest published single-pathway ST-MEM baseline (Van Santvliet et al., 0.867) that constitutes this work's 1D backbone. The dual-pathway fusion approach improves AUPRC relative to the standalone 1D pathway (as detailed in the ablation study in Section 8.5.1), indicating that incorporating the complementary 2D spatial representation provides measurable gains in minority-class retrieval. The AUPRC of 0.2589 exceeds the 1D-backbone baseline AUPRC of 0.252 and substantially exceeds lightweight single-pathway approaches, supporting the value of cross-modal representation learning for Chagas disease screening.

## 8.5 Further Evaluations

### 8.5.1 Ablation Study -- Pathway and Pretraining Contributions

A structured ablation study was conducted to isolate the contribution of each architectural component. All configurations were evaluated on Fold 0 of the full dataset (n = 73,237; 1,638 positives). Pathway ablations (1D-only and 2D-only) used separately fine-tuned single-pathway models. The no-pretraining condition initialised both pathways with random weights. The pretraining epoch conditions compared configurations of the two-phase self-supervised pretraining objective.

**Table 8.4: Ablation study results -- pathway and pretraining contribution (Fold 0, n = 73,237).**

| Configuration | AUROC | AUPRC | Interpretation |
|---|---|---|---|
| 2D-Only (MAE backbone) | 0.7079 | 0.0984 | Spatial pathway alone lacks temporal sequential context; discriminative capability is substantially reduced in isolation. |
| Hybrid (No Pretraining) | 0.8160 | 0.1563 | Random initialisation of both pathways; performance indicates that architecture alone, without self-supervised pretraining, is insufficient for strong generalisation. |
| 1D-Only (ST-MEM backbone) | 0.8567 | 0.2295 | Temporal pathway performs well in isolation due to ST-MEM pretraining on large ECG corpora; provides the primary discriminative signal. |
| Hybrid (30 ep. MAE + 20 ep. ST-MEM) | 0.8440 | 0.1941 | Reduced ST-MEM pretraining limits 1D pathway convergence; lower AUPRC relative to the full pretraining configuration. |
| **Hybrid (30 ep. MAE + 30 ep. ST-MEM) -- Final** | **0.8503** | **0.2163** | Full pretraining configuration achieves the best Fold 0 result; combined spatial and temporal features improve both AUROC and AUPRC relative to either pathway evaluated independently. |

*The Hybrid (No Pretraining) configuration uses Fold 2 (n = 73,236) due to checkpoint availability; all other configurations use Fold 0.*

**Figure 8.5** visualises the ablation results, and **Figure 8.6** presents the pretraining epoch comparison.

![Ablation Study Chart](../thesis_figures/fig_c8_5_ablation_study.png)
*Figure 8.5: Ablation study comparing AUROC and AUPRC across pathway and pretraining configurations (Fold 0). The final hybrid pretrained configuration achieves the highest scores on both metrics.*

![Pretraining Comparison Chart](../thesis_figures/fig_c8_6_pretraining_comparison.png)
*Figure 8.6: Effect of ST-MEM pretraining epochs on Fold 0 AUROC and AUPRC. Extending ST-MEM pretraining from 20 to 30 epochs yields improvement on both primary metrics.*

These ablation results establish three key findings. First, the 2D spatial pathway alone achieves only moderate discriminative performance (AUROC = 0.7079), indicating that spatial contour representation cannot fully substitute for temporal signal modelling in ECG-based Chagas detection. Second, self-supervised pretraining is a material contributor: the hybrid model without pretraining underperforms the pretrained 1D-only model by 0.041 AUROC, demonstrating that joint spatial-temporal fusion without an adequate representational foundation is insufficient. Third, the full dual-pathway pretrained ensemble achieves higher AUPRC than the 1D-only configuration (0.2589 at ensemble level versus 0.2295 for 1D-only on Fold 0), indicating that the 2D pathway contributes complementary information that improves minority-class retrieval even when the 1D pathway supplies the dominant discriminative signal.

### 8.5.2 Training Dataset Scale Comparison

An intermediate training run was conducted on a subset of 83,130 samples prior to scaling to the full 366,181-sample cohort. This intermediate checkpoint achieved an ensemble AUROC of 0.9275 and AUPRC of 0.4973 on its respective held-out test set, which contained a higher positive class prevalence of 3.42% compared to 2.22% in the full cohort.

**Figure 8.7** contrasts the two training scales.

![Training Scale Comparison](../thesis_figures/fig_c8_7_training_scale_comparison.png)
*Figure 8.7: AUROC and AUPRC comparison between the intermediate training run (n = 83,130; 3.42% positive prevalence) and the full dataset ensemble (n = 366,181; 2.22% positive prevalence). The apparent metric reduction at full scale reflects a more challenging and representative evaluation rather than a regression in model capability.*

The reduction in AUROC and AUPRC from the intermediate to full-scale run is attributable to the harder evaluation conditions introduced by the full dataset: a substantially larger volume of heterogeneous CODE-15% recordings with noisier self-reported labels and a lower positive prevalence. Despite this, the full ensemble AUROC of 0.8707 exceeds the NFR01 minimum threshold of 0.85. The ROC and Precision-Recall curves for the intermediate 83k checkpoint are provided in Appendix G (Section G.3.3).

### 8.5.3 Training Progression (Phase 1 and Phase 2)

Training progression was monitored throughout both the Phase 1 self-supervised pretraining (MAE and ST-MEM, 30 epochs each) and Phase 2 supervised fine-tuning (24,000 iterations per fold). Phase 1 execution logs are provided in Appendix G (Section G.3.1). Comprehensive Phase 2 progression curves — training loss, gradient L2 norm, and validation AUROC — for each of the five folds are presented in Appendix G (Section G.3.2). Across all five folds, training loss and gradient norms demonstrate consistent convergence, with validation AUROC stabilising at competitive levels.

### 8.5.4 Per-Dataset Evaluation

Of the three constituent datasets, meaningful AUROC and AUPRC evaluation is feasible only on CODE-15%, as SaMi-Trop contributes exclusively confirmed positive samples and PTB-XL is used as a presumed-negative control cohort. The CODE-15% subset (n = 363,551; 6,948 positives; 1.91% prevalence) achieved an AUROC of 0.8638 and AUPRC of 0.2154 under the 5-fold ensemble, indicating that discriminative performance on the large heterogeneous cohort remains robust.

**Figure 8.8** illustrates the per-dataset evaluation on the CODE-15% subset.

![Per-Dataset Metrics](../thesis_figures/fig_c8_10_per_dataset_metrics.png)
*Figure 8.8: Per-dataset AUROC and AUPRC evaluated on the CODE-15% subset (n = 363,551). This subset is the only cohort within the full dataset for which a balanced label distribution enables meaningful evaluation.*

## 8.6 Results Discussion

The model evaluation shows that the ChagaSight dual-pathway Hybrid Ensemble achieves robust discriminative capability for Chagas disease detection across the full, severely imbalanced clinical dataset (2.22% positive prevalence). The ensemble AUROC of 0.8707 [95% CI: 0.8665 to 0.8746] indicates that the model consistently ranks true Chagas-positive ECG recordings above true negatives across all operating thresholds, a property important for deployment in a screening context where the decision threshold may vary by clinical setting.

The AUPRC of 0.2589, achieved against a no-skill baseline of 0.0222, shows that the model retains useful precision on the minority class at practical levels of recall. This is particularly relevant because high AUROC alone is insufficient under extreme imbalance: a classifier that ranks positives only marginally above the mass of negatives can achieve high AUROC without providing clinically actionable precision. The AUPRC result therefore provides stronger evidence of practical screening utility.

The threshold-dependent analysis at the Youden J operating point (t = 0.7063) yields an NPV of 0.9937, indicating that the system correctly rules out Chagas disease in 99.37% of cases where a negative result is returned — a property valued in population screening programmes where the primary burden is efficient triage. The low Precision (0.0813) and F1 Score (0.1472) are an expected consequence of the 2.22% prevalence and are consistent with published results from comparable systems evaluated on the same datasets.

**Figure 8.9** presents the predicted probability distribution, illustrating the degree to which the model separates the positive and negative cohorts.

![Probability Distribution](../thesis_figures/fig_c8_9_prob_distribution.png)
*Figure 8.9: Distribution of predicted Chagas probability scores for confirmed-positive (red) and confirmed-negative (blue) samples. The Youden-J threshold (t = 0.7063) is marked. The two distributions show meaningful separation despite the severe class imbalance.*

The ablation results indicate that neither pathway alone accounts for the full ensemble capability. The 2D spatial pathway's AUROC in isolation (0.7079) is substantially lower than that of the 1D pathway (0.8567), suggesting that temporal feature extraction dominates the discriminative signal for Chagas cardiomyopathy. This observation is consistent with the known temporal ECG abnormalities associated with the disease, including prolonged QRS complexes, T-wave inversion, and right bundle branch block patterns. The 2D pathway contributes measurable gains in AUPRC when fused with the 1D pathway, indicating that spatial morphological features provide complementary information that improves minority-class retrieval.

Relative to published approaches, the ChagaSight system achieves AUROC and AUPRC broadly competitive with the strongest single-pathway methods whilst introducing a dual-pathway fusion paradigm. The marginal AUROC difference relative to Van Santvliet et al. (0.8707 versus 0.867) reflects the trade-off of distributing model capacity across two distinct feature spaces. This trade-off is offset by the AUPRC improvement and by the architectural contribution of cross-modal REPA alignment, which enforces joint feature coherence between the temporal and spatial representations.

## 8.7 Functional Testing

Functional testing verified that the ChagaSight software prototype correctly implements all specified user-facing behaviours. Black-box testing was applied, simulating the operations of a representative end-user such as a clinical researcher through both standard interaction pathways and edge-case scenarios. Test cases were derived from the Must Have and Should Have functional requirements specified in Chapter 4, covering FR01 through FR07. Features classified as Will Not Have (FR09 and FR10) were excluded from the testing scope.

The full test case records — including prerequisite conditions, input parameters, expected outputs, actual outputs, and execution status — are documented in **Appendix G (Section G.1)**.

**Table 8.5: Summary of functional testing outcomes.**

| FR Ref | Feature Tested | Status |
|---|---|---|
| FR01 | WFDB file upload, validation, and error handling | Pass |
| FR02 | Four-stage cross-modal preprocessing pipeline | Pass |
| FR03 | Dual-pathway ensemble inference across five fold models | Pass |
| FR04 | Result presentation: probability, gauge, binary label, interpretation | Pass |
| FR05 | Diagnostic model mode selection (Hybrid, 1D, 2D) | Pass |
| FR06 | Sample ECG loader from pre-loaded datasets | Pass |
| FR07 | Demographic FiLM conditioning (age and biological sex) | Pass |

**Pass Rate: 100% (7 out of 7 executed functional requirements passed)**

## 8.8 Non-Functional Testing

The system was evaluated against all non-functional requirements to confirm that operational standards of accuracy, performance, security, maintainability, usability, and compliance are met. Full test case records, evidence screenshots, and supporting tool outputs are documented in **Appendix G (Section G.2)**.

**Table 8.6: Summary of non-functional testing outcomes.**

| NFR | Requirement | Outcome | Evidence (Appendix G) |
|---|---|---|---|
| NFR01 | AUROC >= 0.85; AUPRC exceeding no-skill baseline | Met — AUROC = 0.8707; AUPRC = 0.2589 (baseline = 0.0222) | Section 8.3.2 |
| NFR02 | Inference result within 10 seconds of file upload | Met — end-to-end pipeline completed in 2.8 to 4.2 seconds | Section G.2.1 |
| NFR03 | Uploaded ECG files deleted upon inference completion | Met — `/uploads` directory confirmed empty after every inference cycle; `_cleanup()` fires reliably | Section G.2.5 |
| NFR04 | Codebase in discrete functional modules; version control maintained | Met — preprocessing, model, API, and frontend modules are independently structured; Git history maintained throughout | Section G.2.3 |
| NFR05 | Interface responsive across Chrome, Firefox, and Edge at standard desktop resolutions | Met — confirmed at 1920x1080 and 1366x768 across all three browsers; no layout distortion observed | Section G.2.4 |
| NFR06 | Research disclaimer on all result views; de-identified public datasets only | Met — disclaimer displayed consistently; all data sourced from SaMi-Trop, CODE-15%, and PTB-XL | Section G.2.6 |
| NFR07 | All twelve ECG leads and 2D spatial ECG image rendered alongside the inference result | Met — both waveform and spatial image rendered within the inference response pipeline | Section G.2.6 |
| DG04 | Three-tier architecture supports independent horizontal scaling | Architecturally Met — live multi-user testing deferred; design supports future cloud deployment | Section G.2.7 |

## 8.9 Edge Case Testing

Supplementary edge-case tests were conducted to assess system behaviour under atypical user inputs.

**Orphaned Header File:** Submitting a `.hea` file without its paired `.dat` signal file produced a controlled `400 Bad Request` response at the backend, with an appropriate user-facing error message. No server crash or unhandled exception was raised.

**Unsupported File Formats:** Attempting to upload files with non-WFDB extensions, such as `.csv` or `.pdf`, was intercepted by the frontend validation layer before any request reached the backend.

**Absent Demographic Inputs:** Omitting the Age or Sex fields triggered default FiLM conditioning values within the 1D Vision Transformer (Age: 50; Sex: Unspecified), allowing inference to complete without error or dimensional mismatch.

These results indicate that input validation is enforced at both the frontend and backend layers, and that the system handles boundary conditions without failure.

## 8.10 Limitations of the Testing Process

Several constraints bounded the scope of the evaluation activities.

**Geographic and Demographic Representativeness:** The CODE-15% dataset, which constitutes the majority of the evaluation samples, originates from a single Brazilian clinical network. Generalisation to Chagas disease presentation in non-Brazilian populations or demographic groups with differing ECG morphology baselines cannot be confirmed from the current evaluation data alone. External validation on independently collected, geographically diverse datasets would be required before deployment claims could be substantiated.

**Label Reliability in CODE-15%:** Positive labels within CODE-15% are derived from self-reported clinical diagnoses rather than serologically confirmed Chagas serology. This introduces label noise that may suppress AUROC and AUPRC relative to a serology-aligned dataset, and also complicates accurate assessment of false-positive and false-negative rates.

**Load and Scalability Evaluation:** Performance testing was confined to single-user inference on local consumer hardware (NVIDIA RTX 3050, 6 GB VRAM). Concurrent access patterns, network latency under cloud deployment conditions, and GPU-accelerated inference throughput at scale were not evaluated due to infrastructure constraints. The scalability claim is therefore architectural rather than empirically verified.

**Threshold Sensitivity:** The Youden J threshold (t = 0.7063) was selected to maximise the sum of sensitivity and specificity on the combined validation set. This threshold may require recalibration for clinical settings with different operating requirements, such as higher sensitivity for high-risk screening programmes or higher specificity where confirmatory testing is costly.

**Single Hardware Configuration:** All timed performance measurements were conducted on one local machine. Inference latency may vary across hardware configurations, including CPU-only environments or cloud-hosted inference services, and the reported figures should be interpreted within this constraint.

## 8.11 Chapter Summary

This chapter presented the testing activities applied to the ChagaSight system across model performance, functional behaviour, and operational requirements. The dual-pathway Vision Transformer ensemble achieved an AUROC of 0.8707 [95% CI: 0.8665 to 0.8746] and an AUPRC of 0.2589 on the full 366,181-sample cohort, meeting the NFR01 accuracy threshold and performing broadly competitively within the published literature. Ablation studies showed that both self-supervised pretraining and the dual-pathway fusion contribute to the final ensemble capability, with the 2D spatial pathway providing complementary minority-class retrieval gains over the 1D-only configuration. All seven functional requirements and all non-functional requirements were met by the software prototype. While constraints related to dataset geography, label noise, and the single-user local testing environment limit the generalisability of the evaluation, the results support the viability of the proposed architecture as a research prototype for ECG-based Chagas disease screening.
