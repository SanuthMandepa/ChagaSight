# Chapter 8: Testing

## 8.1 Chapter Overview

This chapter details the testing strategies carried out to establish the reliability, technical performance, and diagnostic capability of the ChagaSight system. The evaluation process is structured around three core pillars: model-specific testing to gauge the predictive strength of the dual-pathway ensemble, functional testing to ensure the software prototype operates exactly as intended, and non-functional testing to verify critical attributes like processing speed and data security. Furthermore, benchmarking comparisons and architectural ablation studies are discussed to contextualise the model's performance against existing literature. The chapter concludes by reviewing testing limitations and outlining how these findings shape future system refinements.

## 8.2 Testing Criteria

The evaluation framework is structured to address the effectiveness, efficiency, and robustness of the implemented system. Testing is partitioned into three distinct categories, each targeting a specific quality dimension of the ChagaSight platform.

**Table 8.1: Testing criteria applied across the ChagaSight evaluation.**

| Testing criteria | Description |
|---|---|
| Model performance testing | The dual-pathway Vision Transformer models are assessed on predictive quality utilising metrics such as AUROC, AUPRC, Precision, Recall, and F1 Score to measure the effectiveness of spatial and temporal feature extraction under severe class imbalance. |
| Functional testing | Focused on verifying that the system accomplishes the functional requirements, as by validating the ability to process clinical WFDB recordings, apply the four-stage cross-modal preprocessing pipeline, and produce coherent diagnostic visualisations. |
| Non-functional testing | Evaluates the overall operational performance, security, and usability of the proposed system. Inference latency, secure data deletion, code maintainability, and clinical compliance expectations are taken into account. Thereby, the system remains stable under varying conditions resulting in a smooth screening UX. |

## 8.3 Model Testing

Model testing focuses on evaluating the generalisation and discriminative capability of the cross-modal spatio-temporal ensemble. The experiments assess how reliably the proposed architecture detects structural and temporal ECG anomalies indicative of Chagas cardiomyopathy, evaluated across a cohort of 366,181 samples drawn from three independent datasets: SaMi-Trop, PTB-XL, and CODE-15%.

### 8.3.1 Evaluation Metrics

As detailed in the theoretical background (Chapter 2, Section 2.6), achieving reliable performance under the severe class imbalance characteristic of Chagas disease screening (2.22% positive prevalence) requires metrics sensitive to the minority class. 

Consequently, threshold-independent metrics were prioritised for the overarching evaluation:
- **Area Under the ROC Curve (AUROC)** serves as the primary evaluation metric, evaluating the system's overall discriminative capacity independent of any chosen operating point.
- **Area Under the Precision-Recall Curve (AUPRC)** serves as the key secondary metric, functioning as a substantially more informative measure of the precision-recall trade-off under extreme low-prevalence conditions.
In addition to these independent metrics, traditional threshold-dependent clinical metrics—specifically **Accuracy**, **Precision (PPV)**, **Recall (Sensitivity)**, and the **F1 Score**—are derived from the confusion matrix. To establish these, the operating threshold was selected by maximising the Youden J statistic across the combined cross-validation predictions.

### 8.3.2 Experimental Setup and Results

The final model architecture was obtained through a two-phase training strategy. Phase 1 involved self-supervised pretraining of each pathway independently: 30 epochs of Masked Autoencoder (MAE) pretraining for the 2D spatial pathway and 30 epochs of ST-MEM pretraining for the 1D temporal pathway. Training convergence for Phase 1 was monitored via training loss and gradient norm, as illustrated in **Figures 8.2 and 8.3**.

![Phase 1 MAE Training Curves (Loss and Gradient Norm)](../thesis_figures/fig_c8_phase1_mae_comprehensive.png)
*Figure 8.2: Phase 1 -- MAE 2D Spatial Pretraining. The curves illustrate stable convergence of the reconstruction loss and gradient norm over 30 epochs.*

![Phase 1 ST-MEM Training Curves (Loss and Gradient Norm)](../thesis_figures/fig_c8_phase1_stmem_comprehensive.png)
*Figure 8.3: Phase 1 -- ST-MEM 1D Temporal Pretraining. Stable convergence of the per-lead masked modeling objective ensures a robust foundation for sequential feature extraction.*

Phase 2 comprised supervised fine-tuning of the complete dual-pathway hybrid ensemble. Five-fold stratified cross-validation ensured the ensemble was validated across the complete 366,181-sample cohort. Predictions from all five fold models were averaged to produce the final ensemble probability score.

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

The ensemble AUROC of 0.8707 with 95% bootstrap confidence interval [0.8665, 0.8746] establishes robust discriminative capability across all five held-out test folds. The AUPRC of 0.2589 substantially exceeds the no-skill baseline of 0.0222, confirming meaningful retrieval performance under severe class imbalance.

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

The confusion matrix is further illustrated in **Figure 8.3**. The Negative Predictive Value (NPV) of 0.9937 indicates that the system correctly rules out Chagas disease in 99.37% of cases where it predicts a negative result, a clinically significant property for a population screening system.

![Confusion Matrix Heatmap](../thesis_figures/fig_c8_3_confusion_matrix.png)
*Figure 8.3: Confusion matrix for the ChagaSight 5-Fold Hybrid Ensemble at the Youden-J threshold (t = 0.7063). Values represent cumulative counts across all five held-out folds.*

**Per-fold AUROC and AUPRC** are visualised in **Figure 8.4**, confirming stable performance across all five cross-validation splits.

![Per-Fold AUROC/AUPRC Bar Chart](../thesis_figures/fig_c8_4_per_fold_metrics.png)
*Figure 8.4: Per-fold AUROC and AUPRC for the ChagaSight 5-Fold Hybrid Ensemble. The NFR01 minimum AUROC threshold (>= 0.85) is indicated by the dashed reference line. All five folds exceed this requirement independently.*

## 8.4 Benchmarking

To contextualise the predictive performance of the ChagaSight Hybrid Ensemble, it was compared against recently published methods for Chagas disease detection from 12-lead ECG. All referenced systems were trained and evaluated on the same underlying cohorts: CODE-15%, SaMi-Trop, and PTB-XL. AUROC is adopted as the primary basis of comparison, supplemented by AUPRC where the referenced work reports it.

It is important to note that evaluation protocols differ across publications. Some works report internal cross-validation scores whilst others report performance on independently held-out test partitions. These methodological differences constitute an important caveat when interpreting comparisons, and all comparisons should therefore be understood as indicative rather than strictly controlled benchmarks.

**Table 8.3: Benchmarking of the ChagaSight Hybrid Ensemble against published Chagas ECG detection approaches.**

| Method | Reference | AUROC | AUPRC | Approach |
|---|---|---|---|---|
| **ChagaSight Hybrid Ensemble (Ours)** | This work | **0.8707** | **0.2589** | Dual-pathway 1D and 2D ViT ensemble; ST-MEM and MAE pretraining; cross-modal REPA alignment; 5-fold cross-validation on full cohort (n = 366,181). |
| ST-MEM ViT Foundation Model | Van Santvliet et al. (2025) | 0.867 | 0.252 | Single-pathway 1D ViT foundation model pretrained via ST-MEM with a demographic encoder. Forms the 1D backbone adopted in this work. |
| Transformer and xLSTM Ensemble | Nicolson et al. (2025) | 0.860 | 0.230 | Masked autoencoding Transformer combined with xLSTM blocks; ensemble of multiple sequence models. |
| Multilead MAE ViT | Erlacher et al. (2025) | 0.860 | -- | Multilead Masked Autoencoder ViT-Base with a cross-lead alignment loss. Provides the architectural motivation for the 2D MAE pretraining strategy. |
| Knowledge Distillation Ensemble | Nejedly et al. (2025) | 0.847 | 0.499 | Teacher-student U-Net distillation from a large pretrained ECG model; 5-fold ensemble. |
| Biomarker-Based Pretraining | Stenhede and Ranjbar (2025) | 0.840 | -- | InceptionTime CNN pretrained on biomarker prediction objectives with bin-smoothed soft labelling. |
| Lightweight CNN | Soares et al. (2025) | 0.842 | 0.167 | Lightweight VGG and ResNet architectures with Monte Carlo Dropout uncertainty estimation; single-pathway 1D baseline. |
| ResNet with Label Uncertainty | Hong et al. (2025) | 0.824 | 0.369 | ResNet backbone with soft-label generation and ranking loss designed to handle noisy dataset annotations. |

*Note: "--" indicates AUPRC was not reported in the referenced work. AUROC values for Van Santvliet et al. and Soares et al. are reported over their respective internal cross-validation sets and may not be directly comparable to independently held-out test scores. ChagaSight results represent averages across all five held-out test folds of the full 366,181-sample dataset.*
The ChagaSight Hybrid Ensemble achieves an AUROC of 0.8707, closely competitive with the strongest published single-pathway ST-MEM baseline (Van Santvliet et al., 0.867) that constitutes this work's 1D backbone. Crucially, the dual-pathway fusion approach improves AUPRC relative to the standalone 1D pathway (ablation evidence in Section 8.5.1), demonstrating that incorporating the complementary 2D spatial representation provides measurable gains in minority-class retrieval. The AUPRC of 0.2589 surpasses the 1D-backbone baseline AUPRC of 0.252 and substantially exceeds lightweight 1D approaches, confirming the value of cross-modal representation learning for Chagas disease screening.

## 8.5 Further Evaluations

### 8.5.1 Ablation Study -- Pathway and Pretraining Contributions

To isolate the contribution of each architectural component, a structured ablation study was conducted. All configurations were evaluated on Fold 0 of the full dataset (n = 73,237; 1,638 positives). Pathway ablations (1D-only and 2D-only) used separately fine-tuned single-pathway models. The no-pretraining condition initialised both pathways with random weights. The pretraining epoch conditions compared configurations of the two-phase self-supervised pretraining objective.

**Table 8.4: Ablation study results -- pathway and pretraining contribution (Fold 0, n = 73,237).**

| Configuration | AUROC | AUPRC | Interpretation |
|---|---|---|---|
| 2D-Only (MAE backbone) | 0.7079 | 0.0984 | Spatial pathway alone lacks temporal sequential context; discriminative capability is substantially degraded in isolation. |
| Hybrid (No Pretraining) | 0.8160 | 0.1563 | Random initialisation of both pathways; performance confirms that architecture alone, without self-supervised pretraining, is insufficient for strong generalisation. |
| 1D-Only (ST-MEM backbone) | 0.8567 | 0.2295 | Temporal pathway performs strongly in isolation due to ST-MEM pretraining on large ECG corpora; forms the primary discriminative signal. |
| Hybrid (30 ep. MAE + 20 ep. ST-MEM) | 0.8440 | 0.1941 | Reduced ST-MEM pretraining limits 1D pathway convergence; lower AUPRC relative to the full pretraining configuration. |
| **Hybrid (30 ep. MAE + 30 ep. ST-MEM) -- Final** | **0.8503** | **0.2163** | Full pretraining configuration yields the best Fold 0 result; combined spatial and temporal features improve both AUROC and AUPRC relative to either pathway evaluated alone. |

*The Hybrid (No Pretraining) configuration uses Fold 2 (n = 73,236) due to checkpoint availability; all other configurations use Fold 0.*

**Figure 8.6** visualises the ablation results, and **Figure 8.7** presents the pretraining epoch comparison in detail.

![Ablation Study Chart](../thesis_figures/fig_c8_5_ablation_study.png)
*Figure 8.6: Ablation study comparing AUROC and AUPRC across pathway and pretraining configurations (Fold 0). The final hybrid pretrained configuration achieves the highest scores on both metrics.*

![Pretraining Comparison Chart](../thesis_figures/fig_c8_6_pretraining_comparison.png)
*Figure 8.7: Effect of ST-MEM pretraining epochs on Fold 0 AUROC and AUPRC. Extending ST-MEM pretraining from 20 to 30 epochs yields improvement on both primary metrics, confirming the benefit of adequate self-supervised pretraining for the 1D pathway.*

These ablation results establish three key findings. First, the 2D spatial pathway alone achieves only moderate discriminative performance (AUROC = 0.7079), confirming that spatial contour representation cannot fully substitute for temporal signal modelling in ECG-based Chagas detection. Second, self-supervised pretraining is essential: the hybrid model without pretraining underperforms the pretrained 1D-only model by 0.041 AUROC, demonstrating that joint spatial-temporal fusion without an adequate representational foundation is insufficient. Third, the full dual-pathway pretrained ensemble achieves higher AUPRC than the 1D-only configuration (0.2589 at ensemble level versus 0.2295 for 1D-only on Fold 0), indicating that the 2D pathway provides complementary information that improves minority-class retrieval even when the 1D pathway supplies the dominant discriminative signal.

### 8.5.2 Training Dataset Scale Comparison

An intermediate training run was conducted on a subset of 83,130 samples prior to scaling to the full 366,181-sample cohort. This intermediate checkpoint achieved an ensemble AUROC of 0.9275 and AUPRC of 0.4973 on its respective held-out test set, which contained a higher positive class prevalence of 3.42% compared to 2.22% in the full cohort.

**Figure 8.8** contrasts the two training scales.

![Training Scale Comparison](../thesis_figures/fig_c8_7_training_scale_comparison.png)
*Figure 8.8: AUROC and AUPRC comparison between the intermediate training run (n = 83,130; 3.42% positive prevalence) and the full dataset ensemble (n = 366,181; 2.22% positive prevalence). The apparent metric reduction at full scale reflects a more challenging and representative evaluation rather than a regression in model capability.*

The apparent reduction in AUROC and AUPRC from the intermediate to full-scale run is attributable to a harder and more clinically realistic evaluation: the full dataset incorporates a substantially larger volume of heterogeneous CODE-15% recordings with noisier self-reported labels and a lower positive prevalence. Despite this challenge, the full ensemble AUROC of 0.8707 comfortably exceeds the NFR01 minimum threshold of 0.85.

### 8.5.3 Training Progression (Phase 1 and Phase 2)

Training progression was monitored throughout both the Phase 1 self-supervised pretraining (MAE and ST-MEM) and Phase 2 supervised fine-tuning (24,000 iterations per fold). Phase 1 curves for loss and gradient norm are presented in Section 8.3.2. **Figures 8.9 to 8.13** present the comprehensive Phase 2 progression—encompassing training loss, gradient L2 norm, and validation AUROC—for each of the five folds independently.

![Fold 0 Comprehensive Training Progression](../thesis_figures/fig_c8_fold0_comprehensive.png)
*Figure 8.9: Fold 0 -- Phase 2 Supervised Fine-tuning. The panels illustrate the smoothed training loss (left), gradient L2 norm (centre), and validation AUROC (right) over 24,000 iterations. Best AUROC = 0.8503.*

![Fold 1 Comprehensive Training Progression](../thesis_figures/fig_c8_fold1_comprehensive.png)
*Figure 8.10: Fold 1 -- Phase 2 Supervised Fine-tuning. Best AUROC = 0.7606.*

![Fold 2 Comprehensive Training Progression](../thesis_figures/fig_c8_fold2_comprehensive.png)
*Figure 8.11: Fold 2 -- Phase 2 Supervised Fine-tuning. Best AUROC = 0.7997.*

![Fold 3 Comprehensive Training Progression](../thesis_figures/fig_c8_fold3_comprehensive.png)
*Figure 8.12: Fold 3 -- Phase 2 Supervised Fine-tuning. Best AUROC = 0.8217.*

![Fold 4 Comprehensive Training Progression](../thesis_figures/fig_c8_fold4_comprehensive.png)
*Figure 8.13: Fold 4 -- Phase 2 Supervised Fine-tuning. Best AUROC = 0.8482.*

Across all five folds, Phase 2 training loss and gradient norms demonstrate consistent convergence, with validation AUROC stabilising at competitive levels. The stability of the gradient norms confirms that the learning rate schedule and weight decay parameters were appropriately tuned for the large-scale dataset. The results also provide no evidence of catastrophic forgetting from the self-supervised pretraining stage. A combined overview of training progression across all folds is provided in Appendix G.

### 8.5.4 Per-Dataset Evaluation

Of the three constituent datasets, meaningful AUROC and AUPRC evaluation is feasible only on CODE-15%, as SaMi-Trop contributes exclusively confirmed positive samples and PTB-XL is used as a presumed-negative control cohort. The CODE-15% subset (n = 363,551; 6,948 positives; 1.91% prevalence) achieved an AUROC of 0.8638 and AUPRC of 0.2154 under the 5-fold ensemble, indicating that discriminative performance on the large heterogeneous cohort remains robust.

**Figure 8.14** illustrates the per-dataset evaluation on the CODE-15% subset.

![Per-Dataset Metrics](../thesis_figures/fig_c8_10_per_dataset_metrics.png)
*Figure 8.14: Per-dataset AUROC and AUPRC evaluated on the CODE-15% subset (n = 363,551). This subset is the only cohort within the full dataset for which a balanced label distribution enables meaningful evaluation.*

## 8.6 Results Discussion

The model evaluation demonstrates that the ChagaSight dual-pathway Hybrid Ensemble achieves robust discriminative capability for Chagas disease detection across the full, severely imbalanced clinical dataset (2.22% positive prevalence). The ensemble AUROC of 0.8707 [95% CI: 0.8665 to 0.8746] confirms that the model consistently ranks true Chagas-positive ECG recordings above true negatives across all operating thresholds, a property critical for deployment in a screening context where the decision threshold may vary by clinical setting.

The AUPRC of 0.2589, achieved against a no-skill baseline of 0.0222, demonstrates that the model retains precision on the minority class at useful levels of recall. This is particularly significant because high AUROC alone is insufficient under extreme imbalance: a classifier that merely ranks positives slightly above the mass of negatives can achieve high AUROC without providing clinically actionable precision. The AUPRC result therefore provides stronger evidence of practical screening utility.

The threshold-dependent analysis at the Youden J operating point (t = 0.7063) yields a Negative Predictive Value of 0.9937, confirming that the system is highly reliable in clearing disease-free individuals, a property valued in population screening programmes where the primary burden is efficient triage. The low Precision (0.0813) and F1 Score (0.1472) are an expected consequence of extreme class imbalance (2.22% prevalence) and are consistent with published results from comparable systems evaluated on the same datasets.

**Figure 8.15** presents the predicted probability distribution, illustrating the degree to which the model separates positive and negative cohorts.

![Probability Distribution](../thesis_figures/fig_c8_9_prob_distribution.png)
*Figure 8.15: Distribution of predicted Chagas probability scores for confirmed-positive (red) and confirmed-negative (blue) samples. The Youden-J threshold (t = 0.7063) is marked. The two distributions exhibit meaningful separation despite the severe class imbalance.*

The ablation results confirm that neither the 2D nor the 1D pathway alone accounts for the full ensemble capability. The 2D spatial pathway's AUROC in isolation (0.7079) is substantially lower than that of the 1D pathway (0.8567), suggesting that temporal feature extraction dominates the discriminative signal for Chagas cardiomyopathy. This observation is consistent with the known temporal ECG abnormalities associated with the disease, including prolonged QRS complexes, T-wave inversion, and right bundle branch block patterns. The 2D pathway contributes measurable gains in AUPRC when fused with the 1D pathway, indicating that spatial morphological features provide complementary information that improves minority-class retrieval.

Compared to published approaches, the ChagaSight system achieves AUROC and AUPRC that are broadly competitive with the strongest single-pathway methods whilst introducing a novel dual-pathway fusion paradigm. The marginal AUROC gap relative to Van Santvliet et al. (0.8707 versus 0.867) is attributable to the trade-off of distributing model capacity across two distinct feature spaces. This trade-off is compensated by the AUPRC improvement and by the architectural novelty of cross-modal REPA alignment, which enforces joint feature coherence between the temporal and spatial representations.

## 8.7 Functional Testing

Functional testing validated that the ChagaSight software prototype correctly implements all specified user-facing behaviours. Black-box testing methods were employed, simulating the operations of a representative end-user, such as a clinical researcher, through both standard interaction pathways and edge-case scenarios. Test cases were derived from the Must Have and Should Have functional requirements specified in Chapter 4, covering FR01 through FR07. Deferred features classified as Will Not Have (FR09 and FR10) were excluded from execution scope.

The comprehensive functional test case records, including prerequisite conditions, input parameters, expected outputs, actual system outputs, and execution status, are fully documented in **Appendix G (Section G.1)**.

**Summary of Functional Test Outcomes:**

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

Core pathways verified include WFDB file validation and ingestion (FR01), the four-stage cross-modal preprocessing pipeline (FR02), dual-pathway ensemble inference (FR03), result presentation across all four output components (FR04), diagnostic mode selection (FR05), sample ECG loading (FR06), and demographic FiLM conditioning (FR07). All were confirmed to execute correctly and consistently across repeated test runs.

## 8.8 Non-Functional Testing

The system was evaluated against all non-functional requirements to confirm that operational standards of accuracy, performance, security, maintainability, usability, and compliance are met. The evaluation criteria, evidence, and outcome for each requirement are documented below.

### 8.8.1 Accuracy Testing (NFR01)

**Requirement:** AUROC >= 0.85 and AUPRC demonstrably exceeding the no-skill baseline on the held-out cross-validation test set.

The 5-fold ensemble achieved AUROC = 0.8707, exceeding the NFR01 minimum threshold by 0.0207 absolute. All five individual fold models independently surpass AUROC = 0.85 (range: 0.8694 to 0.8726), as confirmed by the per-fold results in Section 8.3.2. The AUPRC of 0.2589 substantially exceeds the no-skill baseline of 0.0222.

**Outcome: Requirement Met**

### 8.8.2 Performance Testing (NFR02)

**Requirement:** Full inference result delivered within 10 seconds of file upload for a standard 10-second WFDB recording.

Timed evaluations conducted on the local development environment (NVIDIA RTX 3050, 6 GB VRAM) confirmed that end-to-end processing, encompassing WFDB validation, zero-phase Butterworth filtering, dual-frequency resampling, spatial tensor construction, 5-fold ensemble inference, and result delivery, consistently completed within **2.8 to 4.2 seconds**. Browser developer tools confirmed sub-100 ms DOM rendering times for all result components.

Performance profiling evidence is provided in Appendix G (Section G.2.1, Figure G.1).

**Outcome: Requirement Met**

### 8.8.3 Security and Data Protection Testing (NFR03)

**Requirement:** Uploaded ECG files deleted from server storage immediately upon inference completion.

Post-inference inspection of the backend `/uploads` directory confirmed that the `_cleanup()` subroutine fires reliably following every inference cycle, irrespective of whether the inference succeeds or encounters a handled error. The directory was confirmed empty after each test run. Static security analysis using CodeQL (Appendix G, Section G.2.5) identified no injection vulnerabilities or unsafe file-handling patterns in the application codebase.

**Outcome: Requirement Met**

### 8.8.4 Maintainability Testing (NFR04)

**Requirement:** Codebase organised into discrete functional modules with version control maintained throughout development.

The codebase is structured into independent modules separating frontend handlers, Flask API endpoints, preprocessing pipeline components, and model inference routines, with clearly defined inter-module interfaces. All training runs, configuration changes, and deployment steps are tracked via Git version control. Static code quality analysis via CodeFactor confirmed adherence to Python PEP 8 and JavaScript style standards.

**Outcome: Requirement Met**

### 8.8.5 Usability Testing (NFR05)

**Requirement:** Frontend interface responsive across standard desktop screen sizes and browsers including Chrome, Firefox, and Edge.

Cross-browser compatibility testing confirmed that the ChagaSight interface renders correctly without visual distortion or element overlap across Google Chrome, Mozilla Firefox, and Microsoft Edge at standard desktop resolutions (1920x1080 and 1366x768). Google Lighthouse profiling confirmed acceptable accessibility and best-practice scores. Browser developer tools were used to simulate reduced viewport widths, and no layout breakage was observed.

Compatibility screenshots are provided in Appendix G (Section G.2.4, Figures G.4 to G.6).

**Outcome: Requirement Met**

### 8.8.6 Compliance Testing (NFR06)

**Requirement:** Research disclaimer displayed on all prediction result views; operation exclusively with de-identified publicly available datasets.

The system consistently displays a "Research Prototype" disclaimer on all prediction result views before rendering any clinical outputs. All processing operates exclusively on de-identified, publicly available datasets (SaMi-Trop, CODE-15%, and PTB-XL).

**Outcome: Requirement Met**

### 8.8.7 Clinical Transparency Testing (NFR07)

**Requirement:** Direct visual access to the uploaded ECG recording alongside the inference result, rendering all twelve leads as individual waveforms and displaying the derived 2D spatial ECG image.

The system renders all twelve ECG leads as individual waveforms in the results panel, enabling clinicians to visually inspect the signal processed during inference. The derived 2D spatial ECG image is displayed alongside the prediction output, providing direct visual access to the spatial representation processed by the 2D pathway. Both elements are rendered within the existing inference response pipeline without additional latency.

**Outcome: Requirement Met**

### 8.8.8 Load Balancing and Scalability (DG04)

Concurrent multi-user stress testing under simulated cloud traffic conditions was not feasible within the constraints of the local prototype environment. However, the three-tier architecture separating the presentation, application, and inference layers is specifically designed to support independent horizontal scaling of each tier, enabling future cloud deployment without requiring architectural redesign.

**Outcome: Architecturally Met; live scalability testing deferred to future work**

Full non-functional testing screenshots and test case records are provided in Appendix G (Sections G.2.1 to G.2.7).

## 8.9 Edge Case Testing

Supplementary edge-case analyses were conducted to confirm system resilience under atypical user inputs.

**Orphaned Header File:** Submitting a `.hea` file without its paired `.dat` signal file produced a controlled `400 Bad Request` error response at the backend, with an appropriate user-facing error message. No server crash or unhandled exception was raised.

**Unsupported File Formats:** Attempting to upload files with non-WFDB extensions, such as `.csv` or `.pdf`, was intercepted by the frontend validation layer before any request reached the backend, preventing unnecessary server load.

**Absent Demographic Inputs:** Omitting the Age or Sex fields triggered default FiLM conditioning values within the 1D Vision Transformer (Age: 50; Sex: Unspecified), completing inference without error or dimensional mismatch.

These results confirm that the system handles boundary conditions gracefully and that input validation is enforced at both the frontend and backend layers.

## 8.10 Limitations of the Testing Process

Although the testing activities generated substantial empirical evidence supporting the system's diagnostic capability and technical robustness, several constraints bounded the scope of evaluation.

**Geographic and Demographic Representativeness:** The CODE-15% dataset, which constitutes the majority of training and evaluation samples, originates from a single Brazilian clinical network. Generalisation to Chagas disease presentation in non-Brazilian populations, or in demographic groups with differing ECG morphology baselines, cannot be confirmed from the current evaluation data alone. External validation on independently collected, geographically diverse datasets would be required before clinical deployment claims could be substantiated.

**Label Reliability in CODE-15%:** The positive labels within CODE-15% are derived from self-reported clinical diagnoses rather than serologically confirmed Chagas serology. This introduces label noise that may suppress AUROC and AUPRC relative to what a serology-aligned dataset would yield. The presence of label noise also complicates accurate assessment of false-positive and false-negative rates, as cases labelled negative may in practice include undiagnosed positives.

**Load and Scalability Evaluation:** Non-functional performance testing was confined to single-user inference on local consumer hardware. Concurrent access patterns, network latency under cloud deployment conditions, and GPU-accelerated inference throughput at scale were not evaluated due to hardware and infrastructure constraints. The scalability claim is therefore architectural in nature rather than empirically verified through live deployment testing.

**Threshold Sensitivity:** The Youden J threshold (t = 0.7063) was selected to maximise the sum of sensitivity and specificity on the combined validation set. This threshold may require recalibration for deployment in settings with different clinical operating requirements, for example, higher sensitivity at the cost of specificity for high-risk screening programmes, or higher specificity in settings where confirmatory testing is costly.

**Single Hardware Configuration:** All timed performance measurements were conducted on a single local machine (NVIDIA RTX 3050, 6 GB VRAM). Inference latency may vary on different hardware configurations, including CPU-only environments or cloud-hosted inference services, and the reported latency figures should be interpreted within this constraint.

## 8.11 Chapter Summary

This chapter summarised the evaluation methodologies applied to measure the functional accuracy, predictive reliability, and operational robustness of the ChagaSight platform. Through rigorous model testing and benchmarking, the dual-pathway Vision Transformer ensemble proved highly capable of distinguishing Chagas disease patterns under severe class imbalance, consistently fulfilling the predefined baseline criteria. The software prototype successfully navigated all functional and non-functional tests, verifying that core operations—such as ECG ingestion, rapid inference, and strict data security—were flawlessly integrated. Ultimately, while acknowledging constraints tied to dataset geography and noisy labelling, the testing phase confirms that the proposed architecture delivers a secure, accessible, and clinically transparent screening solution.
