# APPENDIX G — Testing Evidence

## G.1 Functional Testing Test Cases

All functional test cases were executed against the deployed ChagaSight prototype. The table below records the prerequisite state, input data, expected output, actual output, and execution status for each test case derived from the Must Have and Should Have functional requirements (FR01–FR07). Deferred Will Not Have requirements (FR09, FR10) were excluded.

**Table G.1: Functional Testing Test Cases.**

| FR Ref | Test Case ID | Objective | Description | Pre-requisites | Input Data | Expected Result | Actual Result | Status | Notes |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **FR01** | TC-FR01-01 | Verify successful upload of valid WFDB file pair | (1) Navigate to application UI. (2) Click file upload. (3) Select matching `.hea` and `.dat` files. (4) Confirm upload. | Application backend running. | `record_100.hea` and `record_100.dat` | Files accepted into temporary storage; filename displayed on dashboard. | Files uploaded successfully; filename displayed correctly. | **Pass** | |
| **FR01** | TC-FR01-02 | Verify rejection of unmatched WFDB files | (1) Navigate to file upload. (2) Select only a `.hea` file without its corresponding `.dat` file. (3) Submit. | Application deployed. | `record_100.hea` (standalone) | System rejects upload; user-facing error message displayed. | `400 Bad Request` returned cleanly; error displayed to user. | **Pass** | |
| **FR01** | TC-FR01-03 | Verify rejection of unsupported file formats | (1) Navigate to file upload. (2) Attempt to submit a `.pdf` or `.csv` file. | Application deployed. | `lab_results.pdf` | Frontend validation blocks submission; unsupported format error displayed. | Upload blocked at frontend before backend request is made. | **Pass** | |
| **FR02** | TC-FR02-01 | Verify four-stage preprocessing pipeline | (1) Upload valid WFDB pair. (2) Click "Analyse ECG". | Valid WFDB file pair uploaded. | Raw WFDB pair | Pipeline executes bandpass filtering (0.5–40 Hz), 100 Hz and 500 Hz resampling, per-lead Z-score normalisation, and WCT re-referencing. Outputs `(1, 12, 1000)` and `(1, 3, 24, 2048)` tensors. | Both tensors constructed successfully; confirmed via backend console logs. | **Pass** | Validated via backend console output. |
| **FR03** | TC-FR03-01 | Verify dual-pathway ensemble inference | (1) Select "Hybrid Ensemble". (2) Click "Analyse". | Preprocessing completed; all five fold model checkpoints loaded. | Dual ECG tensors `(1,12,1000)` and `(1,3,24,2048)` | 1D and 2D models perform parallel inference; fold probabilities averaged to a single ensemble score. | Inference completed across all five active fold checkpoints; ensemble probability returned. | **Pass** | |
| **FR04** | TC-FR04-01 | Verify result presentation components | (1) Wait for inference completion. (2) Inspect UI result panel. | Inference successful. | Predicted probability array | UI renders: (a) probability percentage, (b) visual gauge, (c) binary classification label (Low/High Risk), (d) plain-language interpretation string. | All four result components displayed correctly and proportionally. | **Pass** | |
| **FR05** | TC-FR05-01 | Verify diagnostic mode selection | (1) Open model selector. (2) Select "1D Signal Model". (3) Run analysis. | All model checkpoints instantiated at startup. | Raw WFDB pair | System performs inference exclusively via the 1D temporal pathway; 2D pathway not invoked. | Backend logs confirm 2D MAE model bypassed; 1D-only result returned. | **Pass** | |
| **FR06** | TC-FR06-01 | Verify sample ECG loader | (1) Open "Load Sample" menu. (2) Select a SaMi-Trop patient entry. | Pre-loaded sample files present in `/data` directory. | UI click event | System auto-loads the selected internal sample file pair without requiring manual upload. | Files loaded instantly into the active processing slot. | **Pass** | |
| **FR07** | TC-FR07-01 | Verify demographic FiLM conditioning | (1) Locate the demographics input form. (2) Enter Age: 60, Sex: Male. (3) Run analysis. | User on the main application screen. | Age: 60, Sex: Male | Values parsed (age normalised to 0.60) and passed to FiLM conditioning layers within the 1D Vision Transformer. | Values integrated without dimensional mismatch; inference completes correctly. | **Pass** | |

**Functional Testing Pass Rate: 100% (7 / 7 test cases passed)**

---

## G.2 Non-Functional Testing Evidence

### G.2.1 Performance Testing (NFR02, DG02)

System performance was evaluated for computational efficiency and inference latency on the local development environment (NVIDIA RTX 3050, 6 GB VRAM). Processing standard 10-second, 12-lead WFDB recordings through the complete pipeline — from zero-phase Butterworth filtering and spatial tensor construction to ensemble probability generation — was consistently completed within **2.8 to 4.2 seconds**, comfortably within the 10-second NFR02 threshold.

Frontend responsiveness was profiled using browser developer tools, confirming acceptable JavaScript execution times and DOM rendering behaviour with no blocking of main interaction threads.

![Performance testing — browser developer tools profiling](thesis_figures/performance_browser_results.png)
*Figure G.1: Performance testing — browser developer tools network profiling confirming inference round-trip within the NFR02 threshold.*

### G.2.2 GUI Testing (NFR05, DG06)

Usability and interface stability were assessed to confirm that non-technical operators can navigate the system without requiring knowledge of the underlying ML architecture. Google Lighthouse profiling was applied to formalise accessibility, best practice, and structural performance scores. The interface effectively isolates user-facing complexity, presenting only the file upload boundary, demographic inputs, model selector, and result panel.

![GUI testing — Google Lighthouse profiling](thesis_figures/gui_lighthouse_testing.png)
*Figure G.2: GUI testing — Google Lighthouse audit results confirming accessibility and best-practice compliance.*

### G.2.3 Maintainability Testing (NFR04, DG07)

Code maintainability is enforced through strict separation of concerns across the project structure: frontend handlers, Flask API endpoints, preprocessing pipeline modules, and model inference components are maintained in independent modules with clearly defined interfaces. This structure enables isolated modification of any individual component without propagating changes across the system. Static code quality analysis via CodeFactor was used to assess adherence to Python PEP 8 and JavaScript style standards.

![Maintainability testing — CodeFactor analysis results](thesis_figures/maintainability_codefactor.png)
*Figure G.3: Maintainability testing — CodeFactor static code analysis results.*

### G.2.4 Compatibility Testing (NFR05, DG06)

Cross-browser and responsive layout testing validated that the ChagaSight interface maintains structural integrity across primary rendering environments. Browser developer tools were used to simulate a range of viewport widths. Testing confirmed correct adaptive layout behaviour at standard desktop resolutions (1920×1080, 1366×768) across Google Chrome, Mozilla Firefox, and Microsoft Edge.

![Compatibility testing — Microsoft Edge desktop rendering](thesis_figures/compatibility_desktop_edge.png)
*Figure G.4: Compatibility testing — Microsoft Edge at standard desktop resolution.*

![Compatibility testing — Google Chrome desktop rendering](thesis_figures/compatibility_desktop_chrome.png)
*Figure G.5: Compatibility testing — Google Chrome at standard desktop resolution.*

![Compatibility testing — responsive mobile scaling](thesis_figures/compatibility_mobile.png)
*Figure G.6: Compatibility testing — responsive layout scaling across reduced viewport widths.*

### G.2.5 Security and Data Protection Testing (NFR03, DG08)

Data minimisation was verified by confirming that uploaded `.hea` and `.dat` files do not persist in backend storage beyond the inference cycle. Post-inference inspection of the `/uploads` directory consistently returned an empty state, confirming that the `_cleanup()` subroutine executes reliably. CodeQL static analysis was applied to the project repository to identify potential injection vulnerabilities or unsafe dependency usage at the source code level.

![Security testing — CodeQL analysis results](thesis_figures/security_codeql_results.png)
*Figure G.7: Security testing — CodeQL static analysis results confirming absence of injection vulnerabilities.*

### G.2.6 Repository Status

The project repository was maintained in a stable state throughout development, with all training runs, preprocessing changes, and deployment configurations tracked under version control.

![GitHub repository status](thesis_figures/repo_status.png)
*Figure G.8: ChagaSight GitHub repository status confirming active version control and commit history.*

### G.2.7 Non-Functional and Design Goal Test Cases

**Table G.2: Non-functional and design goal test case results.**

| Test Case | Requirement | Result Description | Status |
|:---|:---|:---|:---|
| TC-NF01 | NFR01 / DG01 | Ensemble AUROC = 0.8707 [95% CI: 0.8665–0.8746]; AUPRC = 0.2589; both metrics exceed the NFR01 minimum (AUROC ≥ 0.85) across all five cross-validation folds. | **Pass** |
| TC-NF02 | NFR02 / DG02 | Complete inference pipeline (WFDB ingestion, preprocessing, 5-fold ensemble, result delivery) completed within 2.8–4.2 seconds on the local environment; within the 10-second requirement. | **Pass** |
| TC-NF03 | NFR03 / DG08 | Post-inference `/uploads` directory confirmed empty; `_cleanup()` subroutine fires correctly after every inference cycle irrespective of success or error state. | **Pass** |
| TC-NF04 | NFR04 / DG07 | Project structure enforces module boundaries across preprocessing, model, frontend, and API components; Git version history maintained throughout development lifecycle. | **Pass** |
| TC-NF05 | NFR05 / DG06 | Responsive layout confirmed across Chrome, Firefox, and Edge at 1920×1080 and 1366×768 resolutions; no element overlap or CSS distortion observed. | **Pass** |
| TC-NF06 | NFR06 | "Research Prototype" disclaimer displayed on all prediction result views; system operates exclusively on de-identified, publicly available datasets (SaMi-Trop, PTB-XL, CODE-15%). | **Pass** |
| TC-NF07 | NFR07 | All twelve ECG leads rendered as individual waveforms in the results panel; 2D spatial ECG image displayed alongside the prediction output. Clinicians can visually inspect the input signal processed during inference without knowledge of the underlying architecture. | **Pass** |
| TC-NF08 | DG03 | Zero-phase Butterworth filtering and per-lead Z-score normalisation execute consistently across WFDB recordings from all three source datasets without signal-specific configuration. | **Pass** |
| TC-NF09 | DG04 | Three-tier separation of presentation, application, and inference layers enables independent horizontal scaling; validated architecturally; live multi-user testing deferred. | **Architecturally Met** |
| TC-NF10 | DG05 | Pipeline modules (upload, preprocessing, embedding, ViT inference, result routing) operate without destructive inter-module dependencies; confirmed via modular integration testing. | **Pass** |

---

## G.3 Training and Results Evidence

### G.3.1 Self-Supervised Pretraining (Phase 1)

Phase 1 pretraining captures the isolated convergence of the Masked Autoencoder (MAE) objective applied to the 2D spatial pathway and the Spatio-Temporal Masked ECG Modelling (ST-MEM) objective governing the 1D temporal pathway. These figures record the execution logs confirming successful pretraining completion prior to supervised fine-tuning.

![MAE pretraining execution log](thesis_figures/training_mae_pretraining.png)
*Figure G.9: Training log — Masked Autoencoder (MAE, 2D pathway) pretraining execution. 30 epochs completed on the full preprocessing cohort.*

![ST-MEM pretraining execution log](thesis_figures/training_stmem_pretraining.png)
*Figure G.10: Training log — Spatio-Temporal Masked ECG Modelling (ST-MEM, 1D pathway) pretraining execution. 30 epochs completed.*

### G.3.2 Supervised Fine-Tuning (Folds 0–4)

Cross-fold training progression logs record per-epoch AUROC, AUPRC, and loss convergence for all five fold models over the full dataset.

![Training progression — Fold 0](thesis_figures/training_fold_0.png)
*Figure G.11: Supervised fine-tuning progression — Hybrid Ensemble Fold 0 (n ≈ 292,944 training, n = 94,037 validation).*

![Training progression — Fold 1](thesis_figures/training_fold_1.png)
*Figure G.12: Supervised fine-tuning progression — Hybrid Ensemble Fold 1 (n ≈ 313,745 training, n = 73,236 validation).*

The combined training curve overview across all five folds is provided in **Figure G.13**.

![Cross-validation training curves — all folds](../thesis_figures/fig_appendix_training_curves_all_folds.png)
*Figure G.13: Training progression across all five cross-validation folds — AUROC and AUPRC convergence during supervised fine-tuning on the full 366,181-sample dataset.*

### G.3.3 Intermediate Training Run (83k Dataset)

Prior to scaling to the full 366,181-sample cohort, an intermediate training run was conducted on a subset of 83,130 samples to validate the training pipeline and model architecture. The corresponding ROC and Precision-Recall curves for this intermediate checkpoint are presented in **Figure G.14**.

![83k dataset ROC and PR curves](../thesis_figures/fig_appendix_83k_roc_pr.png)
*Figure G.14: ROC curve (AUROC = 0.9275) and Precision-Recall curve (AUPRC = 0.4973) for the intermediate 83,130-sample checkpoint. The higher metrics relative to the full dataset reflect the elevated positive prevalence (3.42%) in this subset and the reduced heterogeneity of the test set at this training scale.*

### G.3.4 Ablation Studies (Individual Pathways)

Individual pathway fine-tuning logs record the independent training progression of the 1D-only and 2D-only models used in the ablation study reported in Section 8.5.1.

![1D-only ablation training log](thesis_figures/training_1d_ablation.png)
*Figure G.15: Training log — 1D-only model (ST-MEM backbone) ablation fine-tuning run.*

![2D-only ablation training log](thesis_figures/training_2d_ablation.png)
*Figure G.16: Training log — 2D-only model (MAE backbone) ablation fine-tuning run.*
