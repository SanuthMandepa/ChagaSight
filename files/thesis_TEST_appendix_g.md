# APPENDIX G

## G.1 Non-functional testing

### G.1.1 Performance testing
The performance of the system was assessed based on computational efficiency, inference latency, and system responsiveness, tracing directly to NFR02 and DG02. The implementation was evaluated on a local computational environment (NVIDIA RTX 3050, 6 GB VRAM). Due to the architectural demands of the dual-pathway Vision Transformer ensemble, GPU memory constraints played a significant factor in batch sizing during the training phases, requiring memory offloading and rigorous checkpointing strategies to maintain stable training spans without data loss.

During clinical deployment simulation, the system demonstrated stable inference times. Processing standard 10-second 12-lead WFDB recordings completed the entire pipeline—from zero-phase Butterworth filtering and spatial normalisation to final ensemble probability generation—within 2.8 to 4.2 seconds on average. This consistently satisfies the strict threshold limit of returning an evaluation within 10 seconds. The frontend user experience was profiled using browser developer tools to verify acceptable JavaScript execution times, payload delivery, and DOM rendering without blocking the main interaction threads.

![Performance testing - browser results](thesis_figures/performance_browser_results.png)
*Figure G.1: Performance testing browser profiling results.*

### G.1.2 GUI testing
System usability and interface stability (NFR05, DG06) were validated to ensure non-technical clinical operators could navigate the uploading and prediction modalities seamlessly. Google Lighthouse profiling was conducted to formalise the accessibility, best practices, and frontend structural performance. The GUI effectively isolates complexity, providing only the required upload boundaries, demographic drop-downs, and model-switching modules.

![GUI testing - Google Lighthouse](thesis_figures/gui_lighthouse_testing.png)
*Figure G.2: GUI testing – Google Lighthouse testing.*

### G.1.3 Maintainability testing
Codebase maintainability (NFR04, DG07) was strictly enforced ensuring modular integration of the inference pathways, preprocessing, and frontend logic. This separation of concerns guarantees further improvements, such as the deferred explainability visualisations (NFR07), can be integrated seamlessly. Standard Git version control logic tracked historical progress. Static code analysis via tools such as CodeFactor was utilised to determine the underlying script quality, highlighting adherence to accepted Python and JavaScript styling parameters. 

![Maintainability testing - CodeFactor test results](thesis_figures/maintainability_codefactor.png)
*Figure G.3: Maintainability testing - CodeFactor test results.*

### G.1.4 Compatibility testing
To assure accessibility and responsive structuring of the ChagaSight web interface (NFR05), compatibility tests simulated various resolution densities and screen boundaries using native browser developer tools. The frontend was engineered utilizing flexible layout implementations, enabling the dashboard to shift element scales dynamically based on shifting aspect ratios across different device categories.

Visual element bounds, typographic clarity, and component scaling were checked meticulously. Testing confirmed that the layout preserves structural integrity across primary rendering environments including Google Chrome, Mozilla Firefox, and Microsoft Edge.

![Compatibility testing - Desktop Edge rendering](thesis_figures/compatibility_desktop_edge.png)
*Figure G.4: Microsoft Edge - standard desktop rendering.*

![Compatibility testing - Desktop Chrome rendering](thesis_figures/compatibility_desktop_chrome.png)
*Figure G.5: Google Chrome - standard desktop rendering.*

![Compatibility testing - Tablet/Mobile rendering](thesis_figures/compatibility_mobile.png)
*Figure G.6: Responsive scaling across mobile aspect ratios.*

### G.1.5 Security and Data Protection testing
Data protection remains a fundamental objective, formalised within NFR03 and DG08. The system strictly adheres to data minimisation constraints; uploaded `.hea` and `.dat` artifacts exist within standard storage bounds exclusively during active inference. Validations confirm background routines consistently erase patient metrics from memory buffers following HTTP context resolution. CodeQL vulnerability analysis routines were concurrently executed against the project repository to identify script-level injection flaws or dependency decay. 

![Security testing - CodeQL results](thesis_figures/security_codeql_results.png)
*Figure G.7: Security testing - CodeQL results.*

### G.1.6 Repository status
The system maintains a stable project repository aligning with baseline software reliability standards necessary for reproducible dataset training and application enhancement.

![GitHub repository status](thesis_figures/repo_status.png)
*Figure G.8: ChagaSight GitHub repository status.*

### G.1.7 Non-Functional and Design Goal Test Cases

**Table G.1: Non-Functional and Design Goal test cases.**
| Test Case | ID | Result Description | Status |
|---|---|---|---|
| TC-NF01 | NFR01 / DG01 | The system achieved an AUROC of 0.8707 and sensitivity of 0.4958, effectively exceeding the required baseline AUROC of 0.85 and 0.40 TPR thresholds, demonstrating excellent diagnostic accuracy under class-imbalanced constraints. | Pass |
| TC-NF02 | NFR02 / DG02 | Computations over uploaded 12-lead ECG pairs complete inference entirely within 2.8 to 4.2 seconds under local simulation payloads, satisfying the required 10-second processing thresholds. | Pass |
| TC-NF03 | NFR03 / DG08 | Data minimisation directives consistently fire subroutines ensuring `.dat` and `.hea` WFDB inputs are purged securely from backend storage locations explicitly after inference ends. | Pass |
| TC-NF04 | NFR04 / DG07 | Structural integrity inspections confirm application modularity logic separating frontend handlers, ST-MEM components, and MAE modules via version-tracked Git revisions maintaining clear separation of concerns. | Pass |
| TC-NF05 | NFR05 / DG06 | UI inspections simulated across Chrome, Firefox, and Edge under standard dual-monitor bounds consistently validated dynamic layout scaling without overlay overlap. | Pass |
| TC-NF06 | NFR06 | Predictive outcomes interface consistently displayed the mandatory 'Research Disclaimer' parameters and isolated execution strictly utilizing the de-identified SaMi-Trop, PTB-XL, and CODE-15% components. | Pass |
| TC-NF07 | NFR07 | Implementation of lead-wise attention generation routines remains designated as a deferred addition out of scope for the immediate version phase. | Partially Met |
| TC-NF08 | DG03 | The standardisation processing components strictly execute parallel Z-score standardisation and zero-phase Butterworth implementations reliably across widely dissimilar WFDB baseline source types. | Pass |
| TC-NF09 | DG04 | 3-Tier mapping architectures allow inference modules to decouple strictly from presentation constraints securely isolating variables allowing valid concurrent cloud scaling in eventual network additions. | Pass |
| TC-NF10 | DG05 | The pipeline modules encompassing upload, data embeddings, ViT integration, and result routing operate cohesively without destructive overlaps. | Pass |

---

## G.2 Training and results

### G.2.1 Self-Supervised Pretraining (Phase 1)
Pretraining phase validation capturing the isolated progression dynamics of the Masked Autoencoder (MAE) applied to the 2D path and the ST-MEM objective governing the 1D ViT paths. Plottings monitor internal pattern recognition generation without assigned datasets.

![Training timestamp - MAE Pretraining](thesis_figures/training_mae_pretraining.png)
*Figure G.9: Training timestamp - MAE (2D) pretraining execution script.*

![Training timestamp - ST-MEM Pretraining](thesis_figures/training_stmem_pretraining.png)
*Figure G.10: Training timestamp - ST-MEM (1D) pretraining execution script.*

### G.2.2 Full Supervised Fine-Tuning Ensemble (Folds 0-4)
Cross-component training runs assessing the merged configurations over identical stratifications representing progression up to comprehensive dataset completions. 

![Training timestamp - Full Dataset Fold 0](thesis_figures/training_fold_0.png)
*Figure G.11: Training progression - Hybrid Ensemble Fold 0 supervised fine-tuning.*

![Training timestamp - Full Dataset Fold 1](thesis_figures/training_fold_1.png)
*Figure G.12: Training progression - Hybrid Ensemble Fold 1 supervised fine-tuning.*

![Plotted Results - Cross-validation Evaluation](thesis_figures/plotted_cv_results.png)
*Figure G.13: Plotted loss curves and metric convergence (AUROC, AUPRC) across cross-validation validation sets.*

### G.2.3 Ablation Studies (Independent Pathways)
Validation logs explicitly monitoring architecture independence behaviour outlining specific metric responses to purely 1D or 2D restrictions over similar isolated epochs. 

![Training timestamp - 1D Ablation Run](thesis_figures/training_1d_ablation.png)
*Figure G.14: Training timestamp - 1D-only model fine-tuning progression.*

![Training timestamp - 2D Ablation Run](thesis_figures/training_2d_ablation.png)
*Figure G.15: Training timestamp - 2D-only model fine-tuning progression.*
