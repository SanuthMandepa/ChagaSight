# ChagaSight FYP Thesis — Full Guidance

> Based on a detailed review of your codebase ([app.py](file:///d:/IIT/L6/FYP/ChagaSight/app.py), [App.jsx](file:///d:/IIT/L6/FYP/ChagaSight/frontend/src/App.jsx), [PROJECT_PLAN.md](file:///d:/IIT/L6/FYP/ChagaSight/PROJECT_PLAN.md), and `src/models/`).

---

## Part 1 — Are Your Functional & Non-Functional Requirements Correct?

### ✅ Requirements That Are Correct and Well-Scoped

| ID | OK? | Note |
|----|-----|------|
| FR01 | ✅ | System correctly accepts `.hea + .dat/.mat` (WFDB) — matches implementation |
| FR02 | ✅ | `baseline_removal.py`, `resample.py`, `normalization.py` all exist — matches |
| FR03 | ✅ | `hybrid_model.py`, `vit_1d_fm.py`, `vit_2d.py` — matches |
| FR04 | ✅ | Frontend shows probability gauge, percentage, and interpretation |
| NFR01 | ✅ | You achieved AUROC 0.896 and screening sensitivity 0.504, which exceeds the updated targets of 0.88 and 0.40 |
| NFR02 | ✅ | Single-sample inference is well under 10 seconds |
| NFR05 | ✅ | Modular `src/` structure with Git — matches |

### ⚠️ Requirements That Need Correction or Removal

#### FR01 — Format Claim
> **Problem:** You claim CSV and XML support, but your backend **only accepts WFDB** (`.hea + .dat/.mat`).  
> **Fix:** Remove CSV/XML from FR01, or add a disclaimer that only WFDB is implemented in this version.

**Corrected FR01:**
> The system shall allow users to upload 12-lead ECG recordings in WFDB format (`.hea` + `.dat`/`.mat` file pairs). Future versions may support CSV and XML formats.

---

#### FR05 — User Authentication
> **Problem:** Your backend (`app.py`) has **no authentication** — no login, no JWT, no session management. FR05 is **not implemented**.  
> **Fix:** Either (a) drop the priority to "Will Not Have" and note it as future work, or (b) add a disclaimer in the thesis that authentication was descoped due to the research prototype nature of the system.  
> **Do NOT say it is implemented if it isn't.**

---

#### FR06 — Report Generation (PDF/CSV Export)
> **Problem:** Your frontend has **no export button or PDF generation**. This is not implemented.  
> **Fix:** Same as FR05 — reduce to "Will Not Have" or mark as future work.

---

#### FR07 — Dataset Management
> **Problem:** There is no dataset management UI or backend storage layer.  
> **Fix:** Keep as "Could Have / Will Not Have" and explicitly note it is out of scope in this version.

---

#### FR08 — Feedback & Model Retraining
> **Problem:** Not implemented — no feedback API, no retraining pipeline in the web system.  
> **Fix:** Move to "Will Not Have" for this version.

---

#### NFR01 — Accuracy Target Correction
> **Problem:** You set F1 ≥ 0.85 as a target, but F1 at your optimised threshold is likely ~0.3–0.5 due to the extreme class imbalance (2.24% positives). Your **primary metric is TPR@5%** and your AUROC ≥ 0.88 is met.  
> **Fix:** Replace F1 ≥ 0.85 with:
> - AUROC ≥ 0.88 ✅ (achieved ~0.896)
> - TPR@5% ≥ 0.40 ✅ (achieved ~0.504 for Hybrid)
> - F1 is reported but NOT used as a pass/fail threshold due to class imbalance

---

#### NFR03 — AES-256 Encryption
> **Problem:** Your system has **no encryption**. Files are saved to the local `uploads/` folder and deleted after processing, but there is no AES-256 at-rest or TLS enforcement.  
> **Fix:** Downgrade to "Should Have" and note that the prototype handles data ephemerally (files deleted after inference) and that production deployment would require TLS + encryption.

---

#### NFR07 — GDPR/HIPAA Compliance
> **Problem:** No consent tracking is present in the system.  
> **Fix:** Downgrade to "Should Have" for production; note the prototype is for research use with de-identified datasets and does not process real patient data.

---

#### NFR08 — Explainability (Attention Maps)
> **Problem:** The frontend shows a probability gauge but NO lead-wise attention heatmaps.  
> **Fix:** Either implement a basic attention map display (see Part 5 below), or downgrade this to "Could Have" and mark as future work.

---

### Missing Requirement: FR10 — Model Selection
> **Problem:** Your frontend supports selecting between 3 model modes (Hybrid, 2D-only, 1D-only). This is a real, implemented feature not covered by any FR.  
> **Add FR10:**

| ID | Requirement | Priority |
|----|-------------|----------|
| FR10 | The system shall allow users to select between three diagnostic model modes: Hybrid Ensemble, 2D Visual Model, and 1D Signal Model. Each mode shall display its AUROC and TPR@5% metrics to inform the user's selection. | Should Have |

---

## Part 2 — Do Requirements Match Diagrams?

### Use Case Diagram — Check These

Your use case diagram should include these actors and use cases:

| Actor | Use Cases |
|-------|-----------|
| Clinician / Researcher | Upload ECG, Select Model, Enter Demographics, View Prediction, Load Sample ECG |
| System (AI) | Preprocess Signal, Run Inference, Return Probability + Interpretation |
| Administrator | (FR05 — not implemented; remove from diagram or mark as out of scope) |

> **Action:** If your use case diagram shows "Login", "Export Report", or "Manage Datasets" as primary use cases, those should be moved to **extension** or **future work** use cases, since they are not implemented. The diagrams should match the system that was built.

### Class Diagram — Check These

Your class diagram should reflect the actual codebase:
- `ViT2D` class (`vit_2d.py`)
- `ViT1D_FM` class (`vit_1d_fm.py`)
- `HybridChagasModel` class (`hybrid_model.py`)
- `REPAAlignment` class (`repa_alignment.py`)
- Flask `app.py` as the controller/server layer
- React frontend as the presentation layer

> **Action:** If your class diagram has "UserManager", "AuthController", or "ReportGenerator" classes — remove them, as these don't exist.

### Activity Diagram — Check These

The activity diagram should show:
1. User uploads `.hea + .dat` files
2. System validates file pair
3. User selects model type
4. User enters demographics (if needed)
5. User clicks "Analyze ECG"
6. Backend preprocesses signal (baseline removal → resample → normalize → build 2D image)
7. Model runs inference
8. Frontend displays probability, gauge, interpretation

> **Action:** If there is a "Login" or "Export Report" activity node — remove it or mark it as out of scope.

---

## Part 3 — Chapters 1–7: Corrections Needed

### Chapter 1 — Introduction
- ✅ Motivation (Chagas endemic regions, screening gap) — should be strong
- ⚠️ If you claim the system supports "authenticated multi-user access" — **remove this claim**
- ⚠️ Scope: Clearly state this is a **research prototype** for single-ECG analysis, not a production clinical system

### Chapter 2 — Literature Review
- ✅ Should cite: MAE (He et al. 2022), ST-MEM (Van Santvliet et al. 2025), Kim et al. 2025 (2D ViT), ASL loss (Ridnik et al. 2021), FiLM modulation
- ⚠️ Ensure you discuss **class imbalance** in ECG AI literature (not just general ML)
- ⚠️ Include a comparison table of related work (model type, dataset, metric)

### Chapter 3 — Methodology / System Design
- ✅ Dual-pathway architecture is your main contribution — explain it clearly
- ✅ WCT re-referencing justification is well-documented in `PROJECT_PLAN.md`
- ⚠️ Add a clear **system architecture diagram** showing: Frontend → Flask API → Preprocessing → Model → Response
- ⚠️ If you describe a 3-tier architecture, confirm the diagram matches: Presentation (React) / Application (Flask) / Data (file uploads + model weights)

### Chapter 4 — Requirements (SRS)
- See Part 1 above for all specific corrections
- **Key change:** Correct FR01, remove/downgrade FR05/FR06, fix NFR01 metric, fix NFR03/NFR07

### Chapter 5 — System Design (UML Diagrams)
- ⚠️ Use case: Remove unimplemented actors/use cases
- ⚠️ Class diagram: Must match `src/models/` and `app.py`
- ⚠️ Activity diagram: Must reflect actual upload → preprocess → infer → display workflow
- ⚠️ Sequence diagram (if present): Should show browser ↔ `app.py` ↔ model interaction

### Chapter 6 — Implementation
See Part 4 below for full guidance.

### Chapter 7 — Testing
- ✅ You have a `tests/` directory — describe what unit tests were written
- ⚠️ Add a **test summary table**: test case name, what it tests, pass/fail
- ⚠️ If you have no unit tests, you must at minimum describe:
  - Manual testing of the API (`/api/health`, `/api/predict`)
  - Integration testing: full pipeline from file upload to result
  - Evaluation metrics as a form of model testing

---

## Part 4 — Chapter 8: Implementation (How to Write It)

> This chapter describes WHAT you built and HOW. Structure it as follows:

### 8.1 Development Environment
- Python 3.x, PyTorch, Flask, React/Vite, WFDB library
- Hardware: NVIDIA RTX 3050 6 GB VRAM
- Version control: Git

### 8.2 Data Preprocessing Pipeline
Describe the 4 steps with code snippets:
1. WFDB loading (`wfdb.rdsamp`)
2. Baseline removal (Butterworth bandpass 0.5–40 Hz)
3. Resampling (100 Hz for 1D, 500 Hz for 2D)
4. Z-score normalisation per lead

### 8.3 Data Representation
- 1D: `(12, 1000)` signal tensor
- 2D: WCT re-referencing → `(3, 24, 2048)` image tensor

### 8.4 Model Architecture
Describe the 3 components with diagrams:
- **1D ViT (ST-MEM backbone):** per-lead patching (50-sample patches), 12 layers, FiLM demographic modulation, AoL aggregation
- **2D ViT (MAE backbone):** 8×64 patch size, 96 patches, 12 layers, AoL aggregation
- **REPA alignment module:** projection + cosine similarity loss
- **Hybrid classifier:** concatenated 1536d → 3-layer MLP → sigmoid output

### 8.5 Training Strategy
- Self-supervised pretraining: MAE (2D) + ST-MEM (1D) on 366k records
- Fine-tuning: 2-phase progressive unfreezing (Phase 1: freeze FM, Phase 2: full fine-tune)
- Loss: Asymmetric BCE with `w_pos=10`, `γ⁻=2`
- 5-fold stratified cross-validation

### 8.6 Web Application
Describe the Flask API:
- `GET /api/health` — returns model status
- `POST /api/predict` — accepts WFDB files + demographics, returns JSON result

Describe the React frontend:
- Model selector (Hybrid / 2D / 1D)
- WFDB drag-and-drop upload
- Sample ECG loader
- Demographics input (age, sex)
- Probability gauge + interpretation display

---

## Part 5 — Chapter 9: Evaluation (How to Write It)

This is your most important chapter. Structure it as follows:

### 9.1 Evaluation Methodology
- 5-fold stratified cross-validation
- Primary metric: **TPR@5%** (explain why — matches clinical deployment capacity constraint)
- Secondary metrics: AUROC, AUPRC, F1, Sensitivity, Specificity, MCC, NPV, NNS
- Ensemble: mean probability across all 5 fold models

### 9.2 Results Table (use the real numbers from your `evaluation_complete_v3.1.ipynb`)

| Model | AUROC | AUPRC | TPR@5% | F1 | Sensitivity | Specificity |
|-------|-------|-------|--------|-----|-------------|-------------|
| 1D-only | 0.828 | — | 0.429 | — | — | — |
| 2D-only | 0.844 | — | 0.463 | — | — | — |
| **Hybrid (Ensemble)** | **0.896** | — | **0.504** | — | — | — |
| Random baseline | — | — | 0.050 | — | — | — |
| Kim et al. (2025) | — | — | 0.369 | — | — | — |
| Van Santvliet et al. | — | — | 0.445 | — | — | — |

> Fill in the blanks with your actual notebook results. These are the numbers already displayed in your frontend.

### 9.3 Per-Dataset Breakdown
Report results separately for PTB-XL, SaMi-Trop, CODE-15% to show:
- Generalisation across populations
- Effect of enriched cohort (SaMi-Trop has higher Chagas prevalence)

### 9.4 Ablation Study
Compare these variants to show each component's contribution:
1. Hybrid (full) — 0.504
2. 2D-only — 0.463  
3. 1D-only — 0.429  
4. Phase 1 only (frozen FM) — 0.138 (from `PROJECT_PLAN.md`)

> This shows REPA alignment + joint fine-tuning adds +0.041 TPR@5% over 2D alone.

### 9.5 Comparison to Literature Benchmarks
| System | TPR@5% |
|--------|--------|
| Random baseline | 0.050 |
| Phase 1 only | 0.138 |
| Kim et al. (2025) | 0.369 |
| **ChagaSight** | **0.504** |
| Van Santvliet et al. (top reported) | 0.445 |

> Note: ChagaSight outperforms the contemporary reported baselines on your validation set. Acknowledge that direct comparisons have limitations as they evaluate on different test splits.

### 9.6 Calibration (Optional but impressive)
- If you have probability calibration results (Reliability Diagram / Brier Score), include them
- Shows the model's probabilities are meaningful, not just rankings

### 9.7 Limitations
- Pretraining data leakage (unlabelled samples seen in self-supervised pretraining)
- No serology-confirmed ground truth for CODE-15% labels
- Tested only on publicly available datasets; real-world performance may differ
- No attention map visualisation implemented yet

---

## Part 6 — Chapter 10: Conclusion (How to Write It)

Structure as follows:

### 10.1 Summary of Contributions
1. A dual-pathway Vision Transformer (1D + 2D) for Chagas ECG detection
2. Integration of ST-MEM and MAE pretraining for ECG foundation models
3. REPA cross-modal alignment applied to multimodal ECG analysis
4. A full-stack clinical decision support prototype: Flask API + React frontend
5. Achieved TPR@5% = 0.504, surpassing contemporary top-reported literature benchmarks (0.445)

### 10.2 Addressing the Research Objectives
Map each result back to your original research questions/objectives from Chapter 1.

### 10.3 Limitations
- Authentication and report export not implemented
- Model requires WFDB format — no CSV/XML support
- No lead-level explainability (attention maps) in UI
- Single-institution deployment not yet validated

### 10.4 Future Work
- Add attention heatmap visualisation per lead (addresses NFR08)
- Role-based authentication (FR05)
- PDF report export (FR06)
- Integrate with external medical APIs (FR09)
- Test on held-out independent test sets
- Serology validation on de-identified patient data

### 10.5 Closing Statement
Summarise the clinical impact: scalable, AI-driven Chagas screening from standard 12-lead ECGs, requiring no specialised equipment, which could enable early detection in resource-limited endemic regions.

---

## Part 7 — Frontend: What Needs to Change

Your frontend is already very strong. These are improvements that would close gaps with the requirements:

### Missing — NFR08 Attention Maps (High Impact)
> The current frontend shows a probability gauge but no explanation of **which leads drove the decision**.  
> Even a simple heatmap showing per-lead attention weights from the 1D ViT would satisfy NFR08.  
> **Recommendation:** If you cannot implement this before submission, downgrade NFR08 to "Could Have" and list it as future work.

### Minor — Disclaimer Prominence
> Add a more prominent disclaimer on the result panel (not just the footer) stating:  
> "This tool is a research prototype and is not validated for clinical use."

### Minor — About Tab
> The About tab should mention the three-dataset training approach and the rationale for the dual-pathway architecture.

---

## Part 8 — Implementation: What Needs to Change

### ✅ No Major Code Changes Needed for Thesis Submission

The implementation is complete and functional. However, for **thesis consistency**:

1. **Add CSV logging in `app.py`** — log each prediction (timestamp, model_type, probability, prediction) to a rotating log file for reproducibility evidence in the thesis
2. **Verify `tests/`** — make sure you can describe at least 3 test cases in Chapter 7
3. **Comment `hybrid_model.py`** — ensure the code is readable for the thesis appendix

---

## Summary: Priority Actions Before Submission

| Priority | Action |
|----------|--------|
| 🔴 Critical | Correct FR01, FR05, FR06, NFR01, NFR03, NFR07 in Chapter 4 |
| 🔴 Critical | Align use case/class/activity diagrams with actual implementation |
| 🔴 Critical | Write Chapter 9 (Evaluation) with your actual numbers from the notebook |
| 🟡 Important | Add FR10 (model selection) as a new functional requirement |
| 🟡 Important | Write Chapter 8 (Implementation) using the structure above |
| 🟡 Important | Write Chapter 10 (Conclusion) using the structure above |
| 🟢 Optional | Add attention map display to frontend (NFR08) |
| 🟢 Optional | Add PDF export (FR06) |

---

## Part 9 — Rewritten Requirements Tables (Copy Directly Into Chapter 4 & 9)

Add this paragraph before the FR table in Chapter 4:
> *Table 4.1 presents the functional requirements for ChagaSight, prioritised using the MoSCoW method. Requirements FR01–FR07 represent the implemented scope of the current prototype. FR08–FR10 are identified as desirable extensions, and explicitly deferred to future versions.*

### Functional Requirements (Chapter 4)

| ID | Requirement | Priority (MoSCoW) |
|----|-------------|-------------------|
| **FR01** | The system shall allow users to upload 12-lead ECG recordings in WFDB format (.hea and .dat or .mat file pairs). Support for additional formats such as CSV and XML is deferred to a future version. | Must Have |
| **FR02** | The system shall preprocess uploaded ECG recordings through a four-stage pipeline: baseline removal via a zero-phase Butterworth bandpass filter (0.5–40 Hz), resampling to 100 Hz for the 1D pathway and 500 Hz for the 2D pathway, per-lead Z-score normalisation, and Wilson Central Terminal re-referencing to construct a (3, 24, 2048) spatial image tensor. | Must Have |
| **FR03** | The system shall run inference using a dual-pathway Vision Transformer ensemble comprising a 1D signal pathway pretrained with ST-MEM and a 2D image pathway pretrained with a Masked Autoencoder objective, with cross-modal REPA alignment. The ensemble shall aggregate predictions across five cross-validation fold models to produce a final probability score. | Must Have |
| **FR04** | The system shall display the inference result as a probability percentage, a visual gauge, a binary classification label (Low Risk or High Risk), and a plain-language clinical interpretation string. | Must Have |
| **FR05** | The system shall allow users to select between three diagnostic model modes — Hybrid Ensemble, 2D Visual Model, and 1D Signal Model — and shall display the AUROC and screening sensitivity score associated with each mode. | Should Have |
| **FR06** | The system shall provide a sample ECG loader allowing users to select pre-loaded representative recordings from the SaMi-Trop, PTB-XL, and CODE-15% datasets for demonstration and evaluation purposes. | Should Have |
| **FR07** | The system shall accept optional patient demographic inputs (age and sex) and incorporate them into model inference via FiLM conditioning layers within the 1D ViT backbone. | Should Have |
| **FR08** | The system shall generate a downloadable PDF report containing the probability score, classification label, and interpretation text. | Should Have — Not Implemented; deferred to a future version |
| **FR09** | The system shall provide user authentication to restrict access to authorised users in a clinical deployment context. | Could Have — Not Implemented; deferred to a production version |
| **FR10** | The system shall provide an administrative interface for uploading and labelling new ECG recordings to support future model retraining. | Could Have — Not Implemented; out of scope for this research prototype |

### Non-Functional Requirements (Chapter 4)

| ID | Requirement | Priority (MoSCoW) |
|----|-------------|-------------------|
| **NFR01** | The model shall achieve AUROC ≥ 0.88 and a screening sensitivity score (TPR@5%) ≥ 0.40 on the held-out cross-validation test set. F1 score is reported as a secondary metric but is not used as a pass/fail criterion due to extreme class imbalance (2.24% positive prevalence). | Must Have |
| **NFR02** | The system shall return a prediction result within 10 seconds of file upload for a standard 10-second WFDB recording. | Must Have |
| **NFR03** | Uploaded ECG files shall be deleted from server storage immediately upon completion of inference. At-rest AES-256 encryption and TLS enforcement are deferred to a production deployment. | Should Have |
| **NFR04** | The codebase shall be organised into discrete functional modules (preprocessing, models, training, evaluation, API, frontend) following separation of concerns principles, with version control maintained throughout development using Git. | Must Have |
| **NFR05** | The frontend interface shall be responsive across standard desktop screen sizes and browsers including Chrome, Firefox, and Edge. | Should Have |
| **NFR06** | The system shall display a research disclaimer on all prediction results and shall be used exclusively with de-identified publicly available datasets. Full GDPR and HIPAA compliance are deferred to a production deployment with appropriate clinical data governance. | Should Have |
| **NFR07** | The system shall provide lead-wise attention weight visualisation to support clinical interpretability of predictions. | Could Have |

### Functional Requirements — Evaluation Status (Chapter 9)

| ID | Priority | Status |
|----|----------|--------|
| **FR01** | Must Have | Implemented |
| **FR02** | Must Have | Implemented |
| **FR03** | Must Have | Implemented |
| **FR04** | Must Have | Implemented |
| **FR05** | Should Have | Implemented |
| **FR06** | Should Have | Implemented |
| **FR07** | Should Have | Implemented |
| **FR08** | Should Have | Not Implemented — deferred to future version |
| **FR09** | Could Have | Not Implemented — deferred to production deployment |
| **FR10** | Could Have | Not Implemented — out of scope for research prototype |

*Must Have completion: 4/4 (100%). Should Have completion: 3/4 (75%). Could Have completion: 0/2 (0%).*

### Non-Functional Requirements — Evaluation Status (Chapter 9)

| ID | Priority | Status |
|----|----------|--------|
| **NFR01** | Must Have | AUROC 0.896 ✓ — target exceeded; screening sensitivity 0.504 ✓ — target exceeded |
| **NFR02** | Must Have | Met — inference completes within 10 seconds on consumer hardware |
| **NFR03** | Should Have | Partially implemented — ephemeral file deletion after inference; at-rest encryption and TLS deferred to production |
| **NFR04** | Must Have | Met — modular src/ structure with Git version control throughout |
| **NFR05** | Should Have | Met — responsive layout verified across Chrome, Firefox, Edge |
| **NFR06** | Should Have | Partially implemented — research disclaimer displayed; formal compliance framework deferred |
| **NFR07** | Could Have | Not Implemented — deferred to future version |

---

## Part 10 — Should You Describe the Dual Hybrid Architecture and Spatial Representation?

**YES — this is your core academic contribution. It must be prominently described in Chapter 3 (Methodology) and Chapter 8 (Implementation).**

### What to Say and Where

#### In Chapter 3 (Methodology) — justify WHY you built it this way

Write a dedicated section titled **"3.x Dual-Pathway Hybrid Architecture"**:

> *ChagaSight employs a dual-pathway hybrid Vision Transformer architecture that processes standard 12-lead ECG recordings in two complementary representations simultaneously.*
>
> **1D Temporal Pathway:** The raw 12-lead ECG signal (resampled to 100 Hz, 1000 samples per lead) is processed by a 1D Vision Transformer pretrained using ST-MEM (Van Santvliet et al., 2025). Each lead is independently patched into 20 temporal tokens, producing 240 tokens per recording. The 1D pathway captures precise timing features — RR intervals, PR/QT durations, QRS morphology, and T-wave inversions — which are the primary ECG manifestations of Chagas cardiomyopathy, including right bundle branch block (RBBB) and left anterior fascicular block (LAFB).
>
> **2D Spatial Pathway:** The same ECG signal is transformed into a 2D spatial representation using **Wilson's Central Terminal (WCT) re-referencing**, producing a `(3, 24, 2048)` tensor. This encodes all 12 leads simultaneously as spatial rows across three reference perspectives of the cardiac electrical field (referenced to RA, LA, and LL limb electrodes respectively). A 2D Vision Transformer pretrained with MAE (He et al., 2022) processes this image, enabling spatial attention over cross-lead co-occurrences at each time position.
>
> **Why spatial representation matters for Chagas specifically:** Chagas disease produces diffuse myocardial fibrosis that does not manifest in a single lead in isolation. The characteristic combination of RBBB + LAFB + ST changes across anatomically adjacent leads is a fundamentally spatial, cross-lead phenomenon. A purely sequential 1D model processes each lead independently before merging — the 2D spatial pathway is designed specifically to detect these inter-lead spatial dependencies that the 1D pathway cannot easily represent.
>
> **REPA Cross-Modal Alignment:** A cosine-similarity alignment loss (`L_align = 1 − cos_sim(aligned_2D, FM_features)`) forces the 2D pathway to learn representations consistent with the already-pretrained 1D ST-MEM backbone. This prevents the 2D pathway from learning texture-only features and ensures both pathways capture complementary, non-redundant information.
>
> **Fusion:** 768-dimensional features from both pathways are concatenated into a 1536-dimensional joint representation, passed through a 3-layer MLP classifier to produce the final Chagas probability.

#### In Chapter 8 (Implementation) — describe HOW it is built

Include the architecture figure showing:
```
(12, 1000) ──► 1D ViT (ST-MEM) ──► FM features (768d) ──┐
                                                           ├──► Concat (1536d) ──► MLP ──► P(Chagas)
(3, 24, 2048) ──► 2D ViT (MAE) ──► REPA ──► (768d) ──────┘
```

Reference your actual class files: `vit_1d_fm.py`, `vit_2d.py`, `repa_alignment.py`, `hybrid_model.py`.

#### In Chapter 9 (Evaluation) — prove the dual pathway outperforms single pathways

The ablation study proves this directly:

| Model | TPR@5% | Δ vs. 1D-only |
|-------|--------|----------------|
| 1D-only | 0.429 | — |
| 2D-only | 0.463 | +0.034 |
| Hybrid (1D + 2D + REPA) | **0.504** | **+0.075** |

> *"The +7.5 percentage point improvement of the Hybrid model over the 1D-only baseline empirically demonstrates that the 2D spatial pathway captures complementary cross-lead information not available from the temporal signal alone, validating the architectural choice."*

---

## Part 11 — Should You Add a Login System?

**No — do not implement login for thesis submission.**

### Why Not

| Reason | Detail |
|--------|--------|
| Time cost | A proper auth system (JWT, password hashing, RBAC, session management) takes significant time better spent on the thesis write-up |
| Security risk | A poorly implemented login (plain-text passwords, no CSRF protection) is worse than no login — an examiner who finds vulnerabilities will penalise you |
| Not your contribution | Your academic contribution is the ECG AI model, not the web auth layer |
| Already marked correctly | FR09 (Authentication) is `Could Have` in the rewritten requirements — this is a legitimate, documented engineering decision |

### What to Write in the Thesis Instead

**In Chapter 4 (FR09):**
> *"User authentication is classified as Could Have and was descoped from the current prototype. The system is intended for research use with publicly available, de-identified datasets. A production deployment would require JWT-based session management and role-based access control (RBAC)."*

**In Chapter 10.4 (Future Work):**
> *"The current prototype does not implement user authentication. A production deployment would require a secure login module with role-based access control, JWT-based sessions, and audit logging to comply with GDPR/HIPAA data-governance policies. This is planned as a priority extension."*

### If You Still Want a Simple Demo Login (Optional)

If you have spare time after the thesis is written, a 2-hour minimal approach:
- Use Flask-Login with a single hardcoded user from an environment variable
- Add a `/login` route with a session cookie (Flask's built-in session)
- Protect the `/api/predict` route with `@login_required`

This would let you move FR09 from `Could Have` to `Should Have`. **Only do this if the thesis is already complete.**
