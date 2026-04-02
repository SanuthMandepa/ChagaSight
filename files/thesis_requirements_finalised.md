# ChagaSight — Finalised Requirements (Chapter 4.10)

---

## 4.10.1 Functional Requirements

| ID | Requirement | Priority (MoSCoW) |
|----|-------------|-------------------|
| **FR01** | The system shall permit users to upload 12-lead ECG recordings in WFDB format, accepting paired `.hea` and `.dat` or `.mat` files that share a common base name. | Must Have |
| **FR02** | The system shall subject each uploaded ECG recording to a four-stage preprocessing pipeline, encompassing: zero-phase Butterworth bandpass filtering (0.5–40 Hz) for baseline removal; resampling to 100 Hz for the 1D temporal pathway and 500 Hz for the 2D spatial pathway; per-lead Z-score normalisation; and Wilson's Central Terminal re-referencing to construct a `(3, 24, 2048)` spatial image tensor. | Must Have |
| **FR03** | The system shall perform inference using a dual-pathway Vision Transformer ensemble, comprising a 1D signal pathway pretrained via ST-MEM and a 2D spatial pathway pretrained via a Masked Autoencoder objective, with cross-modal REPA alignment enforced during training. Predictions shall be aggregated across five cross-validation fold models to produce a final ensemble probability score. | Must Have |
| **FR04** | The system shall present the inference result as a probability percentage, a visual gauge indicator, a binary classification label (Low Risk or High Risk), and a plain-language clinical interpretation string. | Must Have |
| **FR05** | The system shall enable users to select between three diagnostic model modes — Hybrid Ensemble, 2D Visual Model, and 1D Signal Model — and shall display the associated AUROC metric for each mode to assist informed selection. | Should Have |
| **FR06** | The system shall provide a sample ECG loader through which users may select pre-loaded representative recordings drawn from the SaMi-Trop, PTB-XL, and CODE-15% datasets, facilitating demonstration and functional evaluation without requiring user-supplied data. | Should Have |
| **FR07** | The system shall accept optional patient demographic inputs — specifically age (in years) and biological sex — and shall incorporate these values into the 1D model inference pathway via FiLM conditioning layers within the 1D Vision Transformer backbone. | Should Have |
| **FR08** | The system shall generate a downloadable PDF report encapsulating the probability score, classification label, and interpretation text, enabling users to retain or share the screening result. | Could Have |
| **FR09** | The system shall enforce user authentication to restrict access to authorised personnel within a clinical deployment context, incorporating role-based access control and session management. | Will Not Have |
| **FR10** | The system shall expose an administrative interface for the upload, labelling, and management of new ECG recordings, thereby supporting iterative model retraining and dataset expansion. | Will Not Have |

> **Note on FR08–FR10:** FR08 is technically feasible and is identified as a near-term extension. FR09 and FR10 have been consciously descoped, as the system is designed for research use with de-identified publicly available datasets.
>
> **Note on FR07:** FR07 is classified as Should Have and is implemented in the current prototype. Patient age and biological sex are incorporated into the 1D ViT backbone via FiLM conditioning at inference time.

---

## 4.10.2 Non-Functional Requirements

| ID | Non-Functional Requirement | Description | Priority (MoSCoW) |
|----|---------------------------|-------------|-------------------|
| **NFR01** | Accuracy | The model shall achieve AUROC ≥ 0.85 and a screening sensitivity score (TPR@5% FPR) ≥ 0.40 on the held-out cross-validation test set. F1 score is reported as a secondary metric but is not used as a pass/fail criterion due to extreme class imbalance (2.22% positive prevalence). | Must Have |
| **NFR02** | Performance | The system shall return a prediction result within 10 seconds of file upload for a standard 10-second WFDB ECG recording. | Must Have |
| **NFR03** | Security and Data Protection | Uploaded ECG files shall be deleted from server storage immediately upon completion of inference, ensuring no patient data is retained beyond the duration strictly necessary. | Must Have |
| **NFR04** | Maintainability | The codebase shall be organised into discrete functional modules — encompassing preprocessing, model architectures, training, evaluation, API serving, and the frontend — following separation of concerns principles, with version control maintained throughout development using Git. | Should Have |
| **NFR05** | Usability | The frontend interface shall be responsive across standard desktop screen sizes and browsers including Chrome, Firefox, and Edge. | Should Have |
| **NFR06** | Compliance | The system shall display a research disclaimer on all prediction result views and shall operate exclusively with de-identified, publicly available datasets. | Should Have |
| **NFR07** | Explainability | The system should provide lead-wise attention weight visualisation to support clinical interpretability of predictions, enabling clinicians to identify which ECG leads contributed most strongly to a screening result. | Could Have |

> **Note on NFR07:** Lead-wise attention visualisation is classified as Could Have and is not implemented in the current prototype. The backend architecture supports future extraction of transformer attention weights at inference time, and implementation is identified as a priority near-term extension.

---

## FR Implementation Status Summary (for Chapter 8 reference)

| ID | Priority | Implemented? |
|----|----------|--------------|
| FR01 | Must Have | Yes |
| FR02 | Must Have | Yes |
| FR03 | Must Have | Yes |
| FR04 | Must Have | Yes |
| FR05 | Should Have | Yes |
| FR06 | Should Have | Yes |
| FR07 | Should Have | Yes |
| FR08 | Could Have | No — deferred |
| FR09 | Will Not Have | No — descoped |
| FR10 | Will Not Have | No — descoped |

*Must Have completion: 4/4 (100%). Should Have completion: 3/3 (100%). Could Have completion: 0/1 (0%).*

## NFR Evaluation Status Summary (for Chapter 8 reference)

| ID | Priority | Status |
|----|----------|--------|
| NFR01 | Must Have | **Met** — AUROC 0.8707 (target ≥ 0.85 ✓); TPR@5% 0.4958 (target ≥ 0.40 ✓) |
| NFR02 | Must Have | **Met** — inference consistently under 10 seconds |
| NFR03 | Must Have | **Met** — `_cleanup()` called in `finally` block of `/api/predict` |
| NFR04 | Should Have | **Met** — modular `src/` structure; Git version control throughout |
| NFR05 | Should Have | **Met** — verified on Chrome, Firefox, Edge at 1920×1080 and 1366×768 |
| NFR06 | Should Have | **Met** — disclaimer displayed; de-identified datasets used |
| NFR07 | Could Have | **Not Met** — attention visualisation not implemented; deferred to future work |
