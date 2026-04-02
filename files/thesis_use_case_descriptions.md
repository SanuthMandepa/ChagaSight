# ChagaSight — 4.9 Use Case Descriptions (Corrected)

> All use cases below reflect the **implemented** ChagaSight prototype.  
> UC-04 Export Report, UC-05 Manage Datasets, and UC-06 Configure System from the original SRS have been **removed** as they are not implemented and have been reclassified as future work in the functional requirements (FR08–FR10, Section 4.10.1).

---

## Use Case Diagram Summary

```
Clinician/Researcher ─── UC-01  Upload ECG Recording
                    ─── UC-02  Select Model Mode
                    ─── UC-03  Enter Patient Demographics  ──«extend»──► UC-05
                    ─── UC-04  Load Sample ECG             ──«extend»──► UC-01
                    ─── UC-05  Request Chagas Screening
                    │         ──«include»──► UC-07  Validate WFDB File Pair
                    │         ──«include»──► UC-08  Preprocess ECG Signal
                    │         ──«include»──► UC-09  Run Model Inference
                    │         ──«include»──► UC-10  Delete Uploaded Files
                    └── UC-06  View Prediction Result
```

---

## UC-01 — Upload ECG Recording

| Field | Detail |
|-------|--------|
| **Use Case** | Upload ECG Recording |
| **ID** | UC-01 |
| **Description** | Describes how a clinician or researcher uploads a 12-lead ECG recording into the ChagaSight system for Chagas disease screening. |
| **Actor** | Clinician / Researcher |
| **Supporting Actor(s)** | None |
| **Stakeholders** | Clinicians, medical researchers, medical students |
| **Pre-conditions** | The user has access to the ChagaSight web interface. A valid 12-lead ECG recording in WFDB format (`.hea` + `.dat` or `.mat` file pair sharing a common base name) is available. |
| **Extended Use Cases** | *(None — UC-04 extends this use case when the user opts for a pre-loaded sample instead)* |
| **Included Use Cases** | *(None directly — UC-07 Validate WFDB File Pair is triggered upon screening in UC-05)* |
| **Main Flow** | 1. User navigates to the ChagaSight upload panel. <br>2. User selects or drags and drops a `.hea` and `.dat`/`.mat` file pair. <br>3. The frontend validates that two files with matching base names are provided. <br>4. Files are submitted to the Flask backend for inference processing. |
| **Alternate Flows** | AF1: User cancels the upload before submission — no files are sent. |
| **Exceptional Flows** | EF1: Only one file of the pair is provided — system displays an error requesting the complete WFDB file pair. <br>EF2: File extension is not `.hea`, `.dat`, or `.mat` — system rejects the upload with a format error. |
| **Post-conditions** | The WFDB file pair is received by the backend and available for screening via UC-05. |

---

## UC-02 — Select Model Mode

| Field | Detail |
|-------|--------|
| **Use Case** | Select Model Mode |
| **ID** | UC-02 |
| **Description** | Allows the user to select between three available diagnostic model modes prior to screening, each with distinct architectural characteristics and reported performance metrics. |
| **Actor** | Clinician / Researcher |
| **Supporting Actor(s)** | None |
| **Stakeholders** | Clinicians, medical researchers, medical students |
| **Pre-conditions** | The ChagaSight interface is loaded and at least one model checkpoint is available on the backend. |
| **Extended Use Cases** | *(None)* |
| **Included Use Cases** | *(None)* |
| **Main Flow** | 1. User views the model selection panel displaying three modes: Hybrid Ensemble, 2D Visual Model, and 1D Signal Model. <br>2. Each mode card displays its AUROC and screening sensitivity (TPR@5%) metric. <br>3. User selects a preferred mode. <br>4. The selected mode is stored in the frontend state and passed to the backend upon screening. |
| **Alternate Flows** | AF1: User changes model selection after an initial screening — a new screening request is initiated with the newly selected mode. |
| **Exceptional Flows** | EF1: The selected model checkpoint is unavailable on the server — system returns a 503 error with a descriptive message. |
| **Post-conditions** | The desired model mode is configured and will be submitted with the subsequent screening request (UC-05). |

---

## UC-03 — Enter Patient Demographics

| Field | Detail |
|-------|--------|
| **Use Case** | Enter Patient Demographics |
| **ID** | UC-03 |
| **Description** | The user optionally provides patient age and biological sex to enable demographic-conditioned inference via FiLM modulation within the 1D Vision Transformer backbone. This use case extends UC-05. |
| **Actor** | Clinician / Researcher |
| **Supporting Actor(s)** | None |
| **Stakeholders** | Clinicians, medical researchers |
| **Pre-conditions** | The ChagaSight interface is loaded. This step is optional; the system defaults to age = 50 and sex = unknown if omitted. |
| **Extended Use Cases** | **Extends UC-05** — demographic inputs are incorporated into the screening request when provided. |
| **Included Use Cases** | *(None)* |
| **Main Flow** | 1. User locates the demographic input panel alongside the upload interface. <br>2. User enters patient age in years using the numeric input. <br>3. User selects biological sex (Male / Female / Unknown) from the dropdown. <br>4. Values are submitted alongside the ECG files when UC-05 is initiated. |
| **Alternate Flows** | AF1: User leaves demographic fields empty — system applies default values (age = 50, sex = unknown = 0.0) without error. |
| **Exceptional Flows** | EF1: An invalid age value (e.g. negative or non-numeric) is entered — frontend validation prevents submission. |
| **Post-conditions** | Demographic values are passed to the 1D ViT backbone as FiLM conditioning inputs (age normalised to centuries; sex as binary float). |

---

## UC-04 — Load Sample ECG

| Field | Detail |
|-------|--------|
| **Use Case** | Load Sample ECG |
| **ID** | UC-04 |
| **Description** | The user selects a pre-loaded representative ECG recording from the SaMi-Trop, PTB-XL, or CODE-15% datasets for demonstration or evaluation purposes, bypassing the need to provide a personal WFDB file. This use case extends UC-01. |
| **Actor** | Clinician / Researcher |
| **Supporting Actor(s)** | None |
| **Stakeholders** | Clinicians, medical researchers, medical students |
| **Pre-conditions** | Sample ECG files are available on the server in the designated samples directory. |
| **Extended Use Cases** | **Extends UC-01** — the sample ECG replaces the user-uploaded file as the input to the screening pipeline. |
| **Included Use Cases** | *(None)* |
| **Main Flow** | 1. User selects the "Load Sample ECG" option in the interface. <br>2. A list of available sample recordings from SaMi-Trop, PTB-XL, and CODE-15% is displayed. <br>3. User selects a sample recording. <br>4. The system loads the selected sample and pre-populates the upload panel, ready for screening. |
| **Alternate Flows** | AF1: User loads a sample but subsequently uploads their own file — the sample selection is replaced with the uploaded file. |
| **Exceptional Flows** | EF1: A requested sample file cannot be located on the server — system displays an error and falls back to manual upload. |
| **Post-conditions** | A sample WFDB recording is loaded and ready for screening via UC-05 without requiring the user to supply their own ECG data. |

---

## UC-05 — Request Chagas Screening

| Field | Detail |
|-------|--------|
| **Use Case** | Request Chagas Screening |
| **ID** | UC-05 |
| **Description** | The user initiates the end-to-end Chagas disease screening pipeline. This is the primary system use case, encapsulating WFDB validation, ECG preprocessing, model inference, and file cleanup as mandatory included behaviours. |
| **Actor** | Clinician / Researcher |
| **Supporting Actor(s)** | None |
| **Stakeholders** | Clinicians, medical researchers, medical students |
| **Pre-conditions** | At least one WFDB file pair has been uploaded or loaded via UC-01 or UC-04. A model mode has been selected via UC-02. Demographics may optionally have been entered via UC-03. |
| **Extended Use Cases** | *(UC-03 extends this use case when demographics are provided)* |
| **Included Use Cases** | **UC-07** Validate WFDB File Pair, **UC-08** Preprocess ECG Signal, **UC-09** Run Model Inference, **UC-10** Delete Uploaded Files |
| **Main Flow** | 1. User clicks "Analyse ECG". <br>2. System validates the WFDB file pair (UC-07). <br>3. System preprocesses the ECG signal through the four-stage pipeline (UC-08). <br>4. System runs inference using the selected model mode (UC-09). <br>5. System deletes the uploaded files from server storage (UC-10). <br>6. Screening result is returned to the frontend and displayed via UC-06. |
| **Alternate Flows** | AF1: User submits without a loaded ECG — frontend prevents submission and displays an upload prompt. |
| **Exceptional Flows** | EF1: WFDB validation fails (missing `.hea` or signal file) — system returns a 400 error with a descriptive message; no inference is performed. <br>EF2: Model checkpoint is unavailable — system returns a 503 error. <br>EF3: An unexpected exception occurs during preprocessing or inference — system returns a 400 error and cleans up uploaded files in the `finally` block. |
| **Post-conditions** | A Chagas disease probability score, binary prediction, and interpretation string are returned to the frontend. All uploaded ECG files are deleted from the server. |

---

## UC-06 — View Prediction Result

| Field | Detail |
|-------|--------|
| **Use Case** | View Prediction Result |
| **ID** | UC-06 |
| **Description** | The system displays the inference result returned from UC-05 in a structured, clinically accessible format comprising a probability gauge, classification label, and plain-language interpretation. |
| **Actor** | Clinician / Researcher |
| **Supporting Actor(s)** | None |
| **Stakeholders** | Clinicians, medical researchers, medical students |
| **Pre-conditions** | A successful screening request (UC-05) has been completed and a result JSON has been received from the backend. |
| **Extended Use Cases** | *(None)* |
| **Included Use Cases** | *(None)* |
| **Main Flow** | 1. System receives the JSON prediction response from the Flask API. <br>2. Frontend renders the probability score as a visual gauge and numerical percentage. <br>3. Frontend displays the binary classification label: "Low Risk" or "High Risk". <br>4. Frontend displays a plain-language interpretation string reflecting the prediction. <br>5. Frontend displays the model metrics (AUROC, TPR@5%) associated with the selected model mode. <br>6. A research disclaimer is presented alongside the result. |
| **Alternate Flows** | AF1: User changes model mode and resubmits the same ECG — the result panel is refreshed with the new prediction. |
| **Exceptional Flows** | EF1: The API returns an error response — the frontend displays the error message and invites the user to retry. |
| **Post-conditions** | The clinician can review the Chagas risk assessment. No data is stored persistently on the server. |

---

## UC-07 — Validate WFDB File Pair *(System)*

| Field | Detail |
|-------|--------|
| **Use Case** | Validate WFDB File Pair |
| **ID** | UC-07 |
| **Description** | The system automatically verifies that the submitted files constitute a valid WFDB pair with both a `.hea` header and a matching `.dat` or `.mat` signal file prior to preprocessing. |
| **Actor** | System (automated) |
| **Supporting Actor(s)** | None |
| **Pre-conditions** | Files have been received server-side via the `/api/predict` endpoint. |
| **Included by** | UC-05 |
| **Main Flow** | 1. System scans the submitted files for a `.hea` file. <br>2. System checks that a `.dat` or `.mat` file sharing the same base name is present. <br>3. Validation passes — processing continues to UC-08. |
| **Exceptional Flows** | EF1: No `.hea` file found — raises `ValueError("Missing .hea file")` and returns 400. <br>EF2: No matching signal file found — raises `ValueError("Missing matching .dat or .mat file")` and returns 400. |
| **Post-conditions** | WFDB file pair is confirmed valid; record base path and name are extracted for preprocessing. |

---

## UC-08 — Preprocess ECG Signal *(System)*

| Field | Detail |
|-------|--------|
| **Use Case** | Preprocess ECG Signal |
| **ID** | UC-08 |
| **Description** | The system applies the four-stage ECG preprocessing pipeline to produce the dual input tensors required by the model: a `(3, 24, 2048)` spatial image tensor for the 2D pathway and a `(12, 1000)` signal tensor for the 1D pathway. |
| **Actor** | System (automated) |
| **Supporting Actor(s)** | None |
| **Pre-conditions** | WFDB file pair has been validated (UC-07). |
| **Included by** | UC-05 |
| **Main Flow** | 1. System reads the WFDB recording using `wfdb.rdsamp`, producing a `(T, 12)` signal array. <br>2. System applies zero-phase Butterworth bandpass filtering (0.5–40 Hz) for baseline removal. <br>3. **2D path:** Signal is resampled to 500 Hz, per-lead Z-score normalised (±3σ clipped), and transformed via Wilson's Central Terminal re-referencing into a `(3, 24, 2048)` spatial image tensor. <br>4. **1D path:** Signal is resampled to 100 Hz, per-lead Z-score normalised, and padded or trimmed to exactly 1000 samples, producing a `(12, 1000)` signal tensor. |
| **Exceptional Flows** | EF1: Loaded signal does not have exactly 12 leads — raises `ValueError` and returns 400. <br>EF2: Constructed image shape does not match `(3, 24, 2048)` — raises `ValueError` and returns 400. |
| **Post-conditions** | Two input tensors are prepared on the inference device: `img_t (1, 3, 24, 2048)` and `sig_t (1, 12, 1000)`. |

---

## UC-09 — Run Model Inference *(System)*

| Field | Detail |
|-------|--------|
| **Use Case** | Run Model Inference |
| **ID** | UC-09 |
| **Description** | The system executes the selected model on the preprocessed input tensors to produce a Chagas disease probability score and binary classification. |
| **Actor** | System (automated) |
| **Supporting Actor(s)** | None |
| **Pre-conditions** | Input tensors have been prepared by UC-08. The selected model checkpoint is loaded in memory. Demographic tensors (age, sex) have been parsed. |
| **Included by** | UC-05 |
| **Main Flow** | 1. System reads the `model_type` parameter from the request (`hybrid`, `2d`, or `1d`). <br>2. **Hybrid:** System averages sigmoid-transformed logits across all five fold models to produce a final probability; applies the calibrated ensemble threshold. <br>3. **2D-only:** System passes `img_t` through the 2D ViT classifier; applies threshold = 0.5. <br>4. **1D-only:** System passes `sig_t`, `ages`, and `sexes` through the 1D ViT-FM classifier; applies threshold = 0.5. <br>5. Binary prediction is derived: `prediction = 1 if probability ≥ threshold else 0`. <br>6. Result JSON is assembled including probability, threshold, prediction, folds used, model metrics, and interpretation string. |
| **Exceptional Flows** | EF1: Selected model is not loaded — returns 503 with a descriptive error. <br>EF2: Runtime error during inference — exception propagated to UC-05 exception handler. |
| **Post-conditions** | A result JSON containing the probability score, binary label, and model metrics is returned to UC-05 for delivery to the frontend. |

---

## UC-10 — Delete Uploaded Files *(System)*

| Field | Detail |
|-------|--------|
| **Use Case** | Delete Uploaded Files |
| **ID** | UC-10 |
| **Description** | The system automatically removes all server-side copies of the uploaded WFDB files immediately upon completion of the inference pipeline, ensuring no patient data is retained beyond the duration strictly necessary. |
| **Actor** | System (automated) |
| **Supporting Actor(s)** | None |
| **Pre-conditions** | Files were saved to the `uploads/` directory during UC-05. |
| **Included by** | UC-05 |
| **Main Flow** | 1. Upon completion or failure of the inference pipeline, the `finally` block in `app.py` calls `_cleanup(saved)`. <br>2. Each saved file path is individually unlinked using `Path.unlink(missing_ok=True)`. <br>3. Files are deleted regardless of whether inference succeeded or raised an exception. |
| **Exceptional Flows** | EF1: A file cannot be unlinked due to a permission error — exception is silently caught to avoid masking the primary inference response. |
| **Post-conditions** | All uploaded ECG files have been removed from the server. No patient data is retained in persistent storage. |
