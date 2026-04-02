# ChagaSight — Thesis Chapter 4 & 6: Requirements & Design Goals
## (Pansilu-style academic writing reference — UK English, humanised tone)

---


---

## 6.2 Design Goals

The desired design goals to be achieved by the system are specified in the table below. Each goal
is traceable to one or more non-functional requirements defined in Section 4.10.2 and guides
architectural and detailed design decisions throughout Chapters 5 and 6.

| ID | Goal | Description |
|----|------|-------------|
| **DG01** | Accuracy | The system must produce clinically meaningful Chagas disease screening predictions from standard 12-lead ECG recordings. This is achieved through a structured four-stage preprocessing pipeline, a dual-pathway Vision Transformer ensemble combining temporal and spatial ECG representations, and cross-modal REPA alignment to ensure complementary, non-redundant feature extraction. |
| **DG02** | Performance | The system must deliver inference results with minimal latency, sustaining a responsive user experience during screening. This is realised through inference-only deployment, preloaded model weights at server startup, and a modular preprocessing pipeline that avoids redundant computation. |
| **DG03** | Robustness | The system must demonstrate reliable and consistent behaviour across heterogeneous ECG datasets and variable recording conditions. Robustness is achieved through zero-phase baseline removal, independent per-lead Z-score normalisation, fixed-length windowing, and dual-frequency resampling, collectively ensuring stable signal representation irrespective of the source dataset or acquisition environment. |
| **DG04** | Scalability | The system design must accommodate future growth, including additional ECG datasets, an expanded user base, and potential cloud deployment. The separation of the presentation, application, and data tiers enables independent horizontal scaling of each layer without necessitating architectural redesign. |
| **DG05** | Modularity | System components — encompassing WFDB validation, signal preprocessing, dual-pathway embedding, ensemble inference, and result logging — must remain independent and cohesive, enabling targeted iterative improvement of individual modules without propagating changes across the system. |
| **DG06** | Usability | The system must be readily navigable for both technical and non-technical users, supporting straightforward WFDB file upload, intuitive model selection, and clear presentation of screening outcomes, without requiring familiarity with the underlying machine learning architecture. |
| **DG07** | Maintainability | The design must support efficient maintenance and debugging through clearly delineated component responsibilities and an unambiguous data flow from upload through preprocessing, inference, and result delivery. |
| **DG08** | Data Minimisation | Uploaded ECG files shall not be retained beyond the duration required to complete inference. The design enforces ephemeral file storage with immediate post-inference deletion, whilst persistent storage is limited to derived results and system logs. |

---

## Use Case Descriptions (Chapter 5 — System Design)

The following use case descriptions correspond to the UML Use Case Diagram presented in Figure 5.x.

| UC ID | Name | Actor | Description |
|-------|------|-------|-------------|
| **UC-01** | Upload ECG Recording | Clinician / Researcher | The user uploads a WFDB-format ECG file pair (`.hea` + `.dat`/`.mat`) via the frontend drag-and-drop interface or file selection dialogue. |
| **UC-02** | Select Model Mode | Clinician / Researcher | The user selects one of three available diagnostic model modes (Hybrid Ensemble, 2D Visual Model, or 1D Signal Model). Each mode displays its associated AUROC and TPR@5% metrics to support informed selection. |
| **UC-03** | Enter Patient Demographics | Clinician / Researcher | The user optionally provides patient age (in years) and biological sex. These values are passed to the inference pipeline and applied via FiLM conditioning within the 1D ViT backbone. *(Extends UC-05)* |
| **UC-04** | Load Sample ECG | Clinician / Researcher | The user selects a pre-loaded representative ECG from the SaMi-Trop, PTB-XL, or CODE-15% datasets for demonstration screening without supplying their own recording. *(Extends UC-01)* |
| **UC-05** | Request Chagas Screening | Clinician / Researcher | The user initiates the screening process by submitting the uploaded ECG and selected model configuration. This use case includes WFDB validation (UC-07), preprocessing (UC-08), model inference (UC-09), and file deletion (UC-10). |
| **UC-06** | View Prediction Result | Clinician / Researcher | The system displays the screening result comprising a probability percentage, a visual gauge, a binary classification label (Low Risk / High Risk), and a plain-language clinical interpretation. |
| **UC-07** | Validate WFDB File Pair | System | The system automatically validates that the uploaded files constitute a complete WFDB pair sharing a common base name, raising an error if the `.hea` file or its accompanying signal file is absent. *(Included by UC-05)* |
| **UC-08** | Preprocess ECG Signal | System | The system automatically applies the four-stage preprocessing pipeline: baseline removal, resampling (100 Hz for 1D; 500 Hz for 2D), per-lead normalisation, and WCT re-referencing to construct the dual-pathway input tensors. *(Included by UC-05)* |
| **UC-09** | Run Model Inference | System | The system executes inference using the selected model mode. For the Hybrid Ensemble, predictions are averaged across all five fold models. The result is returned as a probability score and a binary prediction. *(Included by UC-05)* |
| **UC-10** | Delete Uploaded Files | System | The system automatically removes all server-side copies of the uploaded WFDB files immediately upon completion of inference, ensuring no patient data is retained. *(Included by UC-05)* |
