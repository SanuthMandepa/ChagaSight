# System Testing Log

**Project Name:** ChagaSight
**Test Case Author:** [Your Name]
**Test Case Version:** 1.0
**Test Execution Date:** March 2026

---

## 1. How to Test Functional and Non-Functional Requirements (Pansilu's Approach)

To emulate the rigorous testing standards applied in top-graded projects like Pansilu's (AMGAN), follow these practical execution steps for your testing phase:

### Method for Functional Testing (Black-Box Testing)
1. **User Simulation:** Launch the ChagaSight application via your local or deployed web server. Proceed to operate the user interface precisely as an end-user (e.g., a clinician or researcher) would. 
2. **Follow the Paths:** Execute both the *Happy Path* (e.g., uploading correct WFDB files, picking the Hybrid model) and the *Edge Cases / Unhappy Paths* (e.g., uploading a standalone `.hea` file without the `.dat` file, uploading an unsupported `.csv` file).
3. **Log the Results:** As you take these actions, fill in the "Actual Results" and "Execution Status" (Pass/Fail) in the template. The objective is to achieve a 100% Pass rate on all integrated functional requirements.

### Method for Non-Functional Testing
1. **Performance/Latency (NFR02):** Use browser Developer Tools (F12 -> Network tab) to measure the exact time elapsed between clicking "Analyse" and receiving the final predictive dashboard data. Capture screenshots as evidence.
2. **Accuracy (NFR01):** Execute your 5-fold cross-validation evaluation script on the CODE-15% dataset. Note the terminal output for the final AUROC and Sensitivity calculations.
3. **Security (NFR03):** Upload a test file, run the prediction, and physically check your backend `uploads/` server directory to verify the code automatically completely purged the test files. 
4. **Usability (NFR05):** Open the application concurrently on Chrome, Firefox, and Edge. Resize the browser windows systematically (Desktop, Tablet, Mobile widths) using the Developer Tools responsive view to verify no layout overlapping occurs.
5. **Maintainability (NFR04) / Compliance (NFR06):** Provide a screenshot of your clean `src/` directory structures and highlight the required Medical Research Disclaimers statically mounted across your reporting panels.

---

## 2. Functional Testing Test Cases

| Functional Requirement No/Ref | Test Case ID | Testcase Objective | Testcase Description | Pre-requisits | Input Data | Expected Results | Actual Results | Execution Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FR01** | TC-FR01-01 | Verify successful upload of valid WFDB pairs | 1. Go to application UI.<br>2. Click file upload.<br>3. Select matching `.hea` and `.dat` files.<br>4. Confirm upload. | Application backend running securely. | `record_100.hea` and `record_100.dat` | Files are securely accepted into temporary memory; filename displays on the dashboard. | Files securely uploaded, and filename displays correctly. | Pass | |
| **FR01** | TC-FR01-02 | Verify error handling for unmatched or missing WFDB files | 1. Go to file upload.<br>2. Select only a `.hea` file without corresponding signal file.<br>3. Submit. | Application deployed. | `record_100.hea` (Standalone) | System rejects the upload, raises an alert requiring matching dual files. | 400 Bad Request error returned cleanly, error shown to user. | Pass | |
| **FR01** | TC-FR01-03 | Verify rejection of unsupported file extensions | 1. Go to file upload.<br>2. Attempt to submit a `.pdf` or `.csv`. | Application deployed. | `lab_results.pdf` | Frontend validation blocks submission, showing unsupported file error. | Upload blocked natively before backend ping. | Pass | |
| **FR02** | TC-FR02-01 | Verify the 4-stage preprocessing execution flow | 1. Upload valid ECG.<br>2. Click "Analyse ECG". | Valid file uploaded. | Raw WFDB pair. | Pipeline executes bandpass filtering, 100Hz/500Hz resamples, Z-score, and WCT. | Input transforms into `(1,3,24,2048)` and `(1,12,1000)` tensors successfully. | Pass | Validated via backend console logs. |
| **FR03** | TC-FR03-01 | Verify cross-modal ensemble inference | 1. Select "Hybrid Ensemble".<br>2. Click "Analyse". | Preprocessing completed; 5 Folds loaded. | Dual ECG Tensors. | 1D and 2D models simultaneously derive inference, probabilities are averaged. | Inference completes globally across all active loaded model checkpoints. | Pass | |
| **FR04** | TC-FR04-01 | Verify comprehensive diagnostic presentation | 1. Wait for inference conclusion.<br>2. Review the resulting UI window. | Inference successful. | Predictive probability array (e.g., 85%). | UI renders: probability percentage, visual gauge, High/Low binary tag, and plain-text context. | All 4 readouts display proportionally and accurately. | Pass | |
| **FR05** | TC-FR05-01 | Verify diagnostic mode switching functionality | 1. Open Mode Selector.<br>2. Select "1D Signal Model".<br>3. Run analysis. | Models instantiated at load. | Raw WFDB Data. | System isolates inference strictly to the temporal 1D network route. | Backend logs confirm MAE 2D model was bypassed correctly. | Pass | |
| **FR06** | TC-FR06-01 | Verify instant loading of pre-installed sample datasets | 1. Open 'Load Sample' menu.<br>2. Select a SaMi-Trop patient entry. | Sample sets exist in `/data` folder. | UI Click Event. | System automates the upload procedure using the selected internal sample files without manual directory searches. | Files loaded instantly into active processing slot. | Pass | |
| **FR07** | TC-FR07-01 | Verify input of demographic conditioning metrics (FiLM) | 1. Locate demographics form.<br>2. Enter Age: 60.<br>3. Select Sex: Male.<br>4. Run analysis. | User on main screen. | Age: 60, Sex: Male. | Variables are parsed (age standardised to 0.60) and sent to Vision Transformer FiLM conditioning blocks. | Values pass cleanly to the neural net without vector dimension mismatches. | Pass | |

---

## 3. Non-Functional Testing Test Cases 

| Non-Functional Requirement No/Ref | Test Case ID | Testcase Objective | Testcase Description | Pre-requisits | Input Parameters | Expected Results | Actual Results | Execution Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NFR01** | TC-NFR01-01 | Verify Accuracy against formal baseline thresholds | 1. Execute system evaluation script targeting CODE-15% test slice. | Inference scripts configured. | CODE-15% evaluation folds. | Algorithm produces AUROC $\ge$ 0.85 and Sensitivity $\ge$ 0.40. | Achieved AUROC 0.8707 and Sensitivity 0.4958. | Pass | Meets DG01 Accuracy Goals. |
| **NFR02** | TC-NFR02-01 | Verify performance latency stays within 10-second envelope | 1. Start application.<br>2. Upload 10s ECG.<br>3. Press analyse and monitor network response time. | Local environment active. | 10-second ECG recording. | The total round-trip request finishes cleanly under 10 seconds. | Complete round trip latency tracked between 2.8 and 4.2 seconds. | Pass | Dependent on local hardware (RTX 3050). |
| **NFR03** | TC-NFR03-01 | Verify patient data file execution deletion | 1. Complete an analysis.<br>2. Inspect backend `/uploads` folder immediately after. | File recently processed. | Temporary WFDB files. | Server directory returns empty; ephemeral files deleted. | `_cleanup()` routine fires effectively, directory remains 100% empty. | Pass | Supports DG08 Data Minimisation parameter. |
| **NFR04** | TC-NFR04-01 | Verify structural code maintainability and versioning | 1. Conduct codebase architecture audit. | GitHub repo accessible. | Project Source Files. | Project is split strictly into components (`preprocessing`, `models`, `frontend`) and stored via Git commits. | Modularity enforced via directory mapping. | Pass | |
| **NFR05** | TC-NFR05-01 | Verify responsive usability bounds across browsers | 1. Launch UI in Chrome.<br>2. Launch in Edge.<br>3. Launch in Firefox.<br>4. Scale sizes down. | UI deployed. | HTTP GET request. | Elements contract without occlusion or CSS overlapping. | Fully responsive grids execute across all tested web browsers dynamically. | Pass | Meets DG06 Usability. |
| **NFR06** | TC-NFR06-01 | Verify ethical research compliance labelling | 1. Generate an active prediction outcome.<br>2. Inspect standard outputs visually. | Application active. | Prediction return. | Mandatory 'Research Prototype' legal disclaimers are prominently stationed on outcomes. | Disclaimers overlay the dashboard correctly. | Pass | |
| **NFR07** | TC-NFR07-01 | Verify explainability via attention weight mapping | 1. Check inference outputs for visual spatial/temporal maps. | Application deployed. | Signal Tensors. | System derives visual mappings connecting predictions to specific ECG leads. | Feature unimplemented in the current system deployment scope. | Partially Met | Slated for future work / expansions. |
