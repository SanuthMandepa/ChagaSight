# Chapter 9 : Critical Evaluation

## 9.1 Chapter Overview
This chapter presents a critical evaluation of the ChagaSight system by assessing it against the initial functional and non-functional requirements. A qualitative methodology was employed to gather structured feedback from healthcare professionals, software engineers, and machine learning researchers. The thematic analysis of this feedback highlighted the clinical value of the dual pathway architecture alongside the intuitive nature of the frontend interface. Finally, the chapter outlines current limitations and confirms that the core objectives of developing a responsive diagnostic prototype have been successfully achieved.

## 9.2 Evaluation Methodology and Approach
The main purpose of the critical evaluation phase is to verify the real world viability, technical quality, and overall effectiveness of the completed solution. To meet this goal, a qualitative evaluation approach was chosen using structured online questionnaires. These forms were carefully tailored to different professional groups to capture specific insights relevant to their distinct areas of expertise.

A working prototype of the ChagaSight application was hosted online and shared with the selected evaluators. Detailed architectural diagrams and model performance metrics were provided alongside the link. Evaluators interacted with the interface and reviewed the methodological decisions before submitting their feedback. Google Forms served as the primary data collection tool because it ensured a streamlined and accessible process for busy professionals.

The collected responses were then subjected to thematic analysis. This method allowed for the identification of recurring patterns in the feedback regarding clinical utility, codebase organisation, and interface accessibility. By separating the technical evaluation from the clinical reviews, the methodology ensured that every component of the system was scrutinised by individuals possessing the necessary subject matter expertise.

## 9.3 Evaluation Criteria
The specific criteria used to evaluate the clinical, theoretical, and engineering aspects of the system are detailed in Appendix H.1.

## 9.4 Self-Evaluation
An honest self evaluation was conducted to reflect on the project execution and the final outcomes. A portion of this structured self reflection is presented below, while the remainder of the evaluation addressing the prototype and system limitations can be found in Appendix H.2.

*Table 9.2: Self evaluation covering the core concept, scope, and technical implementation.*

| Theme | Evaluation by the author |
|---|---|
| **Clinical Rationale and Technical Novelty** | The concept of using a dual-pathway Vision Transformer ensemble to analyse 12-lead electrocardiograms is a highly novel approach to detecting Chagas disease complications. Unlike traditional automated systems that typically rely entirely on processing flat temporal signals, the approach presented here processes both the raw 1D signal and a dynamically constructed 2D spatial image. By linking these pathways with a cross-modal representation alignment module, the model successfully captures both fine temporal changes and spatial relationships across the different leads. This strategy pushes forward what is possible in automated cardiac risk stratification, taking a significant step towards creating viable, AI-assisted screening tools that can genuinely operate in resource-constrained medical environments. |
| **Project Scope and Data Complexity** | While medical AI research is usually driven by well funded laboratories with access to enterprise hardware, this project successfully built and fine-tuned a state-of-the-art framework on a standard consumer laptop GPU. Navigating a vast dataset of over 380,000 recordings with a severe class imbalance of just 2.22% required a deep understanding of advanced loss functions, particularly asymmetric binary cross-entropy. The research scope was carefully balanced to ensure technical rigor without sacrificing realistic implementation timelines. Delivering self-supervised pre-training, five-fold cross validation, and an interactive diagnostic prototype all within a constrained undergraduate timeframe easily exceeds the standard expectations of a final year software engineering project. |
| **Decoupled System Architecture** | Modularity and future scalability were the driving priorities when planning the system architecture. By choosing to strictly separate the heavy Python and Flask deep learning backend from the lightweight React frontend, the project inherently avoids the common pitfalls of monolithic design. This decoupled structure follows the best practices seen in commercial medical software, ensuring that future researchers could swap out the underlying predictive model or rebuild the visual interface without breaking the entire system. Furthermore, carefully designing the prediction endpoints to automatically aggregate inference across five parallel fold models demonstrates a strong grasp of creating reliable fault-tolerant data pipelines. |
| **Model Optimization and Inference** | Successfully implementing the machine learning core required overcoming significant technical limitations. Written entirely in PyTorch, the system had to use aggressive gradient accumulation and mixed precision training to process the large Vision Transformer backbones within a 6 GB memory cap. The training pipeline itself was highly structured, employing progressive fine-tuning and strict cross-validation to prevent the model from overfitting to the massive number of negative cases. The final evaluation metrics, particularly the strong AUROC performance, confirm that the model learns efficiently and generates clinically relevant predictions. For future improvements, applying quantization or knowledge distillation could further strip down the model weights to allow it to run on even lower spec hospital machines. |

## 9.5 Selection of the Evaluators
The evaluation panel consisted of 11 professionals selected based on their expertise in fields relevant to the ChagaSight project. The group included five healthcare professionals to assess clinical viability, one machine learning researcher to review the model architecture, four software engineers to evaluate the system design, and one senior web designer to inspect the human computer interaction factors. This diverse selection ensured that the evaluation covered all operational and technical facets of the prototype. The specific roles and affiliations of the evaluators are detailed in Appendix H.3.

## 9.6 Evaluation Results

### 9.6.1 Expert Opinion
Gathering expert opinions was vital for confirming that the research outcome holds practical value outside a purely academic setting. The responses were divided into two main sections: domain experts focusing on the healthcare context and technical experts reviewing the underlying engineering architecture.

### 9.6.2 Domain Experts
The complete qualitative feedback collected from the healthcare professionals regarding the system's clinical utility is detailed in Appendix H.4.

#### 9.6.2.1 Concept
The healthcare professionals unanimously agreed that the project addresses a highly meaningful clinical problem. Evaluators recognised the immense value of using automated electrocardiogram screening in resource limited hospitals where access to specialist cardiology services is rare. They noted that identifying patients at risk of cardiac complications from Chagas disease remains a genuine unmet need. The core concept of an automated triage system was viewed as highly applicable to current medical challenges.

#### 9.6.2.2 Solution
When reviewing the proposed solution, the doctors and medical students found the diagnostic output format clear and very supportive of medical decision making. They appreciated the inclusion of a quantitative percentage alongside a binary classification because it assists directly with risk stratification. However, they expressed valid concerns regarding patient safety and the raw accuracy of the generated predictions. The evaluators stressed that false negative rates must be effectively minimised before such a tool can be adopted safely in live wards. The general consensus showed that while the current setup is a compelling proof of concept, rigorous clinical trials are mandatory improvements for future versions.

### 9.6.3 Technical Experts
The full, structured evaluation data collected from the technical reviewers, including the machine learning researcher, software engineers, and the senior web designer, can be referenced in Appendix H.5, Appendix H.6, and Appendix H.7, respectively.

#### 9.6.3.1 Scope
The technical evaluators concluded that the project scope easily exceeds the standard expected of an undergraduate software engineering degree. The single-researcher implementation of a complete machine learning inference pipeline alongside a supporting web frontend was commended as highly ambitious. The machine learning researcher specifically highlighted that the integration of a dual-pathway architecture with cross-modal alignment displays an impressive grasp of complex algorithms.

#### 9.6.3.2 Architecture of the Solution
The software engineering experts reviewed the structural decisions and found them highly appropriate for a research prototype. The separation of concerns between the backend application programming interface and the client frontend was praised as a standard industry practice. The evaluators pointed out that this decoupled design naturally supports rapid experimentation and simple scaling. Additionally, placing the ensemble model behind dedicated prediction endpoints ensures a clean boundary between request handling logic and heavy computational tasks.

#### 9.6.3.3 Implementation of the Solution
The overall quality of the codebase received highly positive feedback for its clear organisation and consistent naming conventions. Software engineers noted that modularising the preprocessing steps and inference logic keeps the codebase highly maintainable. The user interface was rated well for its responsiveness and intuitive design by both the developers and the senior web designer. Suggestions for improvement focused heavily on preparing the project for clinical production. Evaluators recommended implementing comprehensive activity logging, strictly controlled data encryption, and enhanced visual feedback during the model loading phase to elevate the system beyond prototype status.

## 9.7 Limitations of Evaluation
Certain constraints encountered during the evaluation phase must be acknowledged. Primarily, the overall number of participating experts was restricted by demanding professional schedules. Because the feedback was gathered entirely through asynchronous online questionnaires, the depth of the responses was naturally bounded by the time the clinicians and engineers could spare. Furthermore, there was an expected variance in subject-matter familiarity across the panel. While all domain experts possessed strong general medical backgrounds, very few had direct, specialised experience treating Chagas disease. This limitation may have restricted highly nuanced, disease-specific clinical insights. Similarly, the software engineers reviewing the system were experts in web deployment and architecture, but not necessarily in the complexities of processing medical waveforms. Consequently, their technical feedback leant heavily towards standard software engineering practices rather than the intricacies of spatial electrocardiogram analysis. Finally, because the prototype relies strictly on pre-compiled, de-identified public datasets, the evaluators were unable to observe how the system would behave when directly interfaced with a live, continuous hospital data stream.

## 9.8 Evaluation of Functional Requirements
A comprehensive review was conducted to determine how effectively the developed system aligns with the primary functional objectives outlined in Chapter 4. The detailed breakdown regarding the implementation status of these core features is presented in Appendix H.8.

## 9.9 Evaluation of Non-Functional Requirements
The system was meticulously assessed to ensure it satisfies the strict non-functional constraints regarding operational performance, security, and usability. The complete analysis of these systemic properties and their completion rates can be found in Appendix H.9.

## 9.10 Chapter Summary
The critical evaluation confirmed that ChagaSight effectively addresses a notable healthcare challenge by providing a functional electrocardiogram screening prototype. Expert feedback validated both the underlying deep learning architecture and the practical usability of the web interface. The process also highlighted the absolute necessity of rigorous clinical trials and enhanced data security before any real-world medical deployment. Overall, the project successfully delivered a working system that exceeds standard expectations for undergraduate research while offering a clear direction for future improvements.

<div style="page-break-after: always;"></div>

# APPENDIX H : EVALUATION DATA

## H.1 Evaluation Criteria

*Table H.1: Evaluation criteria*

| Criteria | Purpose |
|---|---|
| **Concept / Novelty / Difficulty / Scope** | |
| Concept and clinical relevance | To evaluate the significance of the proposed AI-assisted screening tool and determine whether it addresses a genuine unmet medical need for Chagas disease triage. |
| Novelty of the proposed approach | To evaluate the uniqueness of combining temporal 1D signals and 2D spatial images via a dual-pathway Vision Transformer architecture. |
| Scope of the project | To assess whether the project depth and research objectives align within the feasible scope of an undergraduate degree. |
| Complexity of the methodologies | To evaluate the technical difficulty of managing extreme class imbalance and implementing cross-modal alignment on consumer hardware. |
| **Design / Architecture / Implementation** | |
| System architecture | To assess the robustness, modularity, and scalability of the decoupled Flask backend and React frontend architecture. |
| Technology stack | To determine whether the selected libraries and frameworks (e.g., PyTorch, Vite) are appropriate and optimal for the proposed medical AI solution. |
| Industry standards | To evaluate whether the system's design and data handling protocols align with standard software engineering practices. |
| **Model Implementation / Code** | |
| Implementation quality | To determine the maintainability and efficiency of the codebase alongside the effectiveness of the chosen evaluation metrics (AUROC and AUPRC). |
| Quality of model output | To evaluate the high-quality of the produced predictions, binary classifications, and risk percentages against clinical baseline expectations. |
| **Usability and Interface (GUI)** | |
| User-friendliness and intuitiveness | To evaluate the accessibility, general responsiveness, and how clearly the generated visual gauges support informed clinical reviews. |
| Clinical transparency | To verify how effectively the 12-lead waveform visualizer communicates the processed inputs to medical end users. |

## H.2 Remainder of Self-Evaluations

*Table H.2: Remainder of self-evaluations covering the finalized solution and its limitations.*

| Theme | Evaluation by the author |
|---|---|
| **Diagnostic Usability and Transparency** | A fully functional online prototype was deployed as the primary means of validating the research outcome with human users. Instead of simply generating command line diagnostics, the web interface translates the complex machine learning outputs into intuitive visual gauges, percentage scores, and plain-language clinical interpretations. The addition of the dynamic 12-lead signal viewer encourages clinical transparency by allowing medical professionals to visually verify the input data being processed by the AI. This highly responsive interface proves that the complex deep learning models running behind the scenes can be packaged into an accessible, user friendly format suitable for non-technical healthcare staff. |
| **Security Boundaries and Future Trials** | The primary limitation of this research lies in its pre-clinical nature and the lack of integrated security layers required for handling real patient data. As an academic prototype relying completely on de-identified open datasets, developing strict user authentication and data encryption was intentionally moved out of scope. Before any real world deployment could be considered, the system would require a massive security overhaul to comply with medical data protection standards alongside rigorous live-ward clinical trials. Despite these natural boundaries, the project successfully delivers a highly stable, deeply researched foundation that proves the feasibility of dual-pathway screening for Chagas disease. |

## H.3 Selection of Evaluators

*Table H.3: Selection of evaluators*

| Category | Name | Professional Role | Company / Institute / Hospital |
|---|---|---|---|
| **Domain Experts** | [Name] | Public Health Specialist | Ministry of Health |
|  | [Name] | Medical Student | University of Colombo Faculty of Medicine |
|  | [Name] | Consultant Cardiologist | Ragama Hospital |
|  | [Name] | Student | Grodno State Medical University |
|  | [Name] | Medical Officer | National Eye Hospital |
| **Technical Experts** | [Name] | Reading PhD Researcher | National Institute of Informatics, Japan |
|  | [Name] | Chief Technology Officer | Adeona Technologies (Pvt) Ltd |
|  | [Name] | Software Engineer | IFS |
|  | [Name] | Senior Software Engineer | Weblook International (Pvt) Ltd |
|  | [Name] | Senior Software Engineer | Tech Mahindra |
|  | [Name] | Senior Web Designer | Weblook International Pvt Ltd |

## H.4 Domain Experts Evaluation (Healthcare Professionals - Form A)

*Table H.4: Domain experts evaluation*

| Question | Responses |
|---|---|
| **Role and Designation** | [Paste Image Here] |
| **Hospital or Institution** | [Paste Image Here] |
| **In your clinical experience, is automated ECG-based pre-screening for cardiac complications a realistic and useful tool in hospital settings with limited access to specialist cardiology services?** | [Paste Image Here] |
| **Are you familiar with Chagas disease from your clinical training or practice? Do you agree that its long-term cardiac complications represent a significant and underdiagnosed public health concern?** | [Paste Image Here] |
| **After exploring the ChagaSight prototype, does the output format communicate results in a way that would support clinical decision-making?** | [Paste Image Here] |
| **Would a system like ChagaSight be valuable in your clinical setting for identifying patients who should be referred for confirmatory testing?** | [Paste Image Here] |
| **What concerns would you have regarding the use of an AI-generated cardiac risk score in clinical practice?** | [Paste Image Here] |
| **On a scale of 1 to 5, how clinically relevant is the problem addressed by this research?** | [Paste Image Here] |
| **From a clinical perspective, what is the most significant limitation of ChagaSight in its current form, and what single improvement would most increase your confidence in using such a system?** | [Paste Image Here] |

## H.5 Machine Learning and Vision Researcher Evaluation (Form B)

*Table H.5: Machine learning and vision researcher evaluation*

| Question | Responses |
|---|---|
| **Role and Institution** | [Paste Image Here] |
| **Does the dual-representation approach represent a meaningful and novel contribution to the field of ECG analysis or medical AI?** | [Paste Image Here] |
| **Is applying pathway-specific self-supervised pretraining objectives separately, followed by cross-modal representation alignment, a technically sound and well-justified design?** | [Paste Image Here] |
| **How does the scope of this project compare to the standard expected of a final year undergraduate software engineering degree?** | [Paste Image Here] |
| **Is AUROC the appropriate primary metric for this severely imbalanced task? Do the results appear credible and competitive?** | [Paste Image Here] |
| **Is this combination of strategies (asymmetric BCE) appropriate and well-justified for this dataset composition and prevalence level?** | [Paste Image Here] |
| **Does this design provide sufficient evidence for the value of the dual-pathway architecture and the REPA alignment module?** | [Paste Image Here] |
| **What do you consider the most significant limitation and future enhancement?** | [Paste Image Here] |
| **Please rate the technical soundness of each component on a scale of 1 to 5.** | [Paste Image Here] |

## H.6 Software Engineers / System Design Evaluation (Form C)

*Table H.6: Software engineers / system design evaluation*

| Question | Responses |
|---|---|
| **Role and Institution** | [Paste Image Here] |
| **Is the Flask backend + React frontend a sound architectural choice for a research-prototype medical AI system?** | [Paste Image Here] |
| **How would you assess the project's codebase organisation from a software engineering perspective?** | [Paste Image Here] |
| **How would you rate the overall implementation quality on a scale of 1 to 5?** | [Paste Image Here] |
| **Are deleting uploaded files immediately appropriate minimum safeguards for a research prototype handling de-identified ECG data?** | [Paste Image Here] |
| **Does this scope meet, fall short of, or exceed the standard you would expect at this academic level?** | [Paste Image Here] |
| **Is omitting user authentication a defensible engineering decision for a research prototype?** | [Paste Image Here] |
| **What architectural or implementation concern would you consider the single highest priority to address before clinical deployment?** | [Paste Image Here] |

## H.7 Senior Web Designer Evaluation (Form D)

*Table H.7: Senior web designer evaluation*

| Question | Responses |
|---|---|
| **Role and Institution** | [Paste Image Here] |
| **Overall UI and UX design rating (1 to 5)** | [Paste Image Here] |
| **Please explain your rating with specific observations about the interface** | [Paste Image Here] |
| **Does the combination of visual elements communicate the diagnostic result clearly?** | [Paste Image Here] |
| **Is the React interface responsive and functional across different screen sizes?** | [Paste Image Here] |
| **From a design perspective, are there specific components you would redesign or improve?** | [Paste Image Here] |
| **Were there any accessibility barriers you noticed?** | [Paste Image Here] |
| **What single UI or UX change would most improve the system's suitability for a clinical deployment context?** | [Paste Image Here] |

## H.8 Evaluation of Functional Requirements

*Table H.8: Evaluation of functional requirements*

| ID | Requirement | Priority | Status |
|---|---|---|---|
| **FR01** | The system shall permit users to upload 12-lead ECG recordings in WFDB format, accepting paired .hea and .dat files. | Must Have | Implemented |
| **FR02** | The system shall subject each uploaded ECG recording to a four-stage preprocessing pipeline, encompassing zero-phase filtering, resampling, per-lead Z-score normalisation, and spatial image tensor construction. | Must Have | Implemented |
| **FR03** | The system shall perform inference using a dual-pathway Vision Transformer ensemble with predictions aggregated across five cross-validation fold models. | Must Have | Implemented |
| **FR04** | The system shall present the inference result as a probability percentage, a visual gauge indicator, a binary classification label, and a plain-language clinical interpretation string. | Must Have | Implemented |
| **FR05** | The system shall enable users to select between three diagnostic model modes as Hybrid Ensemble, 2D Visual Model, and 1D Signal Model and shall display the results. | Should Have | Implemented |
| **FR06** | The system shall provide a sample ECG loader through which users may select pre-loaded representative recordings facilitating functional evaluation. | Should Have | Implemented |
| **FR07** | The system shall accept optional patient demographic inputs specifically age and biological sex and shall incorporate these values into the 1D model inference pathway. | Should Have | Implemented |
| **FR08** | The system shall generate a downloadable PDF report encapsulating the probability score, classification label, and interpretation text. | Could Have | Implemented |
| **FR09** | The system shall enforce user authentication to restrict access to authorised personnel within a clinical deployment context. | Will Not Have | Not Implemented |
| **FR10** | The system shall expose an administrative interface for the upload, labelling, and management of new ECG recordings. | Will Not Have | Not Implemented |

*Functional Requirement Completion Rate for Target Scope (FR01-FR08): 100%*

## H.9 Evaluation of Non-Functional Requirements

*Table H.9: Evaluation of non-functional requirements*

| ID | Non-Functional Requirement | Description | Priority | Status |
|---|---|---|---|---|
| **NFR01** | Accuracy | The model shall achieve AUROC ≥ 0.85 and a screening sensitivity score ≥ 0.40 on the held-out cross-validation test set. | Must Have | Implemented |
| **NFR02** | Performance | The system shall return a prediction result within 10 seconds of file upload for a standard 10-second WFDB recording. | Must Have | Implemented |
| **NFR03** | Security and Data Protection | Uploaded ECG files shall be deleted from server storage immediately upon completion of inference. | Must Have | Implemented |
| **NFR04** | Maintainability | The codebase shall be organised into discrete functional modules following separation of concerns principles with version control maintained. | Should Have | Implemented |
| **NFR05** | Usability | The frontend interface shall be responsive across standard desktop screen sizes and browsers including Chrome, Firefox, and Edge. | Should Have | Implemented |
| **NFR06** | Compliance | The system shall display a research disclaimer on all prediction result views and shall operate exclusively with de-identified datasets. | Should Have | Implemented |
| **NFR07** | Clinical Transparency | The system shall provide direct visual access to the uploaded ECG recording alongside the inference result, rendering all twelve leads as individual waveforms and displaying the derived 2D spatial ECG image. This enables clinicians to visually verify the input signal the model processed, supporting informed review of automated screening outputs without requiring knowledge of the underlying architecture. | Could Have | Implemented |

*Non-Functional Requirement Completion Rate for Target Scope: 100% (7/7 Achieved)*
