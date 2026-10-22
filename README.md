# ECG-Based Heart Attack Detection Using ResNet1D

A deep learning project for developing a heart attack detection system using 12-lead electrocardiogram (ECG) signals and a one-dimensional Residual Neural Network (ResNet1D).

## Project Overview

This project explores the use of deep learning to classify ECG recordings according to a target label related to heart attack detection. The workflow includes ECG data preparation, signal quality assessment, preprocessing, model development, and performance evaluation.

The project uses ECG data associated with the MIMIC-IV ecosystem. The exact prediction target, cohort definition, and evaluation protocol will be documented as the pipeline develops.

**Current status:** Data and model pipeline development. Dataset-wide quality control, model training, and performance evaluation remain to be completed.

## Objectives

- Prepare and preprocess 12-lead ECG signals for deep learning.
- Assess signal quality before model training.
- Develop a ResNet1D architecture for ECG classification.
- Evaluate model performance using appropriate classification metrics.
- Investigate model errors and potential data leakage.

## Methodology

### 1. ECG Data Preparation

ECG recordings are loaded and prepared for subsequent processing. Dataset metadata and target labels are managed separately from the raw signal files.

### 2. Signal Preprocessing

The preprocessing pipeline includes signal preparation and resampling. The current implementation has been tested on ECG signals with a sampling frequency of 500 Hz and converts them to 100 Hz, producing 10-second, 12-lead input segments.

### 3. Signal Quality Control

A preliminary signal quality control module checks basic signal properties, including:

- Signal dimensions and lead count.
- Minimum number of samples.
- Non-finite values.
- Flat or constant leads.
- Extreme signal amplitudes.

A pilot test has been conducted on 20 ECG records. The results are preliminary and do not yet represent quality control of the entire dataset.

### 4. Model Architecture

The project includes a one-dimensional Residual Neural Network (ResNet1D) designed to accept multilead ECG input.

The model's forward pass has been tested. Model training and evaluation are still pending.

### 5. Model Evaluation

Planned evaluation metrics include:

- Sensitivity (Recall)
- Specificity
- Precision
- F1-score
- ROC-AUC
- Confusion matrix

The final metrics will be reported after model training and evaluation are completed.

## Repository Structure

```text
deteksi-serangan-jantung/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── predictions/
├── src/
│   ├── dataset/
│   │   ├── build_master.py
│   │   ├── check_master.py
│   │   ├── cohort.py
│   │   ├── ecg_loader.py
│   │   ├── features_emr.py
│   │   ├── features_mm.py
│   │   ├── labels.py
│   │   └── run_signal_qc_pilot.py
│   └── model/
│       ├── ecg_preprocessing.py
│       ├── ecg_quality.py
│       └── resnet1d.py
└── tests/
```

## Technology Stack

- Python
- PyTorch
- WFDB
- NumPy
- SciPy
- pandas
- pytest

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/melinda-ab/deteksi-serangan-jantung.git
cd deteksi-serangan-jantung
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows Git Bash:

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

Install the project's required packages in the virtual environment. A reproducible dependency file will be documented as the environment configuration is finalized.

### 4. Run tests

If the required dependencies are installed, run:

```bash
python -m pytest
```

## Data Privacy

Raw ECG recordings, patient-related metadata, and derived datasets are not included in this public repository. Access to and use of the underlying dataset must follow the applicable data access agreements, licensing terms, and privacy requirements.

## Limitations

This project is under development. A functioning model architecture does not establish clinical validity or diagnostic performance. The model must undergo training, appropriate validation, and independent evaluation before its suitability for any real-world medical application can be assessed.

## License

A license has not yet been specified. Terms of use will be added when the project license is determined.
