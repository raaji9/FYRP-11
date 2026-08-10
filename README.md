
# FYRP-11 - Breast Cancer Classification Using Machine Learning

## Overview

A comparative machine learning study for breast cancer classification using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The project evaluates multiple machine learning models under different data conditions, including raw, SMOTE-balanced, augmented, hyperparameter-tuned, and different train-test ratio configurations.

## Dataset

The project uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset containing:

- 569 samples
- 30 real-valued features
- Two classes: Benign and Malignant
- Features extracted from digitized fine needle aspirate (FNA) images

## Models Evaluated

| Model | Variants / Configurations |
|---|---|
| **Support Vector Machine (SVM)** | Linear, RBF, Polynomial |
| **Random Forest** | Gini, Entropy |
| **Stacking Ensemble** | Tree-based, Linear-based |
| **XGBoost + Random Forest + SVM Ensemble** | Soft Voting: Raw, Balanced, Augmented, Tuned, Ratio |

## Data Processing

Three primary dataset conditions were evaluated:

- **Raw Dataset** - 569 samples
- **Balanced Dataset** - 714 samples using SMOTE (357 per class)
- **Augmented Dataset** - 1000 samples using SMOTE (500 per class)

Additional experiments were conducted using different train-test split ratios and hyperparameter tuning.

## Evaluation Metrics

All models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Specificity
- Confusion Matrix
- ROC-AUC

## Key Result

The **XGBoost + Random Forest + SVM soft Voting Ensemble** achieved the best reported performance with **99% accuracy on the 90/10 train-test split**.

The experiments also showed that SMOTE improved recall for malignant cases, while Random Forest demonstrated consistent performance across multiple configurations.

SHAP was used for model interpretability and feature analysis.

## My Contributions

As the team leader, my contributions included:

- Literature survey and problem formulation
- Implementation of Support Vector Machine (SVM) models
- Development of the XGBoost + Random Forest + SVM Voting Ensemble
- Hyperparameter tuning experiments
- Train-test ratio experiments
- SHAP-based model interpretation and feature analysis
- Streamlit application development
- Comparative analysis of model performance
- Project documentation, report writing, and presentation

## Project Structure

```text
FYRP-11/
├── Source Code/
│   ├── datasets/
│   │   ├── data.csv
│   │   ├── balanced_data.csv
│   │   └── augmented_data.csv
│   │
│   ├── generate_balanced_data.py
│   ├── generate_augmented_data.py
│   ├── requirements.txt
│   │
│   ├── Support Vector Machine/
│   │   ├── Raw/
│   │   ├── Balanced/
│   │   ├── Augmented Dataset/
│   │   ├── Hyperparameter Tuning/
│   │   └── Ratio/
│   │
│   ├── Random Forest/
│   │   ├── Raw/
│   │   ├── Balanced/
│   │   ├── Augmented Dataset/
│   │   ├── Hyperparameter Tuning/
│   │   └── Ratio/
│   │
│   ├── Stacking Ensemble/
│   │   ├── Raw/
│   │   ├── Balanced/
│   │   ├── Augmented Dataset/
│   │   ├── Hyperparameter Tuning/
│   │   └── Ratio/
│   │
│   └── XGB_RF_SVM_Ensemble/
│       ├── xgb_rf_svm_raw.ipynb
│       ├── xgb_rf_svm_balanced.ipynb
│       ├── xgb_rf_svm_augmented.ipynb
│       ├── xgb_rf_svm_tuned.ipynb
│       └── xgb_rf_svm_ratio.ipynb
│
├── Documentation/
├── Presentation/
├── Support/
└── Help/
````

## Technology Stack

**Programming Language**

* Python

**Libraries**

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* imbalanced-learn
* Matplotlib
* Seaborn
* Streamlit

**Tools**

* Jupyter Notebook
* Visual Studio Code
* Git
* GitHub

## Metrics
All notebooks evaluate on: Accuracy, Precision, Recall, F1-score, Specificity, Confusion Matrix, ROC-AUC.

## Setup

```bash
cd "Source Code"
pip install -r requirements.txt
```

## How to Run

1. Clone or download this repository.
2. Navigate to the `Source Code` directory.
3. Install the required dependencies.
4. Open the required `.ipynb` file using Jupyter Notebook or VS Code.
5. Run the notebook cells to reproduce the corresponding experiment.

Each notebook generates model performance metrics and visualizations.

## Repository Contents

* **Source Code** — Python scripts and Jupyter notebooks for model implementation and experiments
* **Documentation** — Project report and supporting documentation
* **Presentation** — Project presentation slides
* **Support** — Setup and environment information
* **Help** — Instructions for running the project

## Dataset Source

The project uses the Wisconsin Diagnostic Breast Cancer dataset.

Dataset reference:

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Disclaimer

This project was developed as an academic research project for machine learning experimentation and is not intended for clinical diagnosis or medical decision-making.

````
## Google Drive Links

1. **Literature Review**: https://docs.google.com/spreadsheets/d/14nExcRBNWADANKsHMxUqpGTH0wt4Z0x22q4OgC5C5fU/edit?gid=0#gid=0

2. **Presentation (PPT)(Review)**: https://docs.google.com/presentation/d/1fupeUDagPaFznzABTSGKGtTMFJfSp5Jh2gKG5CqdEuo/edit?slide=id.p1#slide=id.p1

3. **Presentation (PPT)**: https://docs.google.com/presentation/d/1besQKvcQTjkuoMVoY1gp1JBOJnQBpIa8Isi9o9TDtdY/edit?slide=id.g3e9ab2575e0_0_942#slide=id.g3e9ab2575e0_0_942

4. **Output Comparison Table**: https://docs.google.com/spreadsheets/d/1eZWgGKjnu2qb5ANJTKJCBkrx1aKak6FugWd8B05Ljs4/edit?gid=584647279#gid=584647279

5. **Project Report**: https://docs.google.com/document/d/1DiWHOM3Z8IJKwRw2X_lQ0YwR-juUhsogrY8R5SY5Mhk/edit?tab=t.0

6. **Manuscript**: https://docs.google.com/document/d/1EglawXvfnSBTBOMJaKA_mBiAA-VSyicE54rMv1lY5Xk/edit?tab=t.0

7. **Dataset**: https://www.kaggle.com/datasets/ucimachinelearning/wisconsin-breast-cancer-dataset

````
