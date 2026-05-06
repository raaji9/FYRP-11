# FYRP-11 — Breast Cancer Classification with ML Models

## Overview
Comparing model performance on the Wisconsin Breast Cancer Dataset using raw (imbalanced) vs balanced (SMOTE) data across 3 models with multiple kernels/variants.

## Models
| Model | Variants |
|-------|----------|
| **SVM** | linear, rbf, poly |
| **Random Forest** | gini, entropy |
| **Stacking Ensemble** | tree-based, linear-based |

## Setup
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## Folder Structure
```
data.csv                          # Dataset (shared across all notebooks)
Support Vector Machine/
├── Raw/       svm_linear.ipynb, svm_rbf.ipynb, svm_poly.ipynb
└── Balanced/  (coming soon)
Random Forest/
├── Raw/       rf_gini.ipynb, rf_entropy.ipynb
└── Balanced/  (coming soon)
Stacking Ensemble/
├── Raw/       se_tree_based.ipynb, se_linear_based.ipynb
└── Balanced/  (coming soon)
```

## How to Run
1. Run `pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn`
2. Open any `.ipynb` file in Jupyter Notebook / VS Code
3. Run all cells — each notebook outputs Accuracy, Precision, Recall, F1-score, Specificity, Confusion Matrix, and ROC curve

## Metrics
All notebooks evaluate on: Accuracy, Precision, Recall, F1-score, Specificity, Confusion Matrix, ROC-AUC.