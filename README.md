# FYRP-11 — Breast Cancer Classification with ML Models

## Overview
Comparing model performance on the Wisconsin Breast Cancer Dataset using raw (imbalanced) vs balanced (SMOTE) vs augmented (500/class) data across 4 models with multiple kernels/variants.

## Models
| Model | Variants |
|-------|----------|
| **SVM** | linear, rbf, poly |
| **Random Forest** | gini, entropy |
| **Stacking Ensemble** | tree-based, linear-based |
| **XGB+RF+SVM Ensemble** | Voting Classifier (soft voting) — Raw, Balanced, Augmented, Tuned, Ratio |

## Setup
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost shap streamlit
```

## Folder Structure
```
FYRP-11/
├── data.csv                          # Original dataset (569 samples, imbalanced)
├── balanced_data.csv                 # SMOTE-balanced (714 samples, 357/class)
├── augmented_data.csv                # SMOTE-augmented (1000 samples, 500/class)
│
├── Support Vector Machine/
│   ├── Raw/           svm_linear, svm_rbf, svm_poly
│   ├── Balanced/      svm_linear, svm_rbf, svm_poly
│   ├── Augmented Dataset/  svm_linear, svm_rbf, svm_poly
│   ├── Hyperparameter Tuning/  svm_linear, svm_rbf, svm_poly
│   └── Ratio/         svm_linear, svm_rbf, svm_poly
│
├── Random Forest/
│   ├── Raw/           rf_gini, rf_entropy
│   ├── Balanced/      rf_gini, rf_entropy
│   ├── Augmented Dataset/  rf_gini, rf_entropy
│   ├── Hyperparameter Tuning/  rf_gini, rf_entropy
│   └── Ratio/         rf_gini, rf_entropy
│
├── Stacking Ensemble/
│   ├── Raw/           se_tree_based, se_linear_based
│   ├── Balanced/      se_tree_based, se_linear_based
│   ├── Augmented Dataset/  se_tree_based, se_linear_based
│   ├── Hyperparameter Tuning/  se_tree_based, se_linear_based
│   └── Ratio/         se_tree_based, se_linear_based
│
├── XGB_RF_SVM_Ensemble/     ← New Ensemble Model
│   ├── xgb_rf_svm_raw.ipynb
│   ├── xgb_rf_svm_balanced.ipynb
│   ├── xgb_rf_svm_augmented.ipynb
│   ├── xgb_rf_svm_tuned.ipynb
│   └── xgb_rf_svm_ratio.ipynb
│
├── app.py                      # Streamlit UI for real-time prediction
├── requirements.txt            # Python dependencies
├── report/                     # Report & manuscript templates
│
├── generate_balanced_data.py   # Script to create balanced_data.csv
├── generate_augmented_data.py  # Script to create augmented_data.csv
└── check_notebooks.py          # Syntax checker for notebooks
```

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Open any `.ipynb` file in Jupyter Notebook / VS Code
3. Run all cells — each notebook outputs Accuracy, Precision, Recall, F1-score, Specificity, Confusion Matrix, ROC curve, and blue-themed visualizations

### Run the Web App
```bash
streamlit run app.py
```

## Metrics
All notebooks evaluate on: Accuracy, Precision, Recall, F1-score, Specificity, Confusion Matrix, ROC-AUC.

## Google Drive Links
<!-- Add your Google Drive links below -->

1. **Literature Review**: https://docs.google.com/spreadsheets/d/14nExcRBNWADANKsHMxUqpGTH0wt4Z0x22q4OgC5C5fU/edit?gid=0#gid=0

2. **Presentation (PPT)**(Review): https://docs.google.com/presentation/d/1fupeUDagPaFznzABTSGKGtTMFJfSp5Jh2gKG5CqdEuo/edit?slide=id.p1#slide=id.p1

3. **Presentation (PPT)**:https://docs.google.com/presentation/d/1besQKvcQTjkuoMVoY1gp1JBOJnQBpIa8Isi9o9TDtdY/edit?slide=id.g3e9ab2575e0_0_942#slide=id.g3e9ab2575e0_0_942

. **Output Comparision Table**: https://docs.google.com/spreadsheets/d/1eZWgGKjnu2qb5ANJTKJCBkrx1aKak6FugWd8B05Ljs4/edit?gid=584647279#gid=584647279

. **Project Report**: https://docs.google.com/document/d/1DiWHOM3Z8IJKwRw2X_lQ0YwR-juUhsogrY8R5SY5Mhk/edit?tab=t.0

. **Manuscript**: https://docs.google.com/document/d/1EglawXvfnSBTBOMJaKA_mBiAA-VSyicE54rMv1lY5Xk/edit?tab=t.0

. **Dataset**: https://www.kaggle.com/datasets/ucimachinelearning/wisconsin-breast-cancer-dataset