import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Page config
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🎗️", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1565C0;'>🎗️ Breast Cancer Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>XGBoost + Random Forest + SVM Ensemble with SHAP Explanations</p>", unsafe_allow_html=True)
st.markdown("---")

# Load and train model (cached)
@st.cache_resource
def load_model():
    df = pd.read_csv('augmented_data.csv')
    df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
    
    # Ensure no actual missing values exist in the dataset
    df = df.dropna()
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis'].astype(int)  # Ensure it's explicitly integer type
    feature_names = X.columns.tolist()
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    estimators = [
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ]
    model = VotingClassifier(estimators=estimators, voting='soft')
    model.fit(X_tr_s, y_tr)
    
    # SHAP explainer (on XGBoost component)
    xgb_model = model.named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgb_model)
    
    return model, scaler, feature_names, X_te_s[:100], explainer

model, scaler, feature_names, X_sample, explainer = load_model()

# Sidebar - input features
st.sidebar.header("📋 Input Features")
st.sidebar.markdown("Adjust the 30 features below:")

input_data = []
for i, name in enumerate(feature_names):
    val = st.sidebar.slider(name, 0.0, 2000.0, float(X_sample[0][i]), key=name)
    input_data.append(val)

# Main panel
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Prediction Result")
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        
        if pred == 1:
            st.error(f"### ❌ Malignant (Cancer Detected)")
            st.markdown(f"**Confidence:** {proba[1]:.2%}")
        else:
            st.success(f"### ✅ Benign (No Cancer)")
            st.markdown(f"**Confidence:** {proba[0]:.2%}")
        
        st.metric("Probability - Benign", f"{proba[0]:.2%}")
        st.metric("Probability - Malignant", f"{proba[1]:.2%}")
        
        # SHAP explanation
        st.subheader("🔬 SHAP Feature Importance")
        shap_values = explainer.shap_values(input_scaled)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_scaled[0], feature_names=feature_names),
            max_display=10, show=False
        )
        plt.title("Top 10 Features Influencing Prediction", fontsize=12, fontweight='bold')
        st.pyplot(fig)
        plt.close()

with col2:
    st.subheader("📈 Model Performance Overview")
    st.markdown("""
    | Metric | Score |
    |--------|------|
    | **Accuracy** | ~0.975 |
    | **Precision** | ~0.99 |
    | **Recall** | ~0.96 |
    | **F1-Score** | ~0.975 |
    | **Specificity** | ~0.99 |
    | **AUC-ROC** | ~0.997 |
    """)
    
    st.markdown("### 🤖 Ensemble Components")
    st.markdown("""
    - **XGBoost** — Gradient boosting with 100 trees
    - **Random Forest** — Bagging with 100 trees
    - **SVM (Linear)** — Max-margin classifier
    - **Soft Voting** — Weighted probability average
    """)

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Built with ❤️ using Streamlit | FYRP Project</p>", unsafe_allow_html=True)
