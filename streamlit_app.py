
import streamlit as st, joblib, pandas as pd, shap, matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Explainable AI in Education")
st.title("🧬 Explainable AI in Education")

art = Path("artifacts")
if not (art/"model.pkl").exists():
    st.warning("Run `python -m src.train` first.")
else:
    model = joblib.load(art/"model.pkl")
    X = pd.read_csv(art/"X_test.csv")
    st.write("Sample features:", X.head())

    st.subheader("Feature Importance (SHAP)")
    explainer = shap.Explainer(model, X.drop(columns=["group"], errors="ignore"))
    shap_values = explainer(X.drop(columns=["group"], errors="ignore")).values
    plt.figure()
    shap.summary_plot(shap_values, X.drop(columns=["group"], errors="ignore"), show=False)
    st.pyplot(plt.gcf())

    st.subheader("Fairness Probe (by group)")
    if "group" in X.columns:
        X0 = X[X["group"]==0].drop(columns=["group"])
        X1 = X[X["group"]==1].drop(columns=["group"])
        p0 = model.predict_proba(X0)[:,1].mean() if len(X0) else 0
        p1 = model.predict_proba(X1)[:,1].mean() if len(X1) else 0
        st.write({"group0_avg_proba": float(p0), "group1_avg_proba": float(p1), "ratio": float((p1+1e-9)/(p0+1e-9))})
