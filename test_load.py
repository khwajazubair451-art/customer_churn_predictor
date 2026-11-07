import streamlit as st
import joblib
from pathlib import Path
import sklearn

st.title("Model Loading Test")

st.write("Scikit-learn version:", sklearn.__version__)

model_path = Path(__file__).resolve().parent / "churn_model.pkl"

st.write("Model path:", str(model_path))

model = joblib.load(model_path)

st.success("MODEL LOADED SUCCESSFULLY!")