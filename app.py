
import streamlit as st
import pandas as pd
import joblib

# Load trained model
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction System")

st.write(
    "Predict customer churn risk and discover "
    "ways to improve customer retention."
)

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender", ["Male", "Female"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen", [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner", ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents", ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure (Months)", 0, 72, 12
)

phone = st.sidebar.selectbox(
    "Phone Service", ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing", ["Yes", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=float(monthly * tenure)
)

if st.button("Predict Churn"):

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Results")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= 0.60:
        st.error("High Churn Risk")
        st.write(
            "Recommended Action: Contact this customer "
            "and offer a suitable retention package."
        )

    elif probability >= 0.30:
        st.warning("Medium Churn Risk")
        st.write(
            "Recommended Action: Offer loyalty benefits "
            "and improve customer engagement."
        )

    else:
        st.success("Low Churn Risk")
        st.write(
            "Recommended Action: Continue providing "
            "good customer service."
        )
