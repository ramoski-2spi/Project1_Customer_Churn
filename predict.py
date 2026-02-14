import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load saved objects
# -----------------------------
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/logistic_model.pkl")

THRESHOLD = 0.50  # your chosen threshold

# -----------------------------
# Define feature order (CRITICAL)
# -----------------------------
FEATURE_COLUMNS = scaler.feature_names_in_

# -----------------------------
# New customer data
# -----------------------------
new_customer = {
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 845.6,
    "gender_Male": 1,
    "Partner_Yes": 1,
    "Dependents_Yes": 0,
    "PhoneService_Yes": 1,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 1,
    "InternetService_Fiber optic": 1,
    "InternetService_No": 0,
    "OnlineSecurity_No internet service": 0,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_No internet service": 0,
    "OnlineBackup_Yes": 1,
    "DeviceProtection_No internet service": 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_No internet service": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No internet service": 0,
    "StreamingTV_Yes": 1,
    "StreamingMovies_No internet service": 0,
    "StreamingMovies_Yes": 1,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaperlessBilling_Yes": 1,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0,
    "tenure_groups_0-1 year": 0,
    "tenure_groups_1-2 years": 1,
    "tenure_groups_2-3 years": 0,
    "tenure_groups_3+ years": 0
}

# -----------------------------
# Create DataFrame (safe order)
# -----------------------------
X_new = pd.DataFrame([new_customer])[FEATURE_COLUMNS]

# -----------------------------
# Scale features
# -----------------------------
X_new_scaled = scaler.transform(X_new)

# -----------------------------
# Predict probability
# -----------------------------
churn_probability = model.predict_proba(X_new_scaled)[0, 1]

# Apply threshold manually
churn_prediction = int(churn_probability >= THRESHOLD)

# -----------------------------
# Output
# -----------------------------
print("Churn Prediction:", "YES" if churn_prediction == 1 else "NO")
print("Churn Probability:", f"{churn_probability:.2%}")

