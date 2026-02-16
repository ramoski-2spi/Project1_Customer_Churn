import pandas as pd
import joblib
import numpy as np
from src.preprocess import transform_data_predictions

# Load saved objects
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Threshold choosen in the hyperparameter tuning notebook
THRESHOLD = 0.35  

# Define feature order 
FEATURE_COLUMNS = model.feature_names_in_

# New customer data example
new_customer = {"SeniorCitizen": 0,
                "MonthlyCharges": 150.8,
                "TotalCharges": 240.6,
                "gender_Male": 1,
                "Partner_Yes": 0,
                "Dependents_Yes": 0,
                "PhoneService_Yes": 1,
                "MultipleLines_No phone service": 0,
                "MultipleLines_Yes": 1,
                "InternetService_Fiber optic": 0,
                "InternetService_No": 0,
                "OnlineSecurity_No internet service": 0,
                "OnlineSecurity_Yes": 1,
                "OnlineBackup_No internet service": 0,
                "OnlineBackup_Yes": 1,
                "DeviceProtection_No internet service": 0,
                "DeviceProtection_Yes": 0,
                "TechSupport_No internet service": 0,
                "TechSupport_Yes": 0,
                "StreamingTV_No internet service": 0,
                "StreamingTV_Yes": 0,
                "StreamingMovies_No internet service": 0,
                "StreamingMovies_Yes": 0,
                "Contract_One year": 0,
                "Contract_Two year": 1,
                "PaperlessBilling_Yes": 1,
                "PaymentMethod_Credit card (automatic)": 1,
                "PaymentMethod_Electronic check": 0,
                "PaymentMethod_Mailed check": 0,
                "tenure_groups_0-1 year": 0,
                "tenure_groups_1-2 years": 0,
                "tenure_groups_2-3 years": 0,
                "tenure_groups_3+ years": 1}

# Create DataFrame (safe order)
X_new = pd.DataFrame([new_customer])[FEATURE_COLUMNS]

# Scale features
cols_to_scale = ['MonthlyCharges', 'TotalCharges']
X_new_scaled = X_new.copy()
X_new_scaled[cols_to_scale] = scaler.transform(X_new[cols_to_scale])

# Predict probability
churn_probability = model.predict_proba(X_new_scaled)[0][1]

# Apply threshold manually
churn_prediction =  int(churn_probability >= THRESHOLD)

# Output
print("Churn Prediction:", "YES" if churn_prediction == 1 else "NO")
print("Churn Probability:", f"{churn_probability:.2%}")