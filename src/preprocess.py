import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file: str):
    return pd.read_csv(file)

#For future use if needed
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'models/scaler.pkl')
    return X_train_scaled, X_test_scaled