from src.preprocess import load_data, standardize_data
from src.train_model import train_logistic, train_rf, train_gb
from src.evaluate import model_evaluation

import pandas as pd

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test = pd.read_csv('data/y_test.csv').squeeze()

X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

log_model = train_logistic(X_train_scaled, y_train)
rf_model = train_rf(X_train_scaled, y_train)
gb_model = train_gb(X_train_scaled, y_train)

print('Logistic Results:')
model_evaluation(log_model, X_test_scaled, y_test)

print('\nRandom Forest Results:')
model_evaluation(rf_model, X_test_scaled, y_test)

print('\nGradient Boosting Results:')
model_evaluation(gb_model, X_test_scaled, y_test)