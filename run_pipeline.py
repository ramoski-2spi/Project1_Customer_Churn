from src.preprocess import load_data
from src.train_model import train_logistic, train_rf, train_gb
from src.evaluate import model_evaluation

X_train = load_data('data/X_train.csv')
X_test = load_data('data/X_test.csv')
y_train = load_data('data/y_train.csv').squeeze()
y_test = load_data('data/y_test.csv').squeeze()

#the data in the csv is already scaled so no need to use standardization function
log_model = train_logistic(X_train, y_train)
rf_model = train_rf(X_train, y_train)
gb_model = train_gb(X_train, y_train)

print('Logistic Results:')
model_evaluation(log_model, X_test, y_test)

print('\nRandom Forest Results:')
model_evaluation(rf_model, X_test, y_test)

print('\nGradient Boosting Results:')
model_evaluation(gb_model, X_test, y_test)