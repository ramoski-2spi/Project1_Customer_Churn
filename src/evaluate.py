from sklearn.metrics import f1_score, recall_score

def model_evaluation(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(f'Recall Score: {recall_score(y_test, predictions)}')
    print(f'F1 Score: {f1_score(y_test, predictions)}')