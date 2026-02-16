from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

def train_logistic(X_train, y_train):
    model = LogisticRegression(class_weight= 'balanced', max_iter= 5000)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'models/Logistic_model.pkl')
    return model

def train_rf(X_train, y_train):
    model = RandomForestClassifier(class_weight= 'balanced', random_state= 42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'models/RandomForest_model.pkl')
    return model

def train_gb(X_train, y_train):
    model = GradientBoostingClassifier(random_state= 42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'models/GradientBoosting_model.pkl')
    return model