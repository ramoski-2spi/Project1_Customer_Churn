# Customer Churn Prediction – Machine Learning Project

## Project Overview
This project builds an end-to-end machine learning pipeline to predict customer churn for a subscription-based business.  
The focus is on **business-driven modeling**, **automation**, and **model interpretability**, rather than chasing unrealistic accuracy scores.

The project follows a real-world data science workflow:
- Data exploration and cleaning
- Feature engineering
- Model training and evaluation
- Hyperparameter and threshold tuning
- Automated training and prediction scripts

---

## Business Problem
Customer churn is costly. Retaining an existing customer is significantly cheaper than acquiring a new one.

**Objective:**  
Identify customers likely to churn so the business can take preventive actions.

**Key Modeling Priority:**  
- **Recall for churned customers**, because missing a churner is more costly than a false positive.

---

## Dataset
- Telecom customer data
- Mix of numerical and categorical features
- Target variable: `Churn` (Yes / No)

### Data Quality Notes
- No missing values were initially detected
- After converting `TotalCharges` to numeric, missing values appeared due to invalid string entries
- These rows were handled during data cleaning

---

## Exploratory Data Analysis (EDA) – Key Insights
- Strong class imbalance (more non-churners than churners)
- Customers on **month-to-month contracts churn significantly more**
- Customers with **short tenure** are more likely to churn
- Churned customers tend to have **higher monthly charges**

These insights guided model choice and evaluation strategy.

---

## Feature Engineering
- One-hot encoding for categorical variables
- Numerical feature scaling using `StandardScaler`
- Tenure grouped into buckets to capture lifecycle effects

> Feature selection and correlation removal were intentionally excluded in the final pipeline, as:
> - Tree-based models are robust to correlated features
> - Logistic Regression performance indicated multicollinearity was not harmful

---

## Models Trained
The following models were trained and compared:

- Logistic Regression
- Random Forest
- Gradient Boosting

Despite its simplicity, **Logistic Regression performed best** on Recall and F1-score, which aligns with the business objective.

---

## Hyperparameter & Threshold Tuning
- `GridSearchCV` was used to tune Logistic Regression
- Optimization focused on **Recall** and **F1-score**
- Decision threshold tuning was applied to improve churn detection

---

## Final Model Performance
**Selected Model:** Tuned Logistic Regression  
**Decision Threshold:** Optimized for Recall

| Metric | Score |
|------|------|
| Recall (Churn) | ~0.93 |
| Precision (Churn) | ~0.41 |
| F1-Score | ~0.58 |
| ROC-AUC | ~0.84 |

### Interpretation
- High recall ensures most churners are detected
- Lower precision is acceptable given the business context
- Performance is realistic and comparable to industry churn models

---

## Automation
This project includes automation scripts to ensure reproducibility:

- Model training pipeline
- Model evaluation
- Model persistence (scaler and model saving)
- Prediction script for new customer data

Run the full pipeline:
```bash
python run_pipeline.py
