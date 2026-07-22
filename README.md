# Customer Churn Prediction — End-to-End ML Project

**Developer:** Jagadishkumar J | Final Year B.Tech CSBS, NIET

An end-to-end machine learning pipeline that predicts whether a subscription-based
customer will churn (leave) or stay, based on their demographics, account details,
and service usage.

---

## Abstract

Customer churn — the loss of existing customers to competitors or cancellation —
is one of the most expensive problems for subscription-based businesses, since
acquiring a new customer typically costs far more than retaining an existing one.
This project builds a binary classification model that predicts customer churn
using demographic information (gender, senior citizen status, partner/dependents),
account information (tenure, contract type, payment method), service usage
(internet, phone, online security, streaming, etc.), and billing data (monthly
and total charges).

The workflow covers the complete ML lifecycle: data collection, exploratory data
analysis, data cleaning, feature engineering, encoding, scaling, model training
(Logistic Regression and Random Forest), evaluation, hyperparameter tuning with
`GridSearchCV`, model persistence, and a deployment-ready prediction function.
The tuned Random Forest model is saved for reuse in a Streamlit/Flask application,
enabling businesses to proactively identify at-risk customers and target them
with retention offers.

---

## Problem Statement

- **Business Problem:** A subscription-based (telecom-style) company is losing
  customers to churn, and retaining an existing customer is significantly cheaper
  than acquiring a new one.
- **Goal:** Predict whether a customer will churn (`Yes`/`No`) using their profile
  and usage data, so the business can proactively offer retention incentives.
- **Type:** Binary Classification (`Churn`: Yes / No)

---

## Dataset

- **File:** `customer_churn_prediction_dataset.csv`
- **Size:** 300 customer records, 21 columns
- **Feature groups:**
  - Demographics — `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Account info — `tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`
  - Services — `PhoneService`, `InternetService`, `OnlineSecurity`, etc.
  - Billing — `MonthlyCharges`, `TotalCharges`
  - Target — `Churn`

---

## Project Workflow

1. **Collect Data** — load the raw CSV (with Google Colab upload support)
2. **Import Libraries** — NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, joblib
3. **Load Dataset**
4. **Data Overview** — shape, dtypes, summary statistics
5. **Data Quality Check** — missing values, duplicates, class balance
6. **Exploratory Data Analysis (EDA)** — churn distribution, numerical vs. churn
   boxplots, categorical vs. churn count plots, correlation heatmap
7. **Data Cleaning** — drop non-predictive identifiers (`customerID`)
8. **Feature Engineering** — `TenureGroup` buckets, `AvgChargePerMonth`
9. **Feature Selection** — split features (`X`) and target (`y`)
10. **Encoding** — one-hot encoding of categorical variables
11. **Feature Scaling** — `StandardScaler` on numerical columns
12. **Train/Test Split** — 80/20 stratified split
13. **Model Selection** — Logistic Regression (baseline) vs. Random Forest
14. **Model Training** — both models trained with `class_weight='balanced'`
15. **Prediction**
16. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices,
    ROC curves, feature importance
17. **Hyperparameter Tuning** — `GridSearchCV` over Random Forest parameters
18. **Model Saving** — model, scaler, and column list persisted with `joblib`
19. **Deployment** — a `predict_churn()` function ready to wrap in a Streamlit/Flask app

---

## Tech Stack

| Category            | Tools/Libraries                                  |
|----------------------|--------------------------------------------------|
| Language             | Python 3                                          |
| Data Handling         | Pandas, NumPy                                    |
| Visualization         | Matplotlib, Seaborn                              |
| Machine Learning      | scikit-learn (Logistic Regression, Random Forest) |
| Model Tuning          | GridSearchCV                                     |
| Model Persistence     | joblib                                           |
| Environment           | Jupyter Notebook / Google Colab                  |

---

## Models & Evaluation

Two models are trained and compared:

- **Logistic Regression** — simple, interpretable baseline
- **Random Forest** — captures non-linear relationships, generally higher accuracy

Both are evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC, with
confusion matrices and ROC curve comparisons. The Random Forest is further tuned
via `GridSearchCV` (grid over `n_estimators`, `max_depth`, `min_samples_split`)
and the best estimator is used as the final model.

---

## Repository Structure

```
├── AIML_PROJECT_1_.ipynb              # Main notebook (full pipeline)
├── customer_churn_prediction_dataset.csv   # Dataset
├── churn_model.pkl                    # Saved tuned Random Forest model
├── scaler.pkl                         # Saved StandardScaler
├── model_columns.pkl                  # Saved feature column order
└── README.md                          # Project documentation
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook AIML_PROJECT_1_.ipynb
   ```
   *(If using Google Colab, upload the dataset CSV when prompted in the notebook.)*

---

## Future Work

- Wrap `predict_churn()` in a **Streamlit** or **Flask** app for a live, shareable demo
- Experiment with additional models (XGBoost, Gradient Boosting)
- Deploy the model as a REST API

---

## Author

**Jagadishkumar J**
Final Year B.Tech Computer Science and Business Systems (CSBS), NIET 
