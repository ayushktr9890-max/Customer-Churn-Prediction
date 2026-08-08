# Customer Churn Prediction

> A complete, industry-level Data Science project for predicting telecom customer churn using Machine Learning.  
> Built with Python, Scikit-learn, XGBoost, and SHAP — portfolio-ready for placement and GitHub.

---

## Project Overview

Customer churn — when a customer stops using a company's service — is one of the most costly problems in the telecom industry. Acquiring a new customer costs 5–7× more than retaining an existing one. This project builds an end-to-end ML pipeline to:

- Identify customers likely to churn before they leave
- Understand *why* they are predicted to churn (explainability)
- Suggest personalised retention actions

**Target variable:** `Churn` (Yes / No)  
**Best model metric (ROC-AUC):** chosen automatically from Logistic Regression, Decision Tree, Random Forest, XGBoost

---

## Dataset Information

| Attribute | Details |
|-----------|---------|
| Name | IBM Telco Customer Churn |
| Source | Local `customer_churn.csv` file |
| Rows | 7,043 |
| Columns | 21 |
| Target | `Churn` (Yes=~27%, No=~73%) |
| Key Features | tenure, Contract, MonthlyCharges, InternetService, PaymentMethod |

Place your dataset in the project root:
```
customer_churn.csv
```

---

## Installation

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

The project uses the local CSV already included in the repository:
```
customer_churn.csv
```

---

## How to Run

### Option A — Jupyter Notebooks (recommended for exploration)

```bash
jupyter notebook
```

Open notebooks in order:

| Notebook | Purpose |
|----------|---------|
| `01_data_understanding.ipynb` | Load data, inspect shape, types, missing values |
| `02_data_cleaning.ipynb` | Fix data quality issues |
| `03_eda.ipynb` | Exploratory Data Analysis + plots |
| `04_model_training.ipynb` | Train & compare models, save best model |
| `05_model_evaluation.ipynb` | Deep evaluation: confusion matrix, ROC, SHAP |
| `06_prediction_demo.ipynb` | Predict churn for sample / custom customers |

### Option B — Python scripts

**Train the model:**
```bash
python src/train.py
```

**Interactive terminal prediction:**
```bash
python src/predict.py
```

The prediction script will prompt you for customer details and output:
- Predicted Churn: Yes / No
- Probability of Churn: e.g. 84%
- Top features influencing the prediction
- Suggested retention actions

Enter monthly and total charges in Indian rupees (₹). The prediction script
converts these values to the scale used by the trained model before predicting.

---

## Machine Learning Pipeline

```
Raw CSV
   │
   ▼
[1] Data Understanding     ← shape, dtypes, nulls, duplicates
   │
   ▼
[2] Data Cleaning          ← drop customerID, fix TotalCharges, handle NaN
   │
   ▼
[3] Feature Engineering    ← tenure_group, monthly_per_tenure
   │
   ▼
[4] Encoding               ← Label Encoding (binary), One-Hot Encoding (multi-class)
   │
   ▼
[5] Train/Test Split       ← 80% train / 20% test (stratified, random_state=42)
   │
   ▼
[6] Scaling                ← StandardScaler (fit on train, transform both)
   │
   ▼
[7] Model Training         ← Logistic Regression, Decision Tree, Random Forest, XGBoost
   │
   ▼
[8] Model Selection        ← Best ROC-AUC wins
   │
   ▼
[9] Evaluation             ← Confusion Matrix, Classification Report, ROC Curve, SHAP
   │
   ▼
[10] Persistence           ← model/churn_model.pkl (model + scaler + feature names)
```

### Live Prediction Workflow

```
Customer details
   │
   ▼
[1] Validate input         ← required fields, valid options, ₹ charge ranges
   │
   ▼
[2] Convert charges        ← ₹ values converted to the model's training scale
   │
   ▼
[3] Preprocess             ← engineer features, encode categories, align columns, scale
   │
   ▼
[4] Predict churn          ← Random Forest returns class and probability
   │
   ▼
[5] Explain result         ← top churn drivers and risk level
   │
   ▼
[6] Recommend action       ← tailored retention suggestions for the customer
```

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | ~0.79 | ~0.64 | ~0.55 | ~0.59 | ~0.84 |
| Decision Tree | ~0.76 | ~0.56 | ~0.59 | ~0.57 | ~0.75 |
| Random Forest | ~0.80 | ~0.66 | ~0.56 | ~0.60 | ~0.85 |
| **XGBoost** | **~0.81** | **~0.67** | **~0.60** | **~0.63** | **~0.86** |

> Exact numbers will vary slightly. XGBoost or Random Forest typically wins on ROC-AUC.

**Top Churn Predictors:**
1. Contract type (month-to-month)
2. Tenure (short tenure = high risk)
3. Internet Service (Fiber optic)
4. Monthly Charges (higher = higher risk)
5. Payment Method (electronic check)

---

## Folder Structure

```
Customer-Churn-Prediction/
├── customer_churn.csv              ← local raw dataset
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_prediction_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            ← cleaning, encoding, splitting, scaling
│   ├── train.py                    ← full training pipeline (run as script)
│   ├── predict.py                  ← interactive terminal prediction
│   └── utils.py                    ← shared helpers (load, save, plot, metrics)
├── model/
│   └── churn_model.pkl             ← saved best model bundle
├── outputs/
│   ├── plots/                      ← all generated charts (PNG)
│   └── reports/                    ← metrics CSV, classification report TXT
├── requirements.txt
└── README.md
```

---

## Future Improvements

- **Hyperparameter Tuning** — GridSearchCV or Optuna for systematic optimisation
- **Cross-Validation** — k-fold CV for more robust model selection
- **Imbalanced Learning** — SMOTE or class-weight tuning (imbalanced-learn)
- **Threshold Optimisation** — tune decision threshold for business-specific recall/precision trade-off
- **Time-Series Features** — incorporate call history, usage trends over time
- **Ensemble Stacking** — combine predictions from multiple models
- **Batch Prediction** — predict churn for a CSV of new customers
- **Monitoring** — detect feature/data drift in production

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Core language |
| pandas | 2.0+ | Data manipulation |
| numpy | 1.24+ | Numerical computing |
| matplotlib | 3.7+ | Visualisation |
| seaborn | 0.12+ | Statistical plots |
| scikit-learn | 1.3+ | ML models, preprocessing, metrics |
| xgboost | 2.0+ | Gradient boosting model |
| shap | 0.43+ | Model explainability |
| jupyter | 7.0+ | Notebook environment |

---

## Author

**Ayush Katiyar**  
B.Tech — Placement Portfolio Project  
[GitHub](https://github.com/your-username)

---

*This project follows PEP-8 coding standards, uses modular and reusable functions, includes exception handling, and is structured for reproducibility.*
