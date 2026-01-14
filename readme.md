# 💳 Fraud Detection Machine Learning Pipeline

This repository implements a full **end-to-end fraud detection system** using supervised machine learning techniques — from raw data ingestion and preprocessing to model training, evaluation, and prediction.

The pipeline is designed with **industry best practices** including:
✅ modular code  
✅ logging  
✅ exception handling  
✅ imbalance handling with SMOTE  
✅ metric-based model comparison and selection  

---

## 📁 Project Folder Structure

```
fraud_detection/
│
├── data/
│   ├── raw/
│   │   └── data_table.csv
│   └── processed/
│       └── processed_data.csv
│
├── models/
│   ├── logistic_model.pkl
│   ├── tree_model.pkl
│   ├── xgb_model.pkl
│   └── scaler.pkl
│
├── reports/
│   └── model_comparison.csv
│
├── logs/
│   └── fraud_detection_YYYY-MM-DD.log
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_models.py
│   ├── predict.py
│   ├── logger.py
│   ├── exceptions.py
│   └── main.py
│
├── notebooks/
│   └── eda.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🧠 Dataset

Due to file size limitations on GitHub, the full dataset is **hosted on Kaggle**:

🔗 https://www.kaggle.com/datasets/tatapudibhaskar/paysim-synthetic-financial-transactions-dataset

This dataset contains synthetic transaction records with a binary `isFraud` label indicating fraudulent behavior.

---

## 🐍 Virtual Environment Setup

### ✅ Using Python (Recommended)

Make sure you are using **Python 3.10.\***.

```bash
python3 -m venv projectenv
```

You may replace `projectenv` with any name.

---

### ✅ Using Conda

```bash
conda create -n projectenv python=3.10 -y
conda activate projectenv
```

---

## ⚙️ Activate Virtual Environment

### ▶ Windows (PowerShell / CMD)

```bash
projectenv\Scripts\activate
```

### ▶ macOS / Linux

```bash
source projectenv/bin/activate
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the ML Pipeline

From project root:

```bash
python -m src.main
```

This executes the full pipeline:
1. Load and clean data
2. Feature engineering
3. Train models
4. Evaluate models
5. Save artifacts
6. Produce logs and reports

---

## 📈 What You Get as Output

✅ **Processed Data**
```
data/processed/processed_data.csv
```

✅ **Saved Models**
```
models/logistic_model.pkl
models/tree_model.pkl
models/xgb_model.pkl
models/scaler.pkl
```

✅ **Evaluation Report**
```
reports/model_comparison.csv
```

✅ **Logs**
```
logs/fraud_detection_YYYY-MM-DD.log
```

---

## 📊 Model Evaluation Metrics

The models are evaluated using:

| Metric | Meaning |
|--------|---------|
| Recall | Fraud capture rate (primary metric) |
| Precision | Correct fraud output ratio |
| F1-Score | Harmonic mean of precision + recall |
| ROC-AUC | Tradeoff between TPR and FPR |
| PR-AUC | Best for imbalanced datasets |

Sample results:

```
Model                   Recall   Precision   F1_Score   ROC_AUC   PR_AUC
XGBoost                 0.93        0.21       0.34      0.97      0.91
Decision Tree           0.89        0.17       0.29      0.92      0.85
Logistic Regression     0.86        0.13       0.23      0.88      0.78
```

> **Recall is prioritized** because missing fraudulent transactions is costly in real systems.

---

## 🔮 Prediction (After Training)

Use the production predictor:

```python
from src.predict import FraudPredictor

predictor = FraudPredictor()

sample_transaction = [
    50000,  # amount
    1,      # type_CASH_OUT
    0,      # type_TRANSFER
    100000, # oldbalanceOrg
    50000,  # newbalanceOrig
    0,      # oldbalanceDest
    0,      # newbalanceDest
    1,      # merchant
    10,     # hour
    2       # day
]

result = predictor.predict(sample_transaction, threshold=0.3)
print(result)
```

---

## 📝 Logging & Exception Handling

This project uses centralized logging (`src/logger.py`) and custom exceptions (`src/exceptions.py`) so that every error and process step is captured in logs for debugging and monitoring.

---

## ✅ Is This Project Good?

### 💡 Yes — It Is Professional

What makes this project stand out:

✅ Production-style folder structure  
✅ Modular and reusable code  
✅ Handling imbalance with SMOTE  
✅ Hyperparameter Tuning  
✅ Evaluation with multiple metrics  
✅ Logging + Exception Handling  
✅ Version-controlled  
✅ Documented  

**Level:**  
Intermediate → Advanced  
**Portfolio-ready:** ✅  
**Interview talking points:** ✅

---

## 🧠 Interview-Ready Summary

> “I built a full end-to-end fraud detection system using Python, handling data imbalance, performing feature engineering, training and tuning multiple models, and designed robust logging and exception handling. It follows best practices for production code.”

---

## 🚀 Future Improvements

Optional enhancements you can build:

✅ Deploy model via FastAPI REST API  
✅ Dashboard with Streamlit  
✅ Automated threshold optimization  
✅ Monitoring + Alerts  
✅ Unit testing with pytest

---

## 👤 Author

**Bhaskar (Phaneendra)**  
Machine Learning | Data Science Enthusiast# transaction_fraud_detection
# transaction_fraud_detection
# transaction_fraud_detection
# payment_fraud_detection
# payment_fraud_detection
# payment_fraud_detection
# payment_fraud_detection
