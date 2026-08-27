# Vehicle Classification ML System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end machine learning pipeline designed to analyze and classify vehicle specifications, comparing Electric Vehicles (EV) against Internal Combustion Engine (ICE) vehicles using historical specification data from 2015 to 2026.

---

## 📌 Features

* **Data Preprocessing Pipeline:** Automated cleaning, missing value handling, and structured dataset formatting.
* **Feature Scaling & Encoding:** Serialized transformers for scalable categorical encoding and numerical feature normalization.
* **Binary Classification Model:** Optimized Logistic Regression model trained to accurately classify vehicle spec profiles.
* **Interactive Dashboard:** Dedicated visualization interface for analyzing spec distributions and model performance.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & Preprocessing:** Scikit-Learn, Joblib
* **Data Visualization & Dashboard:** Matplotlib, Seaborn, Streamlit / Dash

---

## 📁 Repository Structure

| Component | Path / File | Description |
| :--- | :--- | :--- |
| **Raw Dataset** | `Data/Raw/EV_vs_ICE_Vehicle_Specs_2015_2026.csv` | Historical vehicle specifications comparing EV and ICE attributes from 2015 to 2026. |
| **Processed Dataset** | `Data/Processed/Cleaned_EV_Vehicle_Specs.csv` | Cleaned and formatted dataset structured for model training. |
| **Classification Model** | `Artifacts/Logistic_Regression_Model.joblib` | Serialized Logistic Regression model for vehicle classification. |
| **Preprocessing Scaler** | `Artifacts/StandardScaler_Logistic_Regression.joblib` | Saved standard scaler transformer for numerical feature scaling. |
| **Label Encoder** | `Artifacts/LabelEncoder_Fuel_Type.joblib` | Serialized label encoder for encoding categorical fuel type variables. |
| **Analytics Dashboard** | `Dashboard/` | Directory dedicated to visual analytics and interactive reporting interfaces. |
| **Project Settings** | `.idea/` | IDE configuration and inspection profiles for development setups. |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/vehicle-classification-ml-system.git](https://github.com/your-username/vehicle-classification-ml-system.git)
cd vehicle-classification-ml-system
```

### 2. Set Up Virtual Environment & Install Dependencies
```Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Load Model Artifacts in Python
```Python
import joblib
import pandas as pd

# Load serialized preprocessing artifacts and model
model = joblib.load('Artifacts/Logistic_Regression_Model.joblib')
scaler = joblib.load('Artifacts/StandardScaler_Logistic_Regression.joblib')
label_encoder = joblib.load('Artifacts/LabelEncoder_Fuel_Type.joblib')

# Predict on preprocessed numerical features
# predictions = model.predict(scaled_data)
```
📊 Workflow Overview

● Data Cleaning & Ingestion: Raw specification data covering 2015 through 2026 is ingested and formatted into a cleaned dataset saved at Data/Processed/Cleaned_EV_Vehicle_Specs.csv.

● Feature Encoding & Scaling: Categorical variables are encoded using LabelEncoder_Fuel_Type.joblib, while numerical feature variables are normalized using StandardScaler_Logistic_Regression.joblib.

● Model Inference: Preprocessed features are passed to a trained Logistic Regression model stored in Logistic_Regression_Model.joblib to classify vehicle types.

● Visual Analytics: Predictions, feature distributions, and dataset metrics can be interactively analyzed within the Dashboard/ directory.
