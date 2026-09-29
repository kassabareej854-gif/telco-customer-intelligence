# 📊 Telco Customer Intelligence

An end-to-end machine learning project for analyzing telecom customer behavior, predicting customer churn, estimating Customer Lifetime Value (CLTV) as a revenue proxy, and segmenting customers into meaningful groups.

The project includes data preprocessing, exploratory data analysis (EDA), regression, classification, customer segmentation, model evaluation, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Telecom companies need to understand customer behavior in order to:

- Identify customers who are likely to churn.
- Estimate the expected customer value/revenue.
- Discover groups of customers with similar characteristics.
- Support data-driven customer retention and marketing analysis.

This project develops machine learning solutions for these three main tasks:

1. **CLTV / Revenue Prediction** — Regression
2. **Customer Churn Prediction** — Classification
3. **Customer Segmentation** — K-Means Clustering

An interactive **Streamlit dashboard** is also provided to demonstrate the trained models.

---

## 🎯 Project Objectives

### 1. Customer Lifetime Value (CLTV) / Revenue Prediction
Build regression models to estimate customer revenue using demographic, service, contract, tenure, and billing information.

> **Target:** `Total Revenue` when available; otherwise `Total Charges` is used as a CLTV proxy.

### 2. Customer Churn Prediction
Build classification models to predict whether a customer is likely to leave the company.

- `0` → No Churn
- `1` → Churn

### 3. Customer Segmentation
Use K-Means clustering to group customers according to selected behavioral and financial characteristics.

### 4. Interactive Dashboard
Deploy the trained models through a Streamlit application where users can enter customer information and obtain predictions.

---

## 📂 Project Structure

```text
telco-customer-intelligence/
│
├── notebooks/
│   ├── 01_Data_Preprocessing_EDA.ipynb
│   ├── 02_Regression_CLTV.ipynb
│   ├── 03_Classification_Churn.ipynb
│   └── 04_Customer_Segmentation.ipynb
│
├── models/
│   ├── churn_model.pkl
│   ├── cltv_model.pkl
│   ├── kmeans_model.pkl
│   └── segment_scaler.pkl
│
├── data/
│   └── telco_cleaned.csv
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 📓 Notebooks

## 01 — Data Preprocessing & EDA

**File:** `01_Data_Preprocessing_EDA.ipynb`

This notebook prepares the dataset for the machine learning tasks.

### Main steps

- Load the Telco Customer Churn dataset.
- Inspect dataset dimensions and data types.
- Check missing values.
- Check duplicate records.
- Identify constant columns.
- Convert `Total Charges` to numeric.
- Clean whitespace from categorical values.
- Remove duplicate rows.
- Perform exploratory data analysis.
- Visualize important customer characteristics.

### Main EDA areas

- Customer demographics
- Tenure
- Contract type
- Internet service
- Monthly charges
- Total charges
- Churn distribution

---

## 02 — CLTV Regression

**File:** `02_Regression_CLTV.ipynb`

This notebook predicts customer revenue as a proxy for Customer Lifetime Value (CLTV).

### Target selection

The notebook uses:

```text
Total Revenue
```

when available.

If `Total Revenue` is not available, it uses:

```text
Total Charges
```

as the target.

### Features

The regression model uses customer demographic, service, contract, tenure, billing, and payment-related features.

Examples include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure Months
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges

### Models

The notebook compares:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Evaluation metrics

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

The models are evaluated on a held-out test set.

---

## 03 — Customer Churn Classification

**File:** `03_Classification_Churn.ipynb`

This notebook predicts whether a customer will churn.

### Target

The notebook supports either:

```text
Churn Label
```

or:

```text
Churn Value
```

For `Churn Label`:

```text
Yes → 1
No  → 0
```

### Models

Two classification models are compared:

- Logistic Regression
- Random Forest Classifier

Class balancing is used because the dataset contains more non-churn customers than churned customers.

### Evaluation metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

The models are compared using ROC-AUC, and the selected trained pipeline is saved for later use.

---

## 04 — Customer Segmentation

**File:** `04_Customer_Segmentation.ipynb`

This notebook uses **K-Means Clustering** to identify groups of customers with similar characteristics.

### Clustering features

The segmentation uses:

- Tenure Months
- Total Charges
- CLTV
- Churn Score

### Workflow

```text
Customer Data
      ↓
Feature Selection
      ↓
Scaling
      ↓
K-Means
      ↓
Customer Segments
      ↓
Segment Analysis
```

Different values of `K` are evaluated using the **Silhouette Score**.

The selected clustering model and scaler are saved for use in the Streamlit application.

---

# 🤖 Machine Learning Workflow

The complete project follows this workflow:

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ├───────────────┐
     ▼               ▼
Regression       Classification
     │               │
     ▼               ▼
CLTV / Revenue    Churn Prediction
     │               │
     └───────┬───────┘
             ▼
       Customer Segmentation
             │
             ▼
      Streamlit Dashboard
```

---

# 🧠 Preprocessing

The machine learning notebooks use a preprocessing pipeline that handles numerical and categorical variables separately.

### Numerical features

- Missing values → median imputation
- Scaling → StandardScaler

### Categorical features

- Missing values → most-frequent imputation
- Encoding → One-Hot Encoding
- Unknown categories → ignored using `handle_unknown="ignore"`

The preprocessing is integrated into the model pipeline to ensure that the same transformations are applied consistently during training and prediction.

---

# 📈 Model Evaluation

## Regression

Regression models are evaluated using:

| Metric | Purpose |
|---|---|
| MAE | Average absolute prediction error |
| RMSE | Penalizes larger prediction errors more strongly |
| R² | Measures explained variance |

## Classification

Classification models are evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Correct positive predictions among predicted positives |
| Recall | Churn cases correctly identified |
| F1-score | Balance between precision and recall |
| ROC-AUC | Measures class discrimination across thresholds |

A confusion matrix is also generated to visualize:

- True Positives
- True Negatives
- False Positives
- False Negatives

## Clustering

Customer segmentation is evaluated using:

- Silhouette Score

---

# 💾 Saved Models

The trained models are saved using `joblib`.

| File | Purpose |
|---|---|
| `churn_model.pkl` | Trained churn classification pipeline |
| `cltv_model.pkl` | Trained CLTV/revenue regression pipeline |
| `kmeans_model.pkl` | Trained K-Means clustering model |
| `segment_scaler.pkl` | Scaler used before segmentation |

These files allow the Streamlit application to use the trained models without retraining them every time.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive dashboard:

```text
app.py
```

The application allows users to enter customer information and interact with the trained machine learning models.

### Main functionality

- Customer churn prediction
- CLTV / revenue prediction
- Customer segmentation
- Display of prediction results

The dashboard loads the saved `.pkl` model files and applies the same preprocessing used during model training.

---

# 🛠️ Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Model Persistence

- Joblib

### Dashboard

- Streamlit

### Development Environment

- Jupyter Notebook
- Google Colab

---

# 📦 Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project folder:

```bash
cd telco-customer-intelligence
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

From the project directory:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

# 📊 Dataset

The project is based on the **Telco Customer Churn** dataset.

The dataset contains customer demographic, service, contract, billing, and churn-related information.

The main dataset contains:

```text
7,043 customers
33 columns
```

The project performs additional cleaning and feature preparation before applying machine learning models.

---

# 🔍 Key Machine Learning Concepts Demonstrated

This project demonstrates practical applications of:

- Data cleaning
- Exploratory Data Analysis
- Missing-value handling
- Categorical encoding
- Feature scaling
- Train/test splitting
- Machine learning pipelines
- Regression
- Classification
- Class imbalance handling
- Model comparison
- Model evaluation
- Confusion matrices
- ROC-AUC
- K-Means clustering
- Silhouette analysis
- Model serialization
- Streamlit deployment

---

# 🚀 End-to-End Pipeline

```text
1. Load Dataset
        ↓
2. Inspect Data
        ↓
3. Clean Data
        ↓
4. Perform EDA
        ↓
5. Prepare Features and Targets
        ↓
6. Build Preprocessing Pipelines
        ↓
7. Train Multiple Models
        ↓
8. Evaluate Models
        ↓
9. Select Trained Model
        ↓
10. Save Models
        ↓
11. Build Streamlit Dashboard
        ↓
12. Use Saved Models for Predictions
```

---

# 📌 Notes

- The CLTV component uses customer revenue/charges as a practical proxy for customer lifetime value.
- Missing feature values are handled inside the machine learning preprocessing pipeline.
- The churn target is represented as binary values (`0` and `1`).
- K-Means clustering requires feature scaling because the input variables have different numerical ranges.
- The saved models should be used with preprocessing compatible with the preprocessing used during training.

---

# 👩‍💻 Author

**Areege Kassab**

Telco Customer Intelligence — Machine Learning & Data Analytics Project
