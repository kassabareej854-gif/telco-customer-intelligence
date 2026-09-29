Telco Customer Intelligence

A machine learning project for analyzing telecom customer behavior and building predictive models for CLTV estimation, customer churn prediction, and customer segmentation.

Project Overview

This project uses the Telco Customer Churn dataset to build an end-to-end customer intelligence workflow:
Data preprocessing and exploratory data analysis (EDA)
Customer Lifetime Value (CLTV) proxy estimation using regression
Customer churn prediction using classification
Customer segmentation using K-Means clustering
Interactive Streamlit dashboard for customer-level predictions and analysis
Project Structure

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
> The exact folder structure may vary depending on how the project files are uploaded to GitHub.
Notebooks
01 — Data Preprocessing & EDA
This notebook prepares the dataset for machine learning and explores its main characteristics.
Main steps:
Inspect dataset shape and data types
Check missing values and duplicate records
Convert `Total Charges` to numeric
Clean text fields
Remove duplicate rows
Explore customer and service-related variables
02 — CLTV Regression
This notebook estimates a CLTV proxy using customer and service characteristics.
The target is selected as:
`Total Revenue`, if available
Otherwise, `Total Charges`
The notebook includes:
Feature selection
Train/test split
Numerical and categorical preprocessing
Median imputation for numerical features
Most-frequent imputation for categorical features
Standardization
One-hot encoding
Regression model training
Model evaluation using MAE, RMSE, and R²
Selection and saving of the trained model
03 — Churn Classification
This notebook predicts whether a customer is likely to churn.
The target is encoded as:
```text
No  → 0
Yes → 1
```
The workflow includes:
Feature preparation
Train/test split
Data preprocessing
Classification model training
Churn prediction
Model evaluation using classification metrics
Saving the trained churn model
04 — Customer Segmentation
This notebook groups customers into segments based on selected customer-level characteristics.
The segmentation workflow includes:
Feature selection
Feature scaling
K-Means clustering
Evaluation of different cluster numbers using silhouette score
Final cluster assignment
Saving the K-Means model and scaler
Models
The project includes several machine learning approaches for different tasks.
Regression
Linear Regression
Ridge Regression
Random Forest Regressor
Gradient Boosting Regressor
Classification
Logistic Regression
Random Forest Classifier
Clustering
K-Means
Evaluation Metrics
Regression
MAE (Mean Absolute Error): measures the average absolute prediction error.
RMSE (Root Mean Squared Error): gives greater weight to larger prediction errors.
R²: measures how much of the variation in the target is explained by the model.
Classification
The churn model can be evaluated using:
Accuracy
Precision
Recall
F1-score
ROC-AUC
Clustering
Silhouette Score
Streamlit Dashboard
The project includes an interactive Streamlit application in:
```text
app.py
```
The dashboard is designed to provide customer-level intelligence, including:
Churn prediction
CLTV estimation
Customer segmentation
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Streamlit
Jupyter Notebook
Installation
Clone the repository:
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd telco-customer-intelligence
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Running the Dashboard
Run the Streamlit application with:
```bash
streamlit run app.py
```
Dataset
The project is based on the Telco Customer Churn dataset.
The dataset contains customer demographics, account information, subscribed services, charges, and churn-related information.
Project Goal
The goal of this project is to demonstrate an end-to-end customer analytics workflow that combines:
Data Preparation → Exploratory Analysis → Prediction → Segmentation → Interactive Dashboard
This project was developed as a practical machine learning and data analytics portfolio project.
