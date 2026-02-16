# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green.svg)
![Status](https://img.shields.io/badge/Project-Completed-success.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

The **Credit Card Fraud Detection System** is a machine learning project designed to identify fraudulent transactions with high precision.  
It leverages the Kaggle Credit Card dataset and addresses extreme class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)** to ensure the model learns rare fraud patterns effectively.

---

## 📌 Table of Contents

- [📊 Dataset](#-dataset)
- [🎯 Problem Statement](#-problem-statement)
- [✨ Features](#-features)
- [🤖 Models Used](#-models-used)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [🏆 Results](#-results)
- [⚙️ Installation & Setup](#️-installation--setup)
- [📁 Project Structure](#-project-structure)
- [📜 License](#-license)

---

## 📊 Dataset

The dataset used for this project contains transactions made by European cardholders in September 2013.

- **Total Transactions:** 284,807  
- **Fraudulent Transactions:** 492 (0.17%)  
- **Features:** V1–V28 (PCA-transformed components), `Time`, and `Amount`  
- **Target Variable:** `Class`  
  - `1` → Fraud  
  - `0` → Genuine  

🔗 Dataset Source:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🎯 Problem Statement

In financial fraud detection, **Accuracy is misleading**.

Since 99.8% of transactions are legitimate, a model that predicts *“Not Fraud”* every time would achieve 99.8% accuracy — yet be completely useless.

This project prioritizes:

- 🔎 **High Recall** → Minimize False Negatives (missed fraud cases)
- ⚖️ Balanced Precision → Avoid excessive false alarms
- 🎯 Practical real-world fraud detection performance

---

## ✨ Features

✔ Data Preprocessing with Robust Scaling  
✔ SMOTE for Class Imbalance Handling  
✔ Exploratory Data Analysis (EDA)  
✔ Feature Correlation Visualization  
✔ Multiple Model Benchmarking  
✔ Confusion Matrix Analysis  

---

## 🤖 Models Used

The following machine learning models were implemented and compared:

1. **Random Forest Classifier** (Primary Model)
   <img width="3058" height="1629" alt="image" src="https://github.com/user-attachments/assets/9c6b9994-c4b7-4925-ab06-b4bbfbf62c73" />

2. **Logistic Regression** (Baseline Model)
3. **XGBoost Classifier** (Advanced Boosting Model)


---

## 📈 Evaluation Metrics

Since this is a highly imbalanced dataset, the following metrics were prioritized:

- **Precision** → Accuracy of fraud alerts  
- **Recall** → Ability to detect all actual fraud cases  
- **F1-Score** → Balance between Precision & Recall  
- **Confusion Matrix** → Performance visualization  

---

## 🏆 Results

| Model                          | Precision | Recall | F1-Score |
|--------------------------------|-----------|--------|----------|
| Random Forest + SMOTE         | 0.88      | 0.82   | 0.85     |
| Logistic Regression + SMOTE   | 0.06      | 0.91   | 0.11     |

### ✅ Conclusion

The **Random Forest model with SMOTE** achieved the best balance between precision and recall.

Although Logistic Regression achieved higher recall, it produced excessive false positives (very low precision), making it impractical for real-world deployment.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection 

2️⃣ Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost

3️⃣ Dataset Setup

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place creditcard.csv inside the project root directory.

📁 Project Structure
Credit-Card-Fraud-Detection/
│
├── creditcard.csv
├── fraud_detection.ipynb
├── requirements.txt
└── README.md

👨‍💻 Author

Kamal Pokhrel
GitHub: https://github.com/kmlPokhrel
