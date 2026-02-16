# Credit-Card-Fraud-Detection
Machine Learning project to detect fraud using SMOTE and Random Forest.


The Credit Card Fraud Detection System is a machine learning project designed to identify fraudulent transactions with high precision. It leverages the Kaggle Credit Card dataset, addressing extreme class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model learns the patterns of "needle-in-a-haystack" fraud cases.

# 💳 Credit Card Fraud Detection System

The **Credit Card Fraud Detection System** is a machine learning project designed to identify fraudulent transactions with high precision. It leverages the Kaggle Credit Card dataset, addressing extreme class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model learns the patterns of "needle-in-a-haystack" fraud cases.

---

## 📋 Table of Contents
* [📊 Dataset](#-dataset)
* [⚙️ Installation](#️-installation)
* [🎯 Problem Statement](#-problem-statement)
* [✨ Features](#-features)
* [🤖 Models](#-models)
* [📈 Evaluation Metrics](#-evaluation-metrics)
* [🏆 Results](#-results)
* [⚖️ License](#️-license)

---

## 📊 Dataset
The dataset used for this project contains transactions made by European cardholders in September 2013.
* **Total Transactions:** 284,807
* **Fraudulent Transactions:** 492 (0.17%)
* **Features:** V1-V28 (PCA-transformed components), Time, and Amount.
* **Target:** `Class` (1 for Fraud, 0 for Genuine).

---

## ⚙️ Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Credit-Card-Fraud-Detection](https://github.com/your-username/Credit-Card-Fraud-Detection)
   cd Credit-Card-Fraud-Detection

📊 Dataset
The dataset used for this project contains transactions made by European cardholders.

Total Transactions: 284,807

Fraudulent Transactions: 492 (0.17%)

Features: V1-V28 (PCA components), Time, and Amount.

Target: Class (1 for Fraud, 0 for Genuine).

⚙️ Installation
To run this project locally, follow these steps:

Clone the repository:

Bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection
cd Credit-Card-Fraud-Detection
Install required dependencies:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
Download the Dataset: Place the creditcard.csv file in the root directory.

🎯 Problem Statement
In financial fraud detection, Accuracy is a trap. Because 99.8% of transactions are legitimate, a model that simply predicts "Not Fraud" every time would be 99.8% accurate but entirely useless. This project focuses on Recall—minimizing "False Negatives" (missed frauds) to protect the user and the bank.

✨ Features
Data Preprocessing: Robust scaling of transaction amounts to handle outliers.

Imbalance Handling: Implementation of SMOTE to balance the training data.

EDA: Detailed visualizations of feature correlations and class distributions.

Model Comparison: Parallel testing of multiple algorithms.

🤖 Models
The project evaluates the performance of the following classifiers:

Random Forest Classifier (Primary Model)

Logistic Regression (Baseline Model)

XGBoost (Advanced Gradient Boosting)

<img width="3058" height="1629" alt="image" src="https://github.com/user-attachments/assets/a2d35b05-0810-4050-8dc3-85a443e4e587" />

📈 Evaluation Metrics
Since this is an imbalanced classification problem, we evaluate models based on:

Precision: Accuracy of fraud alarms.

Recall: The ability to find all fraudulent transactions.

F1-Score: The harmonic mean of Precision and Recall.

Confusion Matrix: To visualize True Positives vs. False Negatives.

🏆 Results
Model,Precision,Recall,F1-Score
Random Forest + SMOTE,0.88,0.82,0.85
Logistic Regression + SMOTE,0.06,0.91,0.11

Gemini said
You want that professional, "Open Source" look. Since this is your first project and you're heading toward a GitHub upload, this structure will make it look like you've been doing this for years.

Here is your Credit Card Fraud Detection README.md, styled exactly like the high-quality IPL repository you found.

💳 Credit Card Fraud Detection System
The Credit Card Fraud Detection System is a machine learning project designed to identify fraudulent transactions with high precision. It leverages the Kaggle Credit Card dataset, addressing extreme class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model learns the patterns of "needle-in-a-haystack" fraud cases.

📋 Table of Contents
Dataset

Installation

Problem Statement

Features

Models

Evaluation Metrics

Results

License

📊 Dataset
The dataset used for this project contains transactions made by European cardholders.

Total Transactions: 284,807

Fraudulent Transactions: 492 (0.17%)

Features: V1-V28 (PCA components), Time, and Amount.

Target: Class (1 for Fraud, 0 for Genuine).

⚙️ Installation
To run this project locally, follow these steps:

Clone the repository:

Bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection
cd Credit-Card-Fraud-Detection
Install required dependencies:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
Download the Dataset: Place the creditcard.csv file in the root directory.

🎯 Problem Statement
In financial fraud detection, Accuracy is a trap. Because 99.8% of transactions are legitimate, a model that simply predicts "Not Fraud" every time would be 99.8% accurate but entirely useless. This project focuses on Recall—minimizing "False Negatives" (missed frauds) to protect the user and the bank.

✨ Features
Data Preprocessing: Robust scaling of transaction amounts to handle outliers.

Imbalance Handling: Implementation of SMOTE to balance the training data.

EDA: Detailed visualizations of feature correlations and class distributions.

Model Comparison: Parallel testing of multiple algorithms.

🤖 Models
The project evaluates the performance of the following classifiers:

Random Forest Classifier (Primary Model)

Logistic Regression (Baseline Model)

XGBoost (Advanced Gradient Boosting)

Image of random forest architecture
Shutterstock
📈 Evaluation Metrics
Since this is an imbalanced classification problem, we evaluate models based on:

Precision: Accuracy of fraud alarms.

Recall: The ability to find all fraudulent transactions.

F1-Score: The harmonic mean of Precision and Recall.

Confusion Matrix: To visualize True Positives vs. False Negatives.

🏆 Results
Model	Precision	Recall	F1-Score
Random Forest + SMOTE	0.88	0.82	0.85
Logistic Regression + SMOTE	0.06	0.91	0.11

Conclusion: The Random Forest model achieved the best balance for real-world application, successfully detecting fraud while maintaining a low false-alarm rate.
