# Credit Scoring Model

## Project Overview

I built a credit scoring model to predict the likelihood of credit card default using machine learning. The goal is to predict whether a client will default (1) or not (0). The model uses borrower attributes and payment history as input features.
---

## Process
1. **Data Cleaning**: I dropped unnecessary columns, ensured numeric data types, and handled categorical variables.
2. **Feature Engineering**: I included repayment history, bill amounts, and demographic features to train the model.
3. **Model Training**: I used CatBoostClassifier for its efficiency with categorical data and high performance on tabular datasets.
4. **Evaluation**: The model’s performance was measured using metrics like ROC-AUC and accuracy.

---

## Files
- **`catboost_credit_model.cbm`**: Trained CatBoost model saved in `.cbm` format.
- **`app.py`**: UI that allows you to make real-time predictions with the model.

---

## How to Use
1. **Load the Model**: Use the `.cbm` file to load the CatBoost model in any Python environment.
2. **Predict**: Input borrower details (e.g., credit limit, repayment history, demographic data) to predict default probability.
3. **Interface**: Run `app.py` with Streamlit to get a web-based interface for the model.

---

## Dataset Citation
If you use this project or dataset, cite the dataset:  
Yeh, I. (2009). *Default of Credit Card Clients [Dataset].* UCI Machine Learning Repository. [https://doi.org/10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
