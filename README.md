# Credit Scoring Model

## Goal
I aimed to predict whether a loan would be repaid or defaulted on. This was a binary classification task, and the output was a credit score or risk label, such as "good credit" or "bad credit."

## Steps Taken / Process:

## Data Loading
* I loaded the dataset from an Excel file that contained information about credit card clients.

## Data Overview
* The dataset is made up of thousands of rows and columns. Key features included `LIMIT_BAL` (credit limit), `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, and various payment histories. The target variable was whether the client defaulted on their payment the following month.

## Data Cleaning
* I cleaned the data by removing the `UID` column, which was not needed for training. I also ensured that all column names were formatted correctly and converted numeric columns to the appropriate data types.

## Feature Engineering
* I defined my features (the input data) and the target variable (the output I wanted to predict). The main features I focused on included `EDUCATION`, `MARRIAGE`, and `AGE`.

## Splitting the Data - Train Test Split
* I split the dataset into training and testing sets to evaluate the model's performance. This helped me understand how well the model could predict loan repayment or default.

## Training the Model
* I trained a CatBoost model using the training data. This model learned from the features to predict the likelihood of default.

## Predicting and Evaluating Model Accuracy
* After training the model, I made predictions on the test data. I evaluated the model's performance using metrics such as accuracy and ROC-AUC score, which indicated how well the model distinguished between good and bad credit risks.

## Quantitative Results
* The model achieved an accuracy of approximately 82% and a ROC-AUC score of about 0.78. This indicated that the model was reasonably effective at predicting loan repayment or default. 
