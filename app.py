import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Loading in the trained model
model = CatBoostClassifier()
model.load_model("catboost_credit_model.cbm")

# Custom CSS for modern UI
st.markdown(
    """
    <style>
    body {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, #ece9e6, #ffffff);
        color: #333333;
    }
    .stButton button {
        background: linear-gradient(90deg, rgba(0, 0, 0, 1) 0%, rgba(30, 30, 30, 1) 100%);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, rgba(0, 0, 0, 1) 0%, rgba(30, 30, 30, 1) 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Loading in the Google Font
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("Credit Scoring Prediction App")
st.write(
    "This is a machine learning-powered credit loan classification app that predicts the likelihood of a borrower defaulting on a loan. Simply input borrower details like credit limit, repayment history, and payment amounts, and the app provides an easy-to-understand default probability, helping with credit decision-making."
)
st.write(
    "Enter the borrower's details below to predict the likelihood of loan default."
)

# Input features with expanders
with st.expander("Borrower Information"):
    st.write("Provide the details below:")
    limit_bal = st.number_input("Credit Limit (Total Available Credit)", min_value=0, value=50000)
    sex = st.selectbox(
        "Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female"
    )
    education = st.selectbox(
        "Education Level",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Graduate School",
            2: "University",
            3: "High School",
            4: "Other"
        }[x],
    )
    marriage = st.selectbox(
        "Marital Status",
        [1, 2, 3],
        format_func=lambda x: {
            1: "Married",
            2: "Single",
            3: "Other"
        }[x],
    )
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30)

with st.expander("Payment History"):
    st.write(
        "Enter repayment history details for the borrower. Use the provided numeric codes for repayment status."
    )
    pay_0 = st.number_input("Repayment Status Last Month (PAY_0)", min_value=-2, max_value=9, value=0)
    pay_2 = st.number_input("Repayment Status Two Months Ago (PAY_2)", min_value=-2, max_value=9, value=0)
    pay_3 = st.number_input("Repayment Status Three Months Ago (PAY_3)", min_value=-2, max_value=9, value=0)
    pay_4 = st.number_input("Repayment Status Four Months Ago (PAY_4)", min_value=-2, max_value=9, value=0)
    pay_5 = st.number_input("Repayment Status Five Months Ago (PAY_5)", min_value=-2, max_value=9, value=0)
    pay_6 = st.number_input("Repayment Status Six Months Ago (PAY_6)", min_value=-2, max_value=9, value=0)

with st.expander("Billing and Payment Information"):
    st.write("Enter the billing and payment details for the borrower.")
    bill_amt1 = st.number_input("Last Month's Bill Amount (BILL_AMT1)", min_value=0, value=0)
    bill_amt2 = st.number_input("Two Months Ago Bill Amount (BILL_AMT2)", min_value=0, value=0)
    bill_amt3 = st.number_input("Three Months Ago Bill Amount (BILL_AMT3)", min_value=0, value=0)
    bill_amt4 = st.number_input("Four Months Ago Bill Amount (BILL_AMT4)", min_value=0, value=0)
    bill_amt5 = st.number_input("Five Months Ago Bill Amount (BILL_AMT5)", min_value=0, value=0)
    bill_amt6 = st.number_input("Six Months Ago Bill Amount (BILL_AMT6)", min_value=0, value=0)
    pay_amt1 = st.number_input("Last Month's Payment Amount (PAY_AMT1)", min_value=0, value=0)
    pay_amt2 = st.number_input("Two Months Ago Payment Amount (PAY_AMT2)", min_value=0, value=0)
    pay_amt3 = st.number_input("Three Months Ago Payment Amount (PAY_AMT3)", min_value=0, value=0)
    pay_amt4 = st.number_input("Four Months Ago Payment Amount (PAY_AMT4)", min_value=0, value=0)
    pay_amt5 = st.number_input("Five Months Ago Payment Amount (PAY_AMT5)", min_value=0, value=0)
    pay_amt6 = st.number_input("Six Months Ago Payment Amount (PAY_AMT6)", min_value=0, value=0)

# Combining inputs into a DataFrame with all required features
input_data = pd.DataFrame({
    'LIMIT_BAL': [limit_bal],
    'SEX': [sex],
    'EDUCATION': [education],
    'MARRIAGE': [marriage],
    'AGE': [age],
    'PAY_0': [pay_0],
    'PAY_2': [pay_2],
    'PAY_3': [pay_3],
    'PAY_4': [pay_4],
    'PAY_5': [pay_5],
    'PAY_6': [pay_6],
    'BILL_AMT1': [bill_amt1],
    'BILL_AMT2': [bill_amt2],
    'BILL_AMT3': [bill_amt3],
    'BILL_AMT4': [bill_amt4],
    'BILL_AMT5': [bill_amt5],
    'BILL_AMT6': [bill_amt6],
    'PAY_AMT1': [pay_amt1],
    'PAY_AMT2': [pay_amt2],
    'PAY_AMT3': [pay_amt3],
    'PAY_AMT4': [pay_amt4],
    'PAY_AMT5': [pay_amt5],
    'PAY_AMT6': [pay_amt6],
})

# Predictions
if st.button("Predict Default Probability"):
    prediction = model.predict_proba(input_data)[:, 1][0]  # Retreiving the probability of a default
    st.success(f"Default Probability: {prediction:.2%}")
