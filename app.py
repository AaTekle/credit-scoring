import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Load the trained model
model = CatBoostClassifier()
model.load_model("catboost_credit_model.cbm")

# Streamlit UI
st.title("Credit Scoring Prediction App")
st.write("This is a machine learning-powered credit loan classification app that predicts the likelihood of a borrower defaulting on a loan. Simply input borrower details like credit limit, repayment history, and payment amounts, and the app provides an easy-to-understand default probability to aid in credit decision-making.")
st.write("Enter the borrower's details below to predict the likelihood of loan default.")

# Input features
st.header("Borrower Information")
st.write("### Credit Limit")
st.write("The total amount of credit available to the borrower. Enter a value in your local currency (e.g., USD).")
limit_bal = st.number_input("Credit Limit (Total Available Credit)", min_value=0, value=50000)

st.write("### Gender")
st.write("Select the borrower's gender. Male = 1, Female = 2.")
sex = st.selectbox(
    "Gender",
    [1, 2],
    format_func=lambda x: "Male" if x == 1 else "Female"
)

st.write("### Education Level")
st.write("1 = Graduate School, 2 = University, 3 = High School, 4 = Other.")
education = st.selectbox(
    "Education Level",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Other"
    }[x]
)

st.write("### Marital Status")
st.write("1 = Married, 2 = Single, 3 = Other.")
marriage = st.selectbox(
    "Marital Status",
    [1, 2, 3],
    format_func=lambda x: {
        1: "Married",
        2: "Single",
        3: "Other"
    }[x]
)

st.write("### Age")
st.write("Enter the borrower's age in years.")
age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30)

st.header("Payment History")
st.write("### Repayment Status")
st.write("""
- PAY_0: Last month's repayment status.
- PAY_2 to PAY_6: Repayment status for the previous months.
""")
st.write(""" How to Insert Numeric Values:
- -2: No credit usage.
- -1: Account paid in full.
- 0: Payment made on time.
- 1 to 9: Number of months payment is delayed.
""")
pay_0 = st.number_input("Repayment Status Last Month (PAY_0)", min_value=-2, max_value=9, value=0)
pay_2 = st.number_input("Repayment Status Two Months Ago (PAY_2)", min_value=-2, max_value=9, value=0)
pay_3 = st.number_input("Repayment Status Three Months Ago (PAY_3)", min_value=-2, max_value=9, value=0)
pay_4 = st.number_input("Repayment Status Four Months Ago (PAY_4)", min_value=-2, max_value=9, value=0)
pay_5 = st.number_input("Repayment Status Five Months Ago (PAY_5)", min_value=-2, max_value=9, value=0)
pay_6 = st.number_input("Repayment Status Six Months Ago (PAY_6)", min_value=-2, max_value=9, value=0)

st.header("Billing and Payment Information")
st.write("### Billing Amounts")
st.write("""
- BILL_AMT1 to BILL_AMT6: Credit card statement balance at the end of each month (in your local currency, e.g., USD).
""")
bill_amt1 = st.number_input("Last Month's Bill Amount (BILL_AMT1)", min_value=0, value=0)
bill_amt2 = st.number_input("Two Months Ago Bill Amount (BILL_AMT2)", min_value=0, value=0)
bill_amt3 = st.number_input("Three Months Ago Bill Amount (BILL_AMT3)", min_value=0, value=0)
bill_amt4 = st.number_input("Four Months Ago Bill Amount (BILL_AMT4)", min_value=0, value=0)
bill_amt5 = st.number_input("Five Months Ago Bill Amount (BILL_AMT5)", min_value=0, value=0)
bill_amt6 = st.number_input("Six Months Ago Bill Amount (BILL_AMT6)", min_value=0, value=0)

st.write("### Payment Amounts")
st.write("""
- PAY_AMT1 to PAY_AMT6: Payments made toward the credit card balance each month (in your local currency, e.g., USD).
""")
pay_amt1 = st.number_input("Last Month's Payment Amount (PAY_AMT1)", min_value=0, value=0)
pay_amt2 = st.number_input("Two Months Ago Payment Amount (PAY_AMT2)", min_value=0, value=0)
pay_amt3 = st.number_input("Three Months Ago Payment Amount (PAY_AMT3)", min_value=0, value=0)
pay_amt4 = st.number_input("Four Months Ago Payment Amount (PAY_AMT4)", min_value=0, value=0)
pay_amt5 = st.number_input("Five Months Ago Payment Amount (PAY_AMT5)", min_value=0, value=0)
pay_amt6 = st.number_input("Six Months Ago Payment Amount (PAY_AMT6)", min_value=0, value=0)

# Combine inputs into a DataFrame with all required features
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

# Predict
if st.button("Predict Default Probability"):
    prediction = model.predict_proba(input_data)[:, 1][0]  # Get probability of default
    st.success(f"Default Probability: {prediction:.2%}")
