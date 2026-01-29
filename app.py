import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bank Prediction", layout="centered")

st.title("Portuguese Bank Marketing Prediction")

# --- Create SMALL dummy dataset (no CSV, no pickle) ---
data = {
    "age": [25, 35, 45, 30, 50, 40],
    "balance": [1000, 2000, 1500, 3000, 1200, 2500],
    "housing": [1, 0, 1, 0, 1, 0],
    "loan": [0, 0, 1, 0, 1, 0],
    "y": [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop("y", axis=1)
y = df["y"]

# --- Train model ---
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.subheader("Enter Customer Details")

age = st.slider("Age", 18, 70, 30)
balance = st.number_input("Account Balance", 0, 100000, 1000)
housing = st.selectbox("Housing Loan", ["No", "Yes"])
loan = st.selectbox("Personal Loan", ["No", "Yes"])

housing = 1 if housing == "Yes" else 0
loan = 1 if loan == "Yes" else 0

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, balance, housing, loan]],
        columns=["age", "balance", "housing", "loan"]
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Customer WILL subscribe to term deposit")
    else:
        st.warning("❌ Customer will NOT subscribe")
