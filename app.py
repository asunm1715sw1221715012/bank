import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Working ML App")

# Simple dataset
df = pd.DataFrame({
    "age": [20, 30, 40, 50],
    "balance": [1000, 2000, 3000, 4000],
    "y": [0, 1, 0, 1]
})

X = df[["age", "balance"]]
y = df["y"]

model = RandomForestClassifier()
model.fit(X, y)

age = st.slider("Age", 18, 60, 30)
balance = st.number_input("Balance", 0, 10000, 2000)

if st.button("Predict"):
    pred = model.predict([[age, balance]])[0]
    st.success(f"Prediction: {pred}")
