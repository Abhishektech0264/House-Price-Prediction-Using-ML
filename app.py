import streamlit as st
from predict import predict_price

st.title("🏠 House Price Prediction App")

features = [
    st.number_input("CRIM"),
    st.number_input("ZN"),
    st.number_input("INDUS"),
    st.number_input("CHAS"),
    st.number_input("NOX"),
    st.number_input("RM"),
    st.number_input("AGE"),
    st.number_input("DIS"),
    st.number_input("RAD"),
    st.number_input("TAX"),
    st.number_input("PTRATIO"),
    st.number_input("B"),
    st.number_input("LSTAT"),
]

if st.button("Predict Price"):
    result = predict_price(features)
    st.success(f"Estimated House Price: {result:.2f}")
