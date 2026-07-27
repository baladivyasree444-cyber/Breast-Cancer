import streamlit as st
import pickle
import numpy as np
import joblib
@st.cache_resource
def load_model():
    with open("breast_cancer_model.pkl", "rb") as f:
        #return pickle.load(f)
        return joblib.load("breast_cancer_model.pkl")


model = load_model()


st.title("Breast Cancer Prediction App")
st.write("Enter the following details to predict breast cancer:")


st.sidebar.header("Patient Inputs")
st.sidebar.write("Please provide the following details:")


radius_mean = st.sidebar.number_input("Radius Mean", 0.0, 30.0, 14.0)
perimeter_mean = st.sidebar.number_input("Perimeter Mean", 0.0, 200.0, 90.0)
area_mean = st.sidebar.number_input("Area Mean", 0.0, 2500.0, 600.0)
concavity_mean = st.sidebar.number_input("Concavity Mean", 0.0, 0.5, 0.1)
concave_points_mean = st.sidebar.number_input("Concave Points Mean", 0.0, 0.2, 0.05)
radius_worst = st.sidebar.number_input("Radius Worst", 0.0, 30.0, 16.0)
perimeter_worst = st.sidebar.number_input("Perimeter Worst", 0.0, 200.0, 100.0)
area_worst = st.sidebar.number_input("Area Worst", 0.0, 2500.0, 800.0)
concavity_worst = st.sidebar.number_input("Concavity Worst", 0.0, 0.5, 0.15)
concave_points_worst = st.sidebar.number_input("Concave Points Worst", 0.0, 0.2, 0.07)


if st.button("Predict Breast Cancer"):
    features = np.array([[
        radius_mean, perimeter_mean, area_mean, concavity_mean,
        concave_points_mean, radius_worst, perimeter_worst,
        area_worst, concavity_worst, concave_points_worst
    ]]) 

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    st.subheader("Result")

    if prediction == 1:
        st.error(f" Likely Malignant\n\nRisk Score: {probability:.1f}%")
    else:
        st.success(f" Likely Benign\n\nRisk Score: {probability:.1f}%")

    st.progress(float(probability / 100))


    st.write("Note: This prediction is based on a machine learning model and should not be considered a definitive diagnosis. Please consult a healthcare professional for medical advice.")        









