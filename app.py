import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Prediction App")

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader(" Enter Patient Health Information")

# Use layout for cleaner input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, step=1)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 400)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])

with col2:
    restecg = st.selectbox("Rest ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250)
    exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"])

# Convert categorical inputs to numeric values
sex_val = 1 if sex == "Male" else 0
cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
fbs_val = 1 if fbs == "True" else 0
restecg_val = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
exang_val = 1 if exang == "Yes" else 0
slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal_val = ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"].index(thal)

# Prediction
if st.button(" Predict"):
    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                            thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("High Risk: The person may have heart disease. Please consult a doctor.")
    else:
        st.success("Low Risk: The person is unlikely to have heart disease.")

