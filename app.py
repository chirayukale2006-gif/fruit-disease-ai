import streamlit as st
import pandas as pd
import joblib

model = joblib.load("final_fruit_dss_model.pkl")
encoders = joblib.load("final_label_encoders.pkl")

st.title("Fruit Disease AI Diagnosis System")

plant = st.selectbox("Select Plant", list(encoders["Plant"].classes_))
phenol = st.selectbox("Phenol Level", list(encoders["Phenol"].classes_))

samplecount = st.slider("Affected Leaves", 0, 1000, 100)
humidity = st.slider("Humidity (%)", 0, 100, 70)
temperature = st.slider("Temperature (°C)", 0, 50, 25)
soil_ph = st.slider("Soil pH", 4.0, 9.0, 6.5)

if st.button("Diagnose"):

    plant_code = encoders["Plant"].transform([plant])[0]
    phenol_code = encoders["Phenol"].transform([phenol])[0]

    input_df = pd.DataFrame([[plant_code, samplecount, humidity, temperature, soil_ph, phenol_code]],
                            columns=['Plant','SampleCount','Humidity','Temperature','Soil_pH','Phenol'])

    pred = model.predict(input_df)[0]
    disease = encoders["Disease"].inverse_transform([int(pred)])[0]

    if humidity > 85:
        risk = "High fungal infection risk"
    elif temperature > 35:
        risk = "Heat stress risk"
    else:
        risk = "Moderate disease risk"

    st.success(f"Disease: {disease}")
    st.warning(f"Risk Level: {risk}")
