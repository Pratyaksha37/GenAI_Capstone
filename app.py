import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os

# PART B - LOAD MODEL SAFELY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.joblib")
scaler_path = os.path.join(BASE_DIR, "scaler.joblib")

try:
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# PART C - CREATE PREDICTION FUNCTION
def predict_price(latitude, longitude, bedrooms, bathrooms, floorAreaSqM, livingRooms, tenure, property_type):

    # Convert tenure
    tenure_leasehold = 1 if tenure == "LEASEHOLD" else 0

    # Convert propertyType into dummy variables
    prop_detached = 1 if property_type == "DETACHED" else 0
    prop_semi_detached = 1 if property_type == "SEMI_DETACHED" else 0
    prop_terraced = 1 if property_type == "TERRACED" else 0

    # Create input array
    input_data = np.array([[
        latitude,
        longitude,
        bedrooms,
        bathrooms,
        floorAreaSqM,
        livingRooms,
        tenure_leasehold,
        prop_detached,
        prop_semi_detached,
        prop_terraced
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict log price
    pred_log = model.predict(input_scaled)

    # Convert back to real price
    final_price = np.exp(pred_log[0])

    return final_price


# PART D - STREAMLIT UI

st.title("Property Price Prediction App")
st.header("Enter Property Details")

latitude = st.number_input("Latitude", value=51.50)
longitude = st.number_input("Longitude", value=-0.12)

bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0, step=1)

floorAreaSqM = st.number_input("Floor Area (SqM)", min_value=0.0)
livingRooms = st.number_input("Living Rooms", min_value=0, step=1)

tenure = st.selectbox("Tenure", ["FREEHOLD", "LEASEHOLD"])

property_type = st.selectbox(
    "Property Type",
    ["FLAT", "DETACHED", "SEMI_DETACHED", "TERRACED"]
)

if st.button("Predict Price"):

    price = predict_price(
        latitude,
        longitude,
        bedrooms,
        bathrooms,
        floorAreaSqM,
        livingRooms,
        tenure,
        property_type
    )

    st.success(f"Predicted Property Price: £{price:,.2f}")