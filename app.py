import streamlit as st
import numpy as np
import joblib
import pandas as pd

# PART B - LOAD MODEL
try:
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("model.joblib")
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

# PART C - CREATE PREDICTION FUNCTION
def predict_price(latitude, longitude, bedrooms, bathrooms, floorAreaSqM, livingRooms, tenure, property_type):
    # Convert tenure
    tenure_leasehold = 1 if tenure == "LEASEHOLD" else 0
    
    # Convert propertyType into dummy variables
    prop_detached = 1 if property_type == "DETACHED" else 0
    prop_semi_detached = 1 if property_type == "SEMI_DETACHED" else 0
    prop_terraced = 1 if property_type == "TERRACED" else 0
    
    # Create numpy array
    # Order matched exactly what train_model.py generated after pd.get_dummies
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
    
    # Predict using model
    pred_price_log = model.predict(input_scaled)
    
    # Convert back using np.exp()
    final_price = np.exp(pred_price_log[0])
    
    return final_price

# PART D - CREATE USER INTERFACE
st.title("Property Price Prediction App")
st.header("Enter Property Details")

# Numeric Inputs
latitude = st.number_input("Latitude", value=51.50)
longitude = st.number_input("Longitude", value=-0.12)
bedrooms = st.number_input("Bedrooms", value=3, step=1)
bathrooms = st.number_input("Bathrooms", value=2, step=1)
floorAreaSqM = st.number_input("Floor Area (SqM)", value=100.0)
livingRooms = st.number_input("Living Rooms", value=1, step=1)

# Dropdown Inputs
tenure = st.selectbox("Tenure", ["FREEHOLD", "LEASEHOLD"])
property_type = st.selectbox("Property Type", ["FLAT", "DETACHED", "SEMI_DETACHED", "TERRACED"])

# Prediction Button
if st.button("Predict Price"):
    predicted_value = predict_price(
        latitude, longitude, bedrooms, bathrooms, 
        floorAreaSqM, livingRooms, tenure, property_type
    )
    
    # Show Result
    st.success(f"Predicted Property Price: £{predicted_value:,.2f}")
