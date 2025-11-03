import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# App title and description
st.title("🏠 Bangalore Home Price Prediction App")
st.write("Enter the property details below to estimate the price (in Lakhs).")

# Input fields
area = st.number_input("Total Area (sqft)", min_value=200, max_value=10000, step=50)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
location = st.text_input("Location (optional, e.g., Whitefield)")

# Prediction button
if st.button("Predict Price"):
    # Prepare input for model — adjust order/columns to match training
    X = np.array([[area, bhk, bath]])
    prediction = model.predict(X)[0]
    st.success(f"🏡 Estimated Price: ₹ {prediction:,.2f} Lakhs")


