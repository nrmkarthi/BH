import streamlit as st
import numpy as np
import pickle
import json

# Load model and columns
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Extract locations from columns
locations = data_columns[3:]  # ['1st Block Jayanagar', 'AECS Layout', 'Whitefield', ...]

st.title("🏠 Bangalore Home Price Prediction App")
st.write("Enter property details to estimate the price (in Lakhs).")

# Input fields
area = st.number_input("Total Area (sqft)", min_value=200, max_value=10000, step=50)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
location = st.selectbox("Location", locations)

# Prediction button
if st.button("Predict Price"):
    # Create input vector
    x = np.zeros(len(data_columns))
    x[0] = area
    x[1] = bath
    x[2] = bhk
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    prediction = round(model.predict([x])[0], 2)
    st.success(f"🏡 Estimated Price: ₹ {abs(prediction):,.2f} Lakhs")


