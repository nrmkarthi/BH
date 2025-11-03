import streamlit as st
import numpy as np
import pickle

# Load the trained model and column data
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Optional: Load columns (if available from training)
try:
    with open('columns.pickle', 'rb') as f:
        data_columns = pickle.load(f)
except FileNotFoundError:
    data_columns = ['area', 'bhk', 'bath', 'location']

st.title("🏠 Bangalore Home Price Prediction App")
st.write("Enter the property details below to estimate the price (in Lakhs).")

# User inputs
area = st.number_input("Total Area (sqft)", min_value=200, max_value=10000, step=50)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
location = st.text_input("Location (e.g., Whitefield, Electronic City, Indiranagar)").strip().lower()

if st.button("Predict Price"):
    # Create input array based on training columns
    x = np.zeros(len(data_columns))
    x[0] = area
    x[1] = bhk
    x[2] = bath
    
    if 'location' in data_columns:
        try:
            loc_index = data_columns.index(location)
            x[loc_index] = 1
        except ValueError:
            st.warning(f"Location '{location}' not found in training data. Using default average estimate.")

    # Predict
    prediction = model.predict([x])[0]
    st.success(f"🏡 Estimated Price: ₹ {prediction:,.2f} Lakhs")
