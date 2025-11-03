import streamlit as st
import pickle
import numpy as np

# Load model and column data
with open("banglore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    data_columns = pickle.load(f)

st.title("🏠 Bengaluru House Price Predictor")

# User input
location = st.text_input("Enter Location (e.g., Whitefield, Kothanur, Marathahalli):")
sqft = st.number_input("Enter Total Square Feet:", min_value=300.0, max_value=10000.0, value=1000.0)
bath = st.number_input("Enter Number of Bathrooms:", min_value=1, max_value=10, value=2)
bhk = st.number_input("Enter Number of Bedrooms (BHK):", min_value=1, max_value=10, value=2)

# Predict button
if st.button("Estimate Price"):
    try:
        # Prepare input array
        x = np.zeros(len(data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        # Match location (case-insensitive, but keep data_columns format)
        matched_loc = None
        for col in data_columns:
            if col.strip().lower() == location.strip().lower():
                matched_loc = col
                break

        if matched_loc:
            loc_index = data_columns.index(matched_loc)
            x[loc_index] = 1
        else:
            st.warning(f"⚠️ Location '{location}' not found in training data. Using average location effect.")

        # Predict and clean result
        predicted_price = round(model.predict([x])[0], 2)

        # Avoid negative predictions
        if predicted_price < 0:
            predicted_price = 0.0

        st.success(f"🏡 **Estimated Price: ₹ {predicted_price:.2f} Lakhs**")

    except Exception as e:
        st.error(f"Error: {str(e)}")

