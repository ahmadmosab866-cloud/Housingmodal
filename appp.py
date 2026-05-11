import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Load the Housing model and scaler
# Note: Removed "/content/" so it works on GitHub/Streamlit Cloud
with open("housing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🏡 House Price Predictor")
st.write("Enter the details of the house to estimate its price:")

# 2. Inputs based on typical Housing dataset features
# Adjust these names if your columns are different
area = st.number_input("Total Area (sqft)", min_value=100, value=1500)
bedrooms = st.number_input("Number of Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Number of Bathrooms", 1, 5, 2)
stories = st.number_input("Number of Stories (Floors)", 1, 5, 1)
parking = st.number_input("Parking Spaces", 0, 5, 1)

# Categorical inputs (Assuming you used get_dummies in training)
mainroad = st.selectbox("Main Road Access?", ["Yes", "No"])
guestroom = st.selectbox("Has Guestroom?", ["Yes", "No"])
airconditioning = st.selectbox("Has Air Conditioning?", ["Yes", "No"])

# 3. Prepare the data exactly like the training data
# We convert Yes/No to 1/0 to match the model's expectation
input_data = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad_yes': 1 if mainroad == "Yes" else 0,
    'guestroom_yes': 1 if guestroom == "Yes" else 0,
    'airconditioning_yes': 1 if airconditioning == "Yes" else 0
}

# Create a DataFrame to keep feature names in order
input_df = pd.DataFrame([input_data])

# 4. Prediction
if st.button("Predict Price"):
    # Scale the features
    scaled_features = scaler.transform(input_df)
    
    # Get prediction
    prediction = model.predict(scaled_features)
    
    # Display the result as currency
    price = float(prediction[0])
    st.success(f"### 💰 Estimated House Price: ${price:,.2f}")
