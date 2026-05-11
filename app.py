# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# 1. Load the Housing model and scaler
# Note: Removed "/content/" so it works on GitHub/Streamlit Cloud
# with open("housing_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# st.title("🏡 House Price Predictor")
# st.write("Enter the details of the house to estimate its price:")

# 2. Inputs based on typical Housing dataset features
# Adjust these names if your columns are different
# area = st.number_input("Total Area (sqft)", min_value=100, value=20000)
# bedrooms = st.number_input("Number of Bedrooms", 1, 10, 3)
# bathrooms = st.number_input("Number of Bathrooms", 1, 5, 2)
# stories = st.number_input("Number of Stories (Floors)", 1, 5, 1)
# parking = st.number_input("Parking Spaces", 0, 5, 1)

# # Categorical inputs (Assuming you used get_dummies in training)
# mainroad = st.selectbox("Main Road Access?", ["Yes", "No"])
# guestroom = st.selectbox("Has Guestroom?", ["Yes", "No"])
# airconditioning = st.selectbox("Has Air Conditioning?", ["Yes", "No"])

# 3. Prepare the data exactly like the training data
# We convert Yes/No to 1/0 to match the model's expectation
# input_data = {
#     'area': area,
#     'bedrooms': bedrooms,
#     'bathrooms': bathrooms,
#     'stories': stories,
#     'parking': parking,
#     'mainroad_yes': 1 if mainroad == "Yes" else 0,
#     'guestroom_yes': 1 if guestroom == "Yes" else 0,
#     'airconditioning_yes': 1 if airconditioning == "Yes" else 0
# }

# # Create a DataFrame to keep feature names in order
# input_df = pd.DataFrame([input_data])
# Create a dictionary with ALL columns from your X_train
# Create dictionary with EVERY column from your training
# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# Load files (Make sure names match GitHub exactly)
# with open("housing_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# st.title("🏡 House Price Predictor")

# 1. User Inputs from your screenshot
# area = st.number_input("Total Area (sqft)", value=20000)
# bedrooms = st.number_input("Bedrooms", value=3)
# bathrooms = st.number_input("Bathrooms", value=2)
# stories = st.number_input("Stories", value=1)
# parking = st.number_input("Parking Spaces", value=1)

# mainroad = st.selectbox("Main Road Access?", ["yes", "no"])
# guestroom = st.selectbox("Has Guestroom?", ["yes", "no"])
# basement = st.selectbox("Has Basement?", ["yes", "no"])
# hotwater = st.selectbox("Has Hot Water Heating?", ["yes", "no"])
# aircon = st.selectbox("Has Air Conditioning?", ["yes", "no"])
# prefarea = st.selectbox("Is it in a Preferred Area?", ["yes", "no"])
# furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# 2. Match the exact 13 columns from your training
# input_data = {
#     'area': area,
#     'bedrooms': bedrooms,
#     'bathrooms': bathrooms,
#     'stories': stories,
#     'mainroad_yes': 1 if mainroad == "yes" else 0,
#     'guestroom_yes': 1 if guestroom == "yes" else 0,
#     'basement_yes': 1 if basement == "yes" else 0,
#     'hotwaterheating_yes': 1 if hotwater == "yes" else 0,
#     'airconditioning_yes': 1 if aircon == "yes" else 0,
#     'parking': parking,
#     'prefarea_yes': 1 if prefarea == "yes" else 0,
#     'furnishingstatus_semi-furnished': 1 if furnishing == "semi-furnished" else 0,
#     'furnishingstatus_unfurnished': 1 if furnishing == "unfurnished" else 0
# }

# input_df = pd.DataFrame([input_data])

# 3. Prediction
# if st.button("Predict Price"):
    # # Scaling and Predicting
    # scaled_features = scaler.transform(input_df)
    # prediction = model.predict(scaled_features)
    
    # # Show result
    # price = float(prediction[0])
    # st.success(f"### 💰 Estimated House Price: ${price:,.2f}")

# 4. Prediction
# if st.button("Predict Price"):
    # Scale the features
    # scaled_features = scaler.transform(input_df)
    
    # Get prediction
    # prediction = model.predict(scaled_features)
    
    # Display the result as currency
    # price = float(prediction[0])
    # st.success(f"### 💰 Estimated House Price: ${price:,.2f}")
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load files
with open("housing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🏡 House Price Predictor")

# 1. Inputs
area = st.number_input("Total Area (sqft)", value=1500)
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Bathrooms", value=2)
stories = st.number_input("Stories", value=1)
parking = st.number_input("Parking Spaces", value=1)

mainroad = st.selectbox("Main Road Access?", ["yes", "no"])
guestroom = st.selectbox("Has Guestroom?", ["yes", "no"])
basement = st.selectbox("Has Basement?", ["yes", "no"])
hotwater = st.selectbox("Has Hot Water Heating?", ["yes", "no"])
aircon = st.selectbox("Has Air Conditioning?", ["yes", "no"])
prefarea = st.selectbox("Is it in a Preferred Area?", ["yes", "no"])
furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# 2. Convert to numeric array (Exactly 13 features)
# Making sure the order is exactly as per the common Housing dataset
features = [
    area, 
    bedrooms, 
    bathrooms, 
    stories, 
    1 if mainroad == "yes" else 0,
    1 if guestroom == "yes" else 0,
    1 if basement == "yes" else 0,
    1 if hotwater == "yes" else 0,
    1 if aircon == "yes" else 0,
    parking,
    1 if prefarea == "yes" else 0,
    1 if furnishing == "semi-furnished" else 0,
    1 if furnishing == "unfurnished" else 0
]

# 3. Prediction Section
if st.button("Predict Price"):
    # We convert to numpy and RESHAPE so it doesn't look for names
    features_array = np.array(features).reshape(1, -1)
    
    # Scale and Predict
    try:
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)
        price = float(prediction[0])
        st.success(f"### 💰 Estimated House Price: ${price:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check your logs to see if the model expected more or fewer than 13 features.")
