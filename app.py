import streamlit as st
import pandas as pd
import numpy as np
import joblib
import geopy.distance
from geopy.geocoders import Nominatim
from datetime import datetime

# Load trained model and encoder once (cached for efficiency)
@st.cache_resource
def load_model():
    return joblib.load("CatBoost.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("target_encoder.pkl")

@st.cache_data
def load_sales_data():
    try:
        return pd.read_csv("cleaned_sales_data.csv")
    except FileNotFoundError:
        st.error("Historical sales data (cleaned_sales_data.csv) not found.")
        return pd.DataFrame()  # Return empty DataFrame to prevent crash

# Function to calculate distance to city center (vectorized)
def calculate_distance(lat, lon, center=(31.0689, -91.9968)):
    coords = np.column_stack((lat, lon))
    center_coords = np.tile(center, (len(coords), 1))
    return np.linalg.norm(coords - center_coords, axis=1)

# Cache Geopy API Calls to avoid redundant address lookups
@st.cache_data
def get_coordinates(address):
    geolocator = Nominatim(user_agent="my_unique_app")
    location = geolocator.geocode(address, timeout=10)
    if location:
        return location.latitude, location.longitude
    return None, None

# Load cached model and encoder
model = load_model()
model_features = model.feature_names_  # Get expected features
target_encoder = load_encoder()
df_sales_history = load_sales_data()

# Streamlit UI
st.title("Real Estate Sale Amount Prediction")
st.write("Enter details to predict the sale amount of a property.")

# Location input
method = st.radio("Choose how to provide location:", ["Enter Longitude & Latitude", "Enter Full Address"])

longitude, latitude = None, None

if method == "Enter Longitude & Latitude":
    longitude = st.number_input("Longitude", format="%.6f", value=0.0)
    latitude = st.number_input("Latitude", format="%.6f", value=0.0)

elif method == "Enter Full Address":
    address = st.text_input("Enter Full Address")
    if address:
        latitude, longitude = get_coordinates(address)
        if latitude is not None and longitude is not None:
            st.success(f"Coordinates: Latitude {latitude}, Longitude {longitude}")
        else:
            st.error("Could not find coordinates. Try refining the address.")

# Collect user inputs
assessed_value = st.number_input("Assessed Value", min_value=0.0, format="%.2f")
town = st.selectbox("Town", df_sales_history["Town"].unique() if not df_sales_history.empty else ["Unknown"])
property_type = st.selectbox("Property Type", df_sales_history["Property Type"].unique() if not df_sales_history.empty else ["Residential"])
sales_ratio = st.number_input("Sales Ratio", min_value=0.0, format="%.4f")
residential_type = st.selectbox("Residential Type", df_sales_history["Residential Type"].unique() if not df_sales_history.empty else ["Single Family"])
date_recorded = st.date_input("Date Recorded", datetime.today())

# Feature Engineering
if latitude is not None and longitude is not None:
    year_sold = date_recorded.year
    month_sold = date_recorded.month
    day_of_week_sold = date_recorded.weekday()
    quarter_sold = (month_sold - 1) // 3 + 1
    days_since_listing = (datetime.today().year - year_sold) * 365
    distance_to_city_center = calculate_distance(np.array([latitude]), np.array([longitude]))[0]

    # Append new input data dynamically instead of modifying full history
    new_entry = pd.DataFrame({
        "Assessed Value": [assessed_value],
        "Longitude": [longitude],
        "Latitude": [latitude],
        "Town": [town],
        "Property Type": [property_type],
        "Sales Ratio": [sales_ratio],
        "Residential Type": [residential_type],
        "Year Sold": [year_sold],
        "Month Sold": [month_sold],
        "Day of Week Sold": [day_of_week_sold],
        "Quarter Sold": [quarter_sold],
        "Days Since Listing": [days_since_listing],
        "Distance to City Center": [distance_to_city_center]
    })

    # Compute additional features from cached historical data
    if not df_sales_history.empty:
        town_avg_price = df_sales_history.groupby('Town')["Sale Amount"].mean().get(town, np.nan)
        new_entry["Town Avg Sale Price"] = town_avg_price
        new_entry["Market Trend Score"] = df_sales_history[df_sales_history["Town"] == town]["Sale Amount"].rolling(3, min_periods=1).mean().iloc[-1] if town in df_sales_history["Town"].values else np.nan
        new_entry["Sales Ratio Normalized"] = (sales_ratio - df_sales_history["Sales Ratio"].mean()) / df_sales_history["Sales Ratio"].std()
        new_entry["Sales Ratio × Town"] = sales_ratio * town_avg_price

    # Encode categorical features
    new_entry[["Property Type", "Residential Type", "Town"]] = target_encoder.transform(
        new_entry[["Property Type", "Residential Type", "Town"]]
    )

    # Select final input features
    final_features = [
        "Sales Ratio", "Assessed Value", "Market Trend Score", "Sales Ratio × Town", "Sales Ratio Normalized",
        "Month Sold", "Longitude", "Latitude", "Town", "Distance to City Center",
        "Property Type", "Day of Week Sold", "List Year", "Year Sold", "Residential Type",
        "Town Avg Sale Price", "Quarter Sold", "Days Since Listing"
    ]

    # Ensure new_entry has all required features (fill missing ones)
    for col in model_features:
        if col not in new_entry.columns:
            new_entry[col] = np.nan  # Fill missing with NaN
    
    # Keep only model-expected features
    input_data = new_entry[model_features]
    
    # Display processed input
    # st.write("Processed Input Data:")
    # st.write(input_data)
    
    # **PREDICTION BUTTON**
    if st.button("Predict"):
        missing_features = [col for col in model_features if col not in input_data.columns]
        if missing_features:
            st.error(f"Missing features in input data: {missing_features}")
        else:
            predicted_sale_amount = model.predict(input_data)[0]
            st.subheader("Predicted Sale Amount")
            st.write(f"$ {predicted_sale_amount:,.2f}")
