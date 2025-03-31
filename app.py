import streamlit as st
import pandas as pd
import numpy as np
import joblib
import geopy.distance
from geopy.geocoders import Nominatim
from datetime import datetime
from functools import lru_cache
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Cache expensive operations
@st.cache_resource
def load_model():
    model = joblib.load("stacked_model.pkl")
    return model

@st.cache_resource
def load_encoder():
    return joblib.load("target_encoder.pkl")

@st.cache_data
def load_historical_data():
    df = pd.read_csv("cleaned_sales_data.csv")
    # Pre-compute all possible features once with NaN handling
    df['Date Recorded'] = pd.to_datetime(df['Date Recorded'], errors='coerce')
    df["Day of Week Sold"] = df["Date Recorded"].dt.weekday.fillna(0)
    df["Quarter Sold"] = (df["Month Sold"].fillna(1) - 1) // 3 + 1
    current_year = datetime.today().year
    df["Days Since Listing"] = (current_year - df["Year Sold"].fillna(current_year)) * 365
    
    # Pre-compute town averages with NaN handling
    town_stats = df.groupby('Town').agg({
        'Sales Ratio': ['mean', 'std'],
        'Sale Amount': 'mean'
    }).reset_index()
    town_stats.columns = ['Town', 'Town_Sales_Ratio_Mean', 'Town_Sales_Ratio_Std', 'Town_Avg_Sale_Price']
    
    # Fill NaN values in town stats
    town_stats['Town_Sales_Ratio_Mean'] = town_stats['Town_Sales_Ratio_Mean'].fillna(df['Sales Ratio'].mean())
    town_stats['Town_Sales_Ratio_Std'] = town_stats['Town_Sales_Ratio_Std'].fillna(df['Sales Ratio'].std())
    town_stats['Town_Avg_Sale_Price'] = town_stats['Town_Avg_Sale_Price'].fillna(df['Sale Amount'].mean())
    
    return df, town_stats

def calculate_distance(lat, lon, center=(31.0689, -91.9968)):
    """Calculate distance in kilometers between two points on Earth with NaN handling"""
    try:
        if pd.isna(lat) or pd.isna(lon):
            return 0.0
        return geopy.distance.distance((lat, lon), center).km
    except Exception as e:
        st.error(f"Error calculating distance: {e}")
        return 0.0

@lru_cache(maxsize=1000)
def get_coordinates_cached(address):
    """Get coordinates from address with caching and error handling"""
    if not address:
        return (None, None)
    geolocator = Nominatim(user_agent="my_unique_app")
    try:
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
        return (None, None)
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return (None, None)

def prepare_input_data(new_entry, historical_df, town_stats):
    """Prepare input data for prediction with robust NaN handling"""
    # Ensure numeric columns are properly typed
    numeric_cols = ['Assessed Value', 'Sales Ratio', 'Year Sold', 'Month Sold']
    for col in numeric_cols:
        new_entry[col] = pd.to_numeric(new_entry[col], errors='coerce')
    
    # Merge with pre-computed town stats
    input_data = pd.merge(new_entry, town_stats, on='Town', how='left')
    
    # Calculate remaining features with NaN handling
    input_data['Sales Ratio Normalized'] = (
        (input_data['Sales Ratio'] - input_data['Town_Sales_Ratio_Mean']) / 
        input_data['Town_Sales_Ratio_Std'].replace(0, 1)  # Avoid division by zero
    ).fillna(0)
    
    input_data['Sales Ratio × Town'] = (
        input_data['Sales Ratio'].fillna(0) * input_data['Town_Avg_Sale_Price'].fillna(0)
    )
    
    # Calculate distance to city center
    input_data['Distance to City Center'] = input_data.apply(
        lambda row: calculate_distance(row['Latitude'], row['Longitude']), 
        axis=1
    )
    
    # Get market trend from cached data with NaN handling
    try:
        recent_sales = historical_df[
            (historical_df['Town'] == new_entry['Town'].iloc[0]) &
            (historical_df['Date Recorded'] >= (datetime.today() - pd.Timedelta(days=90)))
        ]
        market_trend = recent_sales['Sale Amount'].mean()
    except:
        market_trend = historical_df['Sale Amount'].mean()
    
    input_data['Market Trend Score'] = market_trend
    
    # Select final features for model
    final_features = [
        "Sales Ratio", "Assessed Value", "Market Trend Score", "Sales Ratio × Town", 
        "Sales Ratio Normalized", "Month Sold", "Longitude", "Latitude", "Town", 
        "Distance to City Center", "Property Type", "Day of Week Sold", "List Year", 
        "Year Sold", "Residential Type", "Town_Avg_Sale_Price", "Quarter Sold", 
        "Days Since Listing"
    ]
    
    # Ensure all columns exist and fill any remaining NaNs
    for col in final_features:
        if col not in input_data.columns:
            input_data[col] = 0.0
    
    # Convert all to numeric and fill NaNs
    input_data[final_features] = input_data[final_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return input_data[final_features]

# Initialize all cached resources at startup
model = load_model()
target_encoder = load_encoder()
historical_df, town_stats = load_historical_data()

# Streamlit UI (same as before)
st.title("Real Estate Sale Amount Prediction")
st.write("Enter details to predict the sale amount of a property.")

# Input method selection
method = st.radio("Choose how to provide location:", 
                 ["Enter Longitude & Latitude", "Enter Full Address"])

longitude, latitude = None, None

if method == "Enter Longitude & Latitude":
    longitude = st.number_input("Longitude", format="%.6f", value=0.0)
    latitude = st.number_input("Latitude", format="%.6f", value=0.0)

elif method == "Enter Full Address":
    address = st.text_input("Enter Full Address")
    if address:
        latitude, longitude = get_coordinates_cached(address)
        if latitude is not None and longitude is not None:
            st.success(f"Coordinates found: Latitude {latitude}, Longitude {longitude}")
        else:
            st.error("Could not find coordinates. Try refining the address.")

# User Inputs
assessed_value = st.number_input("Assessed Value", min_value=0.0, format="%.2f")
town = st.selectbox("Town", [
    "Andover", "Ansonia", "Ashford", "Avon", "Barkhamsted", "Beacon Falls", "Berlin", "Bethany", "Bethel", "Bethlehem",
    "Bloomfield", "Bolton", "Bozrah", "Branford", "Bridgeport", "Bridgewater", "Bristol", "Brookfield", "Brooklyn",
    "Burlington", "Canaan", "Canterbury", "Canton", "Chaplin", "Cheshire", "Chester", "Clinton", "Colchester",
    "Colebrook", "Columbia", "Cornwall", "Coventry", "Cromwell", "Danbury", "Darien", "Deep River", "Derby", "Durham",
    "East Granby", "East Haddam", "East Hampton", "East Hartford", "East Haven", "East Lyme", "East Windsor",
    "Eastford", "Ellington", "Enfield", "Essex", "Fairfield", "Farmington", "Franklin", "Glastonbury", "Goshen",
    "Granby", "Greenwich", "Griswold", "Groton", "Guilford", "Haddam", "Hamden", "Hampton", "Hartford", "Hartland",
    "Harwinton", "Hebron", "Kent", "Killingly", "Killingworth", "Lebanon", "Ledyard", "Lisbon", "Litchfield", "Lyme",
    "Madison", "Manchester", "Mansfield", "Marlborough", "Meriden", "Middlebury", "Middlefield", "Middletown",
    "Milford", "Monroe", "Montville", "Morris", "Naugatuck", "New Britain", "New Canaan", "New Fairfield",
    "New Hartford", "New Haven", "New London", "New Milford", "Newington", "Newtown", "Norfolk", "North Branford",
    "North Canaan", "North Haven", "North Stonington", "Norwalk", "Norwich", "Old Lyme", "Old Saybrook", "Orange",
    "Oxford", "Plainfield", "Plainville", "Plymouth", "Pomfret", "Portland", "Preston", "Prospect", "Putnam",
    "Redding", "Ridgefield", "Rocky Hill", "Roxbury", "Salem", "Salisbury", "Scotland", "Seymour", "Sharon",
    "Shelton", "Sherman", "Simsbury", "Somers", "South Windsor", "Southbury", "Southington", "Sprague", "Stafford",
    "Stamford", "Sterling", "Stonington", "Stratford", "Suffield", "Thomaston", "Thompson", "Tolland", "Torrington",
    "Trumbull", "Union", "Vernon", "Voluntown", "Wallingford", "Warren", "Washington", "Waterbury", "Waterford",
    "Watertown", "West Hartford", "West Haven", "Westbrook", "Weston", "Westport", "Wethersfield", "Willington",
    "Wilton", "Winchester", "Windham", "Windsor", "Windsor Locks", "Wolcott", "Woodbridge", "Woodbury"
])
property_type = st.selectbox("Property Type", [
    "Residential", "Single Family", "Condo", "Vacant Land", "Commercial", 
    "Two Family", "Three Family", "Apartments", "Industrial", "Four Family", 
    "Public Utility"
])
sales_ratio = st.number_input("Sales Ratio", min_value=0.0, format="%.4f")
residential_type = st.selectbox("Residential Type", [
    "Single Family", "Condo", "Two Family", "Three Family", "Four Family"
])
date_recorded = st.date_input("Date Recorded", datetime.today())

# Process input when location is available
if latitude is not None and longitude is not None:
    # Create new entry DataFrame
    new_entry = pd.DataFrame({
        "Assessed Value": [assessed_value],
        "Longitude": [longitude],
        "Latitude": [latitude],
        "Town": [town],
        "Property Type": [property_type],
        "Sales Ratio": [sales_ratio],
        "Residential Type": [residential_type],
        "Year Sold": [date_recorded.year],
        "Month Sold": [date_recorded.month],
        "Day of Week Sold": [date_recorded.weekday()],
        "Quarter Sold": [(date_recorded.month - 1) // 3 + 1],
        "Days Since Listing": [(datetime.today().year - date_recorded.year) * 365],
        "List Year": [date_recorded.year],
        "Date Recorded": [date_recorded]
    })
    
    # Prepare input data
    input_data = prepare_input_data(new_entry, historical_df, town_stats)
    
    # Encode categorical variables
    input_data[["Property Type", "Residential Type", "Town"]] = target_encoder.transform(
        input_data[["Property Type", "Residential Type", "Town"]]
    )
    
    # Display processed data
    st.write("Processed Input Data:")
    st.write(input_data)

# In the prediction section:
if st.button("Predict"):
    try:
        # Prepare input data
        input_data = prepare_input_data(new_entry, historical_df, town_stats)
        
        # Encode categorical variables
        cat_cols = ["Property Type", "Residential Type", "Town"]
        input_data[cat_cols] = target_encoder.transform(input_data[cat_cols])
        
        # Make prediction
        predicted_sale_amount = model.predict(input_data)[0]
        st.subheader("Predicted Sale Amount")
        st.write(f"$ {predicted_sale_amount:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
