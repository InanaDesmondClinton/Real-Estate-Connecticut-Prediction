import streamlit as st
import pandas as pd
import numpy as np
import joblib
import geopy.distance
from geopy.geocoders import Nominatim
from datetime import datetime

# Load trained model and encoder
model = joblib.load("CatBoost.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Define city center for distance calculation
city_center = (31.0689, -91.9968)  # Example: Louisiana

# Function to calculate distance to city center
def calculate_distance(lat, lon, center=city_center):
    return geopy.distance.distance((lat, lon), center).km

# Function to get coordinates from an address
def get_coordinates(address):
    geolocator = Nominatim(user_agent="my_unique_app")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    return None, None

# Streamlit UI
st.title("Real Estate Sale Amount Prediction")
st.write("Enter details to predict the sale amount of a property.")

# Input method selection
method = st.radio("Choose how to provide location:", ["Enter Longitude & Latitude", "Enter Full Address"])

longitude, latitude = None, None  # Initialize variables

if method == "Enter Longitude & Latitude":
    longitude = st.number_input("Longitude", format="%.6f", value=0.0)
    latitude = st.number_input("Latitude", format="%.6f", value=0.0)

elif method == "Enter Full Address":
    address = st.text_input("Enter Full Address (e.g., '1600 Amphitheatre Parkway, Mountain View, CA, USA')")
    if address:
        latitude, longitude = get_coordinates(address)
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
]
)
property_type = st.selectbox("Property Type", ["Residential", "Single Family", "Condo", "Vacant Land", 
                                               "Commercial", "Two Family", "Three Family", "Apartments", "Industrial", "Four Family", "Public Utility"])
sales_ratio = st.number_input("Sales Ratio", min_value=0.0, format="%.4f")
residential_type = st.selectbox("Residential Type", ["Single Family", "Condo", "Two Family", "Three Family", "Four Family"])
date_recorded = st.date_input("Date Recorded", datetime.today())

# Feature Engineering
if latitude is not None and longitude is not None:
    #st.write(latitude, longitude, date_recorded)
    year_sold = date_recorded.year
    month_sold = date_recorded.month
    day_of_week_sold = date_recorded.weekday()
    quarter_sold = (month_sold - 1) // 3 + 1
    days_since_listing = (datetime.today().year - year_sold) * 365
    #weekend_sale = 1 if day_of_week_sold >= 5 else 0
    distance_to_city_center = calculate_distance(latitude, longitude)
    #st.write(distance_to_city_center, day_of_week_sold, quarter_sold, days_since_listing)

    # Load historical data (Ensure sales_data.csv exists)
    try:
        df_sales_history = pd.read_csv("cleaned_sales_data.csv")
        #df_sales_history = df_sales_history[["Date Recorded", "Town", "Assessed Value", "Sales Ratio", "Property Type", "Residential Type", "Longitude",
        #                             "Latitude", "Year Sold", "Month Sold", "Sale Amount"]]

        # Append new input data to history for processing
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
            #"Weekend Sale": [weekend_sale],
            "Distance to City Center": [distance_to_city_center]
        })
        expected_columns = [
            "Serial Number", "List Year", "Date Recorded", "Town", "Address", "Assessed Value", 
            "Sale Amount", "Sales Ratio", "Property Type", "Residential Type", "Location", "State", 
            "Longitude", "Latitude", "Year Sold", "Month Sold"
        ]

        # Ensure new_entry has all required columns (fill missing with default values)
        for col in expected_columns:
            if col not in new_entry.columns:
                new_entry[col] = np.nan  # or use a meaningful default

        # Ensure column order matches
        new_entry = new_entry[expected_columns]
        # Concatenate properly
        df_sales_history = pd.concat([df_sales_history, new_entry], ignore_index=True)
        #df_sales_history = pd.concat([df_sales_history, new_entry], ignore_index=True)



        # Ensure data is sorted before rolling operations
        df_sales_history = df_sales_history.sort_values(["Town", "Year Sold", "Month Sold"])
        # Convert date columns to datetime
        df_sales_history['Date Recorded'] = pd.to_datetime(df_sales_history['Date Recorded'], errors='coerce')
        df_sales_history["Day of Week Sold"] = df_sales_history["Date Recorded"].apply(lambda x: x.weekday() if pd.notna(x) else np.nan)

        df_sales_history["Quarter Sold"] = df_sales_history["Month Sold"].apply(lambda x: (x - 1) // 3 + 1 if pd.notna(x) else np.nan)
        df_sales_history["Days Since Listing"] = (datetime.today().year - df_sales_history["Year Sold"]) * 365

        # Compute Distance (Ensure lat/lon are not missing)
        df_sales_history["Distance to City Center"] = df_sales_history.apply(
            lambda row: calculate_distance(row["Latitude"], row["Longitude"]) 
            if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]) else np.nan, axis=1
        )

        # Compute additional features
        df_sales_history['Sales Ratio Normalized'] = df_sales_history.groupby('Town')['Sales Ratio'].transform(lambda x: (x - x.mean()) / x.std()).fillna(0)
        df_sales_history['Town Avg Sale Price'] = df_sales_history.groupby('Town')['Sale Amount'].transform('mean')
        df_sales_history['Market Trend Score'] = df_sales_history.groupby('Town')['Sale Amount'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df_sales_history['Sales Ratio × Town'] = df_sales_history['Sales Ratio'] * df_sales_history['Town Avg Sale Price']

        # Get the last processed row (new input)
        input_data = df_sales_history.iloc[-1:].copy()

    except FileNotFoundError:
        st.error("Historical sales data (sales_data.csv) not found. Ensure it's available.")

    # Encode categorical variables
    input_data[["Property Type", "Residential Type", "Town"]] = target_encoder.transform(
        input_data[["Property Type", "Residential Type", "Town"]]
    )

    # Select final input features (Ensure model expects these)
    final_features = [
        "Sales Ratio", "Assessed Value", "Market Trend Score", "Sales Ratio × Town", "Sales Ratio Normalized",
        "Month Sold", "Longitude", "Latitude", "Town", "Distance to City Center",
        "Property Type", "Day of Week Sold", "List Year", "Year Sold", "Residential Type",
        "Town Avg Sale Price", "Quarter Sold", "Days Since Listing"
    ]
    
    # Ensure final input columns match model expectations
    input_data = input_data[final_features]
    # Display the input data
    st.write("Processed Input Data:")
    st.write(input_data)

    # **PREDICTION BUTTON**
    if st.button("Predict"):
        # Make prediction
        predicted_sale_amount = model.predict(input_data)[0]

        # Display result
        st.subheader("Predicted Sale Amount")
        st.write(f"$ {predicted_sale_amount:,.2f}")

else:
    st.warning("Please enter a valid location (Longitude & Latitude or Address) to proceed.")
