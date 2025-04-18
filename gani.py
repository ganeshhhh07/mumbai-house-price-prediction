import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("C://Users//polab//Downloads//mumbai_house_price_final.csv")

# Encode categorical features
categorical_cols = ['property_type', 'New/Resale', 'market_trend', 'property_category', 'furnishing_status',
                    'Gymnasium', 'Lift', 'Parking', 'Security', 'Clubhouse', 'Swimming Pool']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features & target
X = data[['area', 'bedrooms', 'bathrooms', 'balcony_num', 'property_age', 'total_floors',
          'lat', 'lon', 'crime_rate', 'aqi', 'market_trend', 'neighborhood_rating', 'property_type',
          'New/Resale', 'property_category', 'furnishing_status', 'Gymnasium', 'Lift', 'Parking',
          'Security', 'Clubhouse', 'Swimming Pool']]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
lgbm = LGBMRegressor(n_estimators=100, random_state=42)

models = {'RandomForest': rf, 'XGBoost': xgb, 'LightGBM': lgbm}
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'{name}_model.pkl')

# Streamlit UI
st.set_page_config(page_title="Mumbai House Price Prediction", layout="wide")
st.title("\U0001F3E0 Mumbai House Price Prediction")

st.sidebar.header("\U0001F4CA Enter Property Details")
area = st.sidebar.number_input("\U0001F4CF Area (sq ft)", min_value=100, max_value=5000, value=800)
bedrooms = st.sidebar.selectbox("\U0001F6CF No of Bedrooms", [1, 2, 3, 4])
bathrooms = st.sidebar.selectbox("\U0001F6BF No of Bathrooms", [1, 2, 3, 4])
balcony = st.sidebar.selectbox("\U0001F3E2 No of Balconies", [0, 1, 2, 3])
property_age = st.sidebar.number_input("\U0001F3DB Property Age (years)", min_value=0, max_value=50, value=5)
total_floors = st.sidebar.number_input("\U0001F3E2 Total Floors", min_value=1, max_value=50, value=10)
property_type = st.sidebar.selectbox("\U0001F3E1 Property Type", label_encoders['property_type'].classes_)
new_resale = st.sidebar.selectbox("\U0001F3E0 New/Resale", label_encoders['New/Resale'].classes_)
property_category = st.sidebar.selectbox("\U0001F3DA Property Category", label_encoders['property_category'].classes_)
furnishing_status = st.sidebar.selectbox("\U0001F4BC Furnishing Status", label_encoders['furnishing_status'].classes_)
crime_rate = st.sidebar.slider("\U0001F46E Crime Rate (1-10)", 1, 10, 5)
aqi = st.sidebar.slider("\U0001F32B AQI (50-300)", 50, 300, 100)
market_trend = st.sidebar.selectbox("\U0001F4C8 Market Trend", label_encoders['market_trend'].classes_)
neighborhood_rating = st.sidebar.slider("\U0001F3D9 Neighborhood Rating", 1, 5, 3)

location = st.sidebar.selectbox("\U0001F3D9 Location", data['location'].unique())

# Fetch lat/lon based on location
location_data = data[data['location'] == location].iloc[0]
lat, lon = location_data['lat'], location_data['lon']

gymnasium = st.sidebar.selectbox("🏋 Gymnasium", label_encoders['Gymnasium'].classes_)
lift = st.sidebar.selectbox("🏢 Lift", label_encoders['Lift'].classes_)
parking = st.sidebar.selectbox("🚗 Parking", label_encoders['Parking'].classes_)
security = st.sidebar.selectbox("🚔 Security", label_encoders['Security'].classes_)
clubhouse = st.sidebar.selectbox("🏛 Clubhouse", label_encoders['Clubhouse'].classes_)
swimming_pool = st.sidebar.selectbox("🏊 Swimming Pool", label_encoders['Swimming Pool'].classes_)

# Convert user input
input_data = pd.DataFrame([[area, bedrooms, bathrooms, balcony, property_age, total_floors, lat, lon,
                            crime_rate, aqi, market_trend, neighborhood_rating, property_type, new_resale,
                            property_category, furnishing_status, gymnasium, lift, parking, security,
                            clubhouse, swimming_pool]], columns=X.columns)

for col in categorical_cols:
    input_data[col] = label_encoders[col].transform(input_data[col])

# Prediction
model_choice = st.sidebar.selectbox("📊 Choose Model", ['RandomForest', 'XGBoost', 'LightGBM'])
model = joblib.load(f'{model_choice}_model.pkl')
predicted_price = model.predict(input_data)[0]
st.write(f"### 💰 Predicted Price: ₹{predicted_price:,.2f}")

# Visualization: Price Distribution
st.subheader("📊 Price Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(y=data['price'], ax=ax)
st.pyplot(fig)

# Visualization: Heatmap
st.subheader("🗺️ Geographical Heatmap")
map_ = folium.Map(location=[19.0760, 72.8777], zoom_start=11)
HeatMap(data[['lat', 'lon', 'price']].values, radius=15).add_to(map_)
folium_static(map_)

# Visualization: Time Series
st.subheader("📈 Market Trends Over Time")
fig = px.line(data, x='property_age', y='price', title='Price Trends Over Property Age')
st.plotly_chart(fig)

st.write("### 📐 Price vs Area")
fig = px.scatter(data, x='area', y='price', title="Price vs Area", trendline="ols")
st.plotly_chart(fig)

st.write("### 💰 Price Distribution")
fig, ax = plt.subplots()
sns.histplot(data['price'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Ensure only numeric columns are used
numeric_data = data.select_dtypes(include=['number'])

st.write("### 📊 Correlation Heatmap")

# Set figure size
fig, ax = plt.subplots(figsize=(12, 8))

# Generate the heatmap with better styling
sns.heatmap(
    numeric_data.corr(), 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f', 
    linewidths=0.5,  # Adds spacing between cells
    square=True,  
    annot_kws={"size": 12, "color": "black", "weight": "bold"},  # Increase font size, set color & bold text
    cbar_kws={"shrink": 0.75},  
    ax=ax
)

# Improve label readability
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Add a clear title
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')

st.pyplot(fig)