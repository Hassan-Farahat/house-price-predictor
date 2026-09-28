import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(page_title="Real Estate Price Predictor", page_icon="🏡", layout="wide")


# Train ML Model on synthetic real estate data
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000

    sqft = np.random.randint(600, 4500, n)
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.randint(1, 5, n)
    age = np.random.randint(0, 40, n)
    location_score = np.random.randint(1, 10, n) # 1 = Rural, 10 = Prime City Center

    # Valuation formula: $150/sqft + $25k/bed + $35k/bath - $2k/year age + $20k/location score
    base_price = (sqft * 150) + (bedrooms * 25000) + (bathrooms * 35000) - (age * 2000) + (location_score * 20000)
    noise = np.random.normal(0, 15000, n)
    price = np.maximum(base_price + noise, 50000)

    X = pd.DataFrame({
        'Square Feet': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Property Age (Years)': age,
        'Location Rating (1-10)': location_score
    })

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, price)
    return model
model = train_model()

# Header
st.title("🏡 Real Estate Valuation Predictor")
st.markdown("Estimate residential property prices using Machine Learning.")


# Sidebar Inputs
st.sidebar.header("Property Features")
sqft = st.sidebar.slider("Square Footage", 600, 4500, 1800, step=50)
bedrooms = st.sidebar.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=2)
bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
age = st.sidebar.slider("Property Age (Years)", 0, 40, 10)
location_score = st.sidebar.slider("Location Rating (1 = Rural, 10 = Prime)", 1, 10, 7)


# Format Input
input_data = pd.DataFrame({
    'Square Feet': [sqft],
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms],
    'Property Age (Years)': [age],
    'Location Rating (1-10)': [location_score]
})


# Prediction
predicted_price = model.predict(input_data)[0]


# Display Results
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Estimated Market Value")
    st.metric("Predicted Price", f"${predicted_price:,.0f}")
    st.info("💡 Valuation is generated based on real-time comparative feature regression.")

with col2:
    st.subheader("Feature Weight Breakdown")
    importance_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)

    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)