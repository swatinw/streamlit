import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# Load the CatBoost model
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# Streamlit page config
st.set_page_config(page_title="Product Rating Predictor", page_icon="📦")
st.title("📦 Product Rating Predictor")
st.markdown("Predict the ⭐ rating a user would give to a product based on product details.")

# Input form
with st.form("prediction_form"):
    user_id = st.text_input("🆔 User ID")
    product_name = st.text_input("📦 Product Name")
    brand = st.text_input("🏷️ Brand")
    category = st.text_input("🗂️ Category")
    price = st.number_input("💲 Price", min_value=0.0, step=1.0)
    color = st.text_input("🎨 Color")
    size = st.text_input("📏 Size")
    
    submit = st.form_submit_button("Predict 🚀")

if submit:
    with st.spinner('Predicting... 🧠'):
        # Create a dataframe from user input
        input_data = pd.DataFrame({
            'User ID': [user_id],
            'Product Name': [product_name],
            'Brand': [brand],
            'Category': [category],
            'Price': [price],
            'Color': [color],
            'Size': [size]
        })

        # Define categorical features (excluding Price)
        cat_features = ['User ID', 'Product Name', 'Brand', 'Category', 'Color', 'Size']

        # Make prediction
        prediction = model.predict(input_data)
        
        st.success(f"⭐ Predicted Rating: **{prediction[0]:.2f}**")
