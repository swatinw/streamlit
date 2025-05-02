
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import os

st.set_page_config(page_title="Product Rating Predictor", page_icon="📦")
st.title("📦 Product Rating Predictor")
st.write("Enter product details to predict the user rating.")

# Load the CatBoost model with error handling
model_path = "catboost_model.cbm"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Make sure it's uploaded.")
    st.stop()

try:
    model = CatBoostRegressor()
    model.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input form
with st.form("prediction_form"):
    color = st.selectbox("Color", ['Red', 'Blue', 'Black', 'White'])
    product_name = st.text_input("Product Name")
    category = st.selectbox("Category", ['Shoes', 'Clothing', 'Accessories'])
    brand = st.selectbox("Brand", ['Nike', 'Adidas', 'Puma'])
    user_id = st.text_input("User ID")
    size = st.selectbox("Size", ['S', 'M', 'L', 'XL'])
    price = st.text_input("Price", value="100")

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    try:
        input_df = pd.DataFrame([{
            'Color': color,
            'Product Name': product_name,
            'Category': category,
            'Brand': brand,
            'User ID': user_id,
            'Size': size,
            'Price': price
        }])

        # Ensure column order matches training
        input_df = input_df[['Color', 'Product Name', 'Category', 'Brand', 'User ID', 'Size', 'Price']]

        prediction = model.predict(input_df)
        st.success(f"⭐ Predicted Rating: **{prediction[0]:.2f}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
