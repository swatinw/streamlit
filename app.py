import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# Load the trained CatBoost model
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# App config
st.set_page_config(page_title="Product Rating Predictor", page_icon="📦")
st.title("📦 Product Rating Predictor")
st.write("Enter product details to predict the user rating.")

# Input form
with st.form("prediction_form"):
    color = st.selectbox("Color", ['Red', 'Blue', 'Black', 'White'])  # Update with your actual categories
    product_name = st.text_input("Product Name")
    category = st.selectbox("Category", ['Shoes', 'Clothing', 'Accessories'])  # Customize as needed
    brand = st.selectbox("Brand", ['Nike', 'Adidas', 'Puma'])  # Adjust to your training data
    user_id = st.text_input("User ID")
    size = st.selectbox("Size", ['S', 'M', 'L', 'XL'])  # Match what your model expects
    price = st.number_input("Price", min_value=0.0, max_value=10000.0, value=100.0)  # ✅ Float input

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_df = pd.DataFrame([{
        'Color': color,
        'Product Name': product_name,
        'Category': category,
        'Brand': brand,
        'User ID': user_id,
        'Size': size,
        'Price': price  # ✅ Keep as float
    }])

    prediction = model.predict(input_df)
    st.success(f"⭐ Predicted Rating: **{prediction[0]:.2f}**")
