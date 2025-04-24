import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# Load the CatBoost model
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

st.set_page_config(page_title="Product Rating Predictor", page_icon="📦")
st.title("📦 Product Rating Predictor")
st.write("Enter product details to predict the user rating.")

# Input form
with st.form("prediction_form"):
    color = st.selectbox("Color", ['Red', 'Blue', 'Black', 'White'])  # Adjust based on training
    product_name = st.text_input("Product Name")
    category = st.selectbox("Category", ['Shoes', 'Clothing', 'Accessories'])
    brand = st.selectbox("Brand", ['Nike', 'Adidas', 'Puma'])
    user_id = st.text_input("User ID")
    size = st.selectbox("Size", ['S', 'M', 'L', 'XL'])
    price = st.text_input("Price", value="100")  # ✅ as string

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
        'Price': price  # ✅ now a string
    }])

    prediction = model.predict(input_df)
    st.success(f"⭐ Predicted Rating: **{prediction[0]:.2f}**")
