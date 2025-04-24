import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# Load the CatBoost model
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

st.set_page_config(page_title="Product Rating Predictor", page_icon="📦")
st.title("📦 Product Rating Predictor")

st.markdown("Provide product details to predict the user rating.")

# Input form
with st.form("prediction_form"):
    color = st.selectbox("Color", ['Red', 'Blue', 'Black', 'White'])  # Update with actual categories
    product_name = st.text_input("Product Name")
    category = st.selectbox("Category", ['Shoes', 'Clothing', 'Accessories'])  # Update as needed
    brand = st.selectbox("Brand", ['Nike', 'Adidas', 'Puma'])  # Update with your data
    user_id = st.text_input("User ID")  # User ID used as category during training
    size = st.selectbox("Size", ['S', 'M', 'L', 'XL'])  # Adjust to your data
    price = str(st.number_input("Price", min_value=0.0, max_value=10000.0, value=100.0))

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        'Color': color,
        'Product Name': product_name,
        'Category': category,
        'Brand': brand,
        'User ID': user_id,
        'Size': size,
        'Price': price
    }])

    prediction = model.predict(input_df)
    st.success(f"⭐ Predicted Rating: **{prediction[0]:.2f}**")
