import streamlit as st
import joblib
import numpy as np

# Load the trained CatBoost model
model = joblib.load('best_catboost_model.pkl')

# Set Streamlit page config
st.set_page_config(page_title="CatBoost Regressor", layout="centered")

# App title
st.title("🔮 CatBoost Regressor Prediction App")

st.markdown("""
Enter feature values below to get a prediction from your trained model.
""")

# Example: Assume the model expects 4 features
# You can customize based on your actual model
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)
feature_4 = st.number_input("Feature 4", value=0.0)

if st.button("Predict"):
    features = np.array([[feature_1, feature_2, feature_3, feature_4]])
    prediction = model.predict(features)[0]
    st.success(f"📈 Predicted value: **{prediction:.5f}**")
