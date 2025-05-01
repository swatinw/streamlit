import streamlit as st
from catboost import CatBoostRegressor
import pandas as pd

# Load the model safely
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm", format="cbm")  # Explicitly set format
    return model

model = load_model()

st.title("CatBoost Regression Predictor")

# Input fields (adjust to your features)
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[feature_1, feature_2, feature_3]],
                            columns=["feature_1", "feature_2", "feature_3"])
    prediction = model.predict(input_df)
    st.success(f"Prediction: {prediction[0]:.2f}")
