📦 Product Rating Predictor
Predict a user's product rating based on product details using machine learning!

🚀 Overview
This project is a Streamlit web app that predicts the likely user rating for a product based on inputs like:
User ID
Product Name
Brand
Category
Price
Color
Size

Built with:
Python
CatBoost Regressor (for prediction modeling)
Streamlit (for interactive web app)
Pandas (for data handling)
🌟 Live Demo

👉 Click here to try the app!
(https://app-kvhehbbmkzeuwzd6vqkkpq.streamlit.app)

🛠️ Installation
To run this app locally:

1. Clone this repository:
git clone https://github.com/your-username/product-rating-predictor.git
cd product-rating-predictor

2. Install the dependencies:
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run app.py
📂 Repository Structure

├── app.py                # Streamlit app script
├── catboost_model.cbm     # Pre-trained CatBoost model
├── requirements.txt       # Python dependencies
└── README.md              # Project description
📈 Model

The machine learning model is trained using CatBoost Regressor, a gradient boosting algorithm that works well with categorical features and missing data.
During prediction, numerical and categorical inputs are properly handled to avoid mismatches.

✨ Features
Clean and user-friendly UI with Streamlit
Handles both numerical and categorical product features
Fast and responsive predictions
Deployed live on Streamlit Cloud

🔥 Future Improvements
Improve model accuracy with more data
Enable batch predictions for multiple products at once
📬 Contact

If you have any questions or feedback, feel free to reach out to me here on LinkedIn!
