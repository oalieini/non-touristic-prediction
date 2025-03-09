import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and scaler
rf_model = joblib.load("rf_model_amen10.joblib")  # Make sure this file exists in the same directory
scaler = joblib.load("scaler_amen10.joblib")      # Same for this file

# Define feature order (must match training order exactly!)
feature_order = [
    'price', 'host_listings_count', 'host_response_rate', 'host_acceptance_rate',
    'minimum_nights', 'maximum_nights', 'accomodate', 'TV', 'bath_tub', 'days_since_host'
]

# Apply custom CSS for improved styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f7f7;
        }
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #ff7777;
        }
        .prediction-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #e8f0fe;
            border-left: 5px solid #3b82f6;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App Title
st.title("🏠 Apartment Success Prediction")
st.write("Enter apartment details below to predict its success level.")

# Collect user inputs
user_input = {}

col1, col2 = st.columns(2)
with col1:
    user_input['price'] = st.number_input("💰 Price", min_value=0, step=1, value=120)
    user_input['host_listings_count'] = st.number_input("📦 Host Listings Count", min_value=0, step=1, value=5)
    user_input['host_response_rate'] = st.number_input("📬 Host Response Rate (%)", min_value=0, max_value=100, step=1, value=98)
    user_input['minimum_nights'] = st.number_input("🌙 Minimum Nights", min_value=1, step=1, value=2)
    user_input['accomodate'] = st.number_input("👥 Accommodate", min_value=1, step=1, value=4)

with col2:
    user_input['host_acceptance_rate'] = st.number_input("✅ Host Acceptance Rate (%)", min_value=0, max_value=100, step=1, value=99)
    user_input['maximum_nights'] = st.number_input("🏡 Maximum Nights", min_value=1, step=1, value=30)
    user_input['TV'] = st.selectbox("📺 TV Available", [0, 1])
    user_input['bath_tub'] = st.selectbox("🛁 Bath Tub Available", [0, 1])
    user_input['days_since_host'] = st.number_input("📆 Days Since Host Joined", min_value=0, step=1, value=500)

# Prediction logic
def predict_success(user_input):
    input_df = pd.DataFrame([user_input])
    input_df = input_df[feature_order]  # Enforce correct feature order
    normalized_input = scaler.transform(input_df)
    probabilities = rf_model.predict_proba(normalized_input)[0]
    class_labels = ["Unsuccessful", "Moderate Success", "Very Successful"]
    best_class_index = np.argmax(probabilities)
    prediction_label = class_labels[best_class_index]
    return prediction_label, probabilities

# Predict Button
if st.button("🔍 Predict Success"):
    prediction_label, probabilities = predict_success(user_input)

    # Display result
    st.markdown(f"""
    <div class="prediction-box">
        <h3>🔮 Prediction: <strong>{prediction_label}</strong></h3>
        <p><strong>📊 Success Probability Breakdown:</strong></p>
        <ul>
            <li>❌ Unsuccessful: {probabilities[0]:.2f}</li>
            <li>⚖️ Moderate Success: {probabilities[1]:.2f}</li>
            <li>✅ Very Successful: {probabilities[2]:.2f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
