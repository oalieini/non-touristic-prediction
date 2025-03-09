# 🏠 Non-Touristic Apartment Success Prediction

This project provides a **Streamlit-based web application** to predict the success of non-touristic apartment listings. It leverages a **Random Forest model** to estimate the likelihood of an apartment being **Unsuccessful, Moderately Successful, or Very Successful** based on key features.


## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/oalieini/non-touristic-prediction.git
cd non-touristic-prediction
```

### 2️⃣ Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv env
source env/bin/activate  # For Windows: env\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have **Streamlit** installed:

```bash
pip install streamlit
```

### 4️⃣ Ensure Model Files Exist

The application depends on the following trained model files, so extract them from zip files:

- `rf_model_5.joblib`
- `scaler_5_features.joblib`

## ▶️ Running the Application

launch the Streamlit app:

```bash
streamlit run app.py
```


## 📌 Input Features
You can chenge the features in test.ipynb and reload the model.
The model uses the following **5 key features**:

- `price`
- `number_of_reviews_ltm`
- `number_of_reviews`
- `host_acceptance_rate`
- `host_total_listings_count`

