# Project Title: Predictive Modeling for Heart Attack Diagnosis
# Author: Rutuja Keshav Jadhav
# Guide: Gemini (Google)
# Date: August 20, 2025

# This Python script implements a machine learning pipeline to predict heart attack diagnosis
# based on various patient attributes. The project follows standard data science practices
# including data loading, preprocessing, exploratory data analysis, feature selection,
# model training, hyperparameter tuning, and comprehensive evaluation.

# --- Streamlit is imported for the web interface ---
import streamlit as st

# --- 1. Import Necessary Libraries ---
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import joblib 

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Remove Streamlit UI elements ---
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
)

st.markdown("""
<style>
.st-emotion-cache-18ni7ap {
    display: none;
}
.st-emotion-cache-h5h06g {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.title("Predictive Modeling for Heart Attack Diagnosis")
st.markdown("This model predicts the risk of a heart attack based on patient data.")

# --- 2. Load the Dataset (for training) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('heart.csv')
        return df
    except FileNotFoundError:
        st.error("ERROR: 'heart.csv' not found. Please ensure the dataset file is in the same directory as this script.")
        st.stop()

df = load_data()


# --- 3. Data Preprocessing & Model Training (from your original script) ---
@st.cache_resource
def train_and_save_model():
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Split data and apply preprocessing
    df['target'] = df['condition'].map({0: 0, 1: 1})
    X = df.drop(['target', 'condition'], axis=1)
    y = df['target']
    X_transformed = preprocessor.fit_transform(X, y)
    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_transformed, columns=feature_names)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=15)
    X_selected = selector.fit_transform(X_processed, y)
    selected_feature_names = X_processed.columns[selector.get_support(indices=True)]
    
    # Train the final model
    best_rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
    best_rf_model.fit(X_selected, y)

    # Save models for deployment (Streamlit automatically loads them from the same directory)
    joblib.dump(best_rf_model, 'best_rf_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(selector, 'feature_selector.pkl')
    joblib.dump(selected_feature_names, 'selected_feature_names.pkl')
    
    return "Model trained and saved."

train_and_save_model()

# Now, load the saved models for the prediction function
loaded_preprocessor = joblib.load('preprocessor.pkl')
loaded_selector = joblib.load('feature_selector.pkl')
loaded_model = joblib.load('best_rf_model.pkl')
loaded_selected_feature_names = joblib.load('selected_feature_names.pkl')

# --- 4. User Input-Based Prediction System with Streamlit UI ---

st.subheader("Patient Health Details")
st.markdown("Please enter the following health details to get a risk assessment.")

with st.form(key='user_input_form'):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (in years):", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex:", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type:", options=[0, 1, 2, 3], format_func=lambda x: {0:"Typical Angina", 1:"Atypical Angina", 2:"Non-anginal Pain", 3:"Asymptomatic"}[x])
        trestbps = st.number_input("Resting Blood Pressure (trestbps in mm/Hg):", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (chol in mg/dl):", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl:", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG Results:", options=[0, 1, 2], format_func=lambda x: {0:"Normal", 1:"ST-T wave abn", 2:"LV hypertrophy"}[x])

    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved:", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina:", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression:", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment:", options=[0, 1, 2], format_func=lambda x: {0:"Upsloping", 1:"Flat", 2:"Downsloping"}[x])
        ca = st.number_input("Number of Major Vessels (0-3):", min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassemia:", options=[0, 1, 2, 3], format_func=lambda x: {0:"NULL", 1:"normal", 2:"fixed defect", 3:"reversible defect"}[x])
        
    submit_button = st.form_submit_button(label='Predict Heart Attack Risk')

# --- Prediction Logic and Output Display ---
if submit_button:
    user_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }
    
    user_df = pd.DataFrame([user_data])
    
    # Preprocess the user input using the loaded objects
    user_transformed = loaded_preprocessor.transform(user_df)
    processed_feature_names = loaded_preprocessor.get_feature_names_out()
    user_processed_df = pd.DataFrame(user_transformed, columns=processed_feature_names)
    user_selected = user_processed_df[loaded_selected_feature_names]

    # Make prediction
    prediction_proba = loaded_model.predict_proba(user_selected)[0]
    risk_probability = prediction_proba[1]
    risk_percentage = risk_probability * 100

    # Display results with different colors based on risk
    st.subheader("Prediction Result")
    
    if risk_percentage < 30:
        st.success(f"You have a **{risk_percentage:.0f}% chance** of heart attack – **Low Risk**. Keep up your healthy lifestyle! 🎉")
    elif risk_percentage < 70:
        st.warning(f"You have a **{risk_percentage:.0f}% chance** of heart attack – **Moderate Risk**. It's advisable to **Consult a Doctor** for further evaluation.")
    else:
        st.error(f"You have a **{risk_percentage:.0f}% chance** of heart attack – **High Risk**. Please **Consult a Doctor immediately** for a comprehensive medical assessment. Your health is important! 🚨")

st.markdown("---")
st.info("Disclaimer: This is a predictive model based on historical data and should not replace professional medical advice. Always consult a qualified healthcare professional for any health concerns.")
