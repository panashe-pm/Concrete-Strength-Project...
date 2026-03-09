import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the model and scaler
# These names MUST match the files you uploaded to GitHub exactly
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 2. Web App Interface
st.set_page_config(page_title="Concrete Strength AI", page_icon="🏗️")
st.title("🏗️ Concrete Strength Predictor")
st.markdown("Enter the mixture proportions below to predict the **Compressive Strength (MPa)**.")

st.sidebar.header("Input Parameters")

# 3. User Inputs (Sliding bars)
def user_input_features():
    cement = st.sidebar.slider('Cement (kg/m³)', 100.0, 550.0, 280.0)
    slag = st.sidebar.slider('Blast Furnace Slag (kg/m³)', 0.0, 360.0, 70.0)
    ash = st.sidebar.slider('Fly Ash (kg/m³)', 0.0, 200.0, 50.0)
    water = st.sidebar.slider('Water (kg/m³)', 120.0, 250.0, 180.0)
    superplasticizer = st.sidebar.slider('Superplasticizer (kg/m³)', 0.0, 32.0, 6.0)
    coarse_agg = st.sidebar.slider('Coarse Aggregate (kg/m³)', 700.0, 1150.0, 950.0)
    fine_agg = st.sidebar.slider('Fine Aggregate (kg/m³)', 500.0, 1000.0, 750.0)
    age = st.sidebar.slider('Age (Days)', 1, 365, 28)
    
    data = {
        'cement': cement,
        'slag': slag,
        'ash': ash,
        'water': water,
        'superplastic': superplasticizer,
        'coarse_agg': coarse_agg,
        'fine_agg': fine_agg,
        'age': age
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# Display the inputs to the user
st.subheader('Current Mix Design')
st.write(df)

# 4. Prediction Logic
if st.button('Predict Strength'):
    # Apply the same scaling used during training
    scaled_inputs = scaler.transform(df.values)
    
    # Run the prediction
    prediction = model.predict(scaled_inputs)
    
    # Show the result in a nice box

    st.success(f"### Predicted Compressive Strength: {prediction[0]:.2f} MPa")
