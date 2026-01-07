import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import requests
import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# ==========================================
# 1. API KEYS & CONFIGURATION
# ==========================================

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)


# 2. LOAD ML ARTIFACTS

@st.cache_resource
def load_ml_assets():
    BASE_DIR = r"D:\major project final year\src\components\artifacts"
    try:
        model = joblib.load(os.path.join(BASE_DIR, "aqi_lr_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "lr_scaler.pkl"))
        features = joblib.load(os.path.join(BASE_DIR, "lr_feature_columns.pkl"))
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model_lr, scaler_lr, features_list = load_ml_assets()

# 3. TOOL DEFINITION
def get_aqi_prediction(city: str):
    """Fetches real-time weather/pollution data and uses the Linear Regression model for AQI prediction."""
    try:
        # 1. Geocoding
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_data = requests.get(geo_url).json()
        if not geo_data: return f"Error: City '{city}' not found."
        lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]

        # 2. Live Data Retrieval
        pol_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        met_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        
        pol_res = requests.get(pol_url).json()['list'][0]['components']
        met_res = requests.get(met_url).json()

        # 3. Feature Engineering
        now = datetime.datetime.now()
        data = {
            "pm2_5": pol_res['pm2_5'], "pm10": pol_res['pm10'], "no2": pol_res['no2'],
            "so2": pol_res['so2'], "co": pol_res['co'], "o3": pol_res['o3'],
            "temperature": met_res['main']['temp'], "humidity": met_res['main']['humidity'],
            "wind_speed": met_res['wind']['speed'], "rainfall": met_res.get('rain', {}).get('1h', 0),
            "hour_sin": np.sin(2 * np.pi * now.hour / 24), "hour_cos": np.cos(2 * np.pi * now.hour / 24),
            "dow_sin": np.sin(2 * np.pi * now.weekday() / 7), "dow_cos": np.cos(2 * np.pi * now.weekday() / 7),
            "month_sin": np.sin(2 * np.pi * now.month / 12), "month_cos": np.cos(2 * np.pi * now.month / 12)
        }

        # 4. Inference
        input_df = pd.DataFrame([data])[features_list]
        input_scaled = scaler_lr.transform(input_df)
        pred = model_lr.predict(input_scaled)[0]

        return {
            "city": city, 
            "predicted_aqi": round(float(pred), 2),
            "pollutant_breakdown": pol_res, 
            "current_temp": f"{data['temperature']}°C",
            "condition": met_res['weather'][0]['description']
        }
    except Exception as e:
        return f"Error: {str(e)}"

# 4. CHAT INTERFACE LOGIC (WRITTEN FOR 2026)

st.set_page_config(page_title="AQI Forecast AI", page_icon="🌬️")
st.title("🌬️ AQI Forecasting AI Dashboard")
st.caption("2026 Edition | Gemini 2.5 Flash")

# Consolidated Initialization
if "chat" not in st.session_state:
    # Use 'gemini-2.5-flash' (Stable) or 'gemini-3-flash-preview' (Agentic)
    MODEL_ID = 'gemini-2.5-flash' 
    
    instruction = (
        "You are an AQI expert. Use the 'get_aqi_prediction' tool to fetch real-time data "
        "and predict AQI for a city. Analyze the pollutants and provide health advice "
        "based on CPCB India categories."
    )
    
    try:
        gemini_model = genai.GenerativeModel(
            model_name=MODEL_ID,
            tools=[get_aqi_prediction],
            system_instruction=instruction
        )
        # Start chat with automatic tool calling enabled
        st.session_state.chat = gemini_model.start_chat(enable_automatic_function_calling=True)
        st.session_state.messages = []
    except Exception as e:
        st.error(f"Model Initialization Failed: {e}")
        st.stop()

# 1. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle User Input
if prompt := st.chat_input("What is the air quality in Mumbai?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        try:
            # The model will automatically call get_aqi_prediction() if a city is named 
            response = st.session_state.chat.send_message(prompt)
            full_response = response.text
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Chat Error: {e}")