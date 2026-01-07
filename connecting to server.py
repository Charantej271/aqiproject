from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.pyfunc

# ================= MLFLOW CONFIG =================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
MODEL_NAME = "AQI_Best_Model"

print("Loading best model from MLflow Registry...")
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
    print("Best model loaded successfully")
except Exception as e:
    raise RuntimeError("Failed to load model from MLflow Registry") from e


# ================= PATHS =================
BASE_DIR = r"D:\major project final year\src\components"
DATASET_PATH = r"D:\major project final year\notebook\aqi_cleaned_processed.csv"

FEATURE_PATH = os.path.join(BASE_DIR, "artifacts", "lr_feature_columns.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "artifacts", "lr_scaler.pkl")

REQUIRED_FILES = [FEATURE_PATH, SCALER_PATH, DATASET_PATH]
for f in REQUIRED_FILES:
    if not os.path.exists(f):
        raise RuntimeError(f"Missing required file: {f}")



# ================= LOAD DATA =================
df = pd.read_csv(DATASET_PATH)
df = pd.read_csv(DATASET_PATH, low_memory=False)

# -------- CLEAN DATE --------
df["date"] = df["date"].replace(
    [999999, "999999", "", "na", "null", "unknown"],
    np.nan
)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# -------- CLEAN HOUR --------
df["hour"] = pd.to_numeric(df["hour"], errors="coerce")

# keep valid hours only
df.loc[(df["hour"] < 0) | (df["hour"] > 23), "hour"] = np.nan

# convert hour to integer safely
df["hour"] = df["hour"].fillna(df["hour"].mode()[0]).astype(int)

# -------- SAFE DATETIME CREATION --------
df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

# drop invalid rows
df = df.dropna(subset=["datetime"])


feature_columns = joblib.load(FEATURE_PATH)
scaler = joblib.load(SCALER_PATH)



# ================= APP =================
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "AQI Prediction API is running"})



# ================= FEATURE EXTRACTION =================
def extract_features(city, dt):
    city_df = df[df["city"] == city]

    hour_avg = city_df[city_df["hour"] == dt.hour].mean(numeric_only=True)
    if hour_avg.isnull().any():
        hour_avg = city_df.mean(numeric_only=True)

    dow = dt.dayofweek
    month = dt.month

    model_features = {
        "pm2_5": hour_avg["pm2_5"],
        "pm10": hour_avg["pm10"],
        "no2": hour_avg["no2"],
        "so2": hour_avg["so2"],
        "co": hour_avg["co"],
        "o3": hour_avg["o3"],
        "temperature": hour_avg["temperature"],
        "humidity": hour_avg["humidity"],
        "wind_speed": hour_avg["wind_speed"],
        "rainfall": hour_avg["rainfall"],
        "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }

    display_features = {
        "pm2_5": round(float(hour_avg["pm2_5"]), 2),
        "pm10": round(float(hour_avg["pm10"]), 2),
        "no2": round(float(hour_avg["no2"]), 2),
        "temperature": round(float(hour_avg["temperature"]), 2),
        "humidity": round(float(hour_avg["humidity"]), 2),
    }

    return model_features, display_features




# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    city = data["city"]
    start_dt = pd.to_datetime(data["start_date"])
    hours = int(data["hours_ahead"])

    future_dates = pd.date_range(start_dt, periods=hours, freq="h")
    result = []
    prev_aqi = None

    for dt in future_dates:
        feats, disp = extract_features(city, dt)

        if prev_aqi is not None:
            feats["pm2_5"] *= np.clip(prev_aqi / 100, 0.7, 1.3)

        X = pd.DataFrame([feats]).reindex(columns=feature_columns, fill_value=0)
        X_scaled = scaler.transform(X)

        aqi_pred = model.predict(pd.DataFrame(X_scaled, columns=feature_columns))[0]
        prev_aqi = aqi_pred

        result.append({
            "datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "city": city,
            "predicted_aqi": round(float(aqi_pred), 2),
            **disp
        })

    return jsonify({"predictions": result})



# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
