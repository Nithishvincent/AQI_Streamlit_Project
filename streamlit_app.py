import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AQI Prediction System", layout="wide")

DATA_PATH = "data/aqi_dataset.csv"
MODEL_PATH = "model/aqi_model.pkl"

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ---------------- FUNCTIONS ----------------
def generate_dataset(n=2000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "PM2.5": np.random.uniform(0, 200, n),
        "PM10": np.random.uniform(0, 300, n),
        "NO2": np.random.uniform(0, 150, n),
        "SO2": np.random.uniform(0, 80, n),
        "CO": np.random.uniform(0, 8, n),
        "O3": np.random.uniform(0, 180, n),
        "Temperature": np.random.uniform(-10, 45, n),
        "Humidity": np.random.uniform(20, 95, n),
    })

    df["AQI"] = (
        df["PM2.5"] * 0.9 +
        df["PM10"] * 0.45 +
        df["NO2"] * 0.35 +
        df["SO2"] * 0.25 +
        df["CO"] * 8.0 +
        df["O3"] * 0.5 -
        df["Temperature"] * 0.25 -
        df["Humidity"] * 0.12 +
        np.random.normal(0, 10, n)
    ).clip(0, 500)

    df.to_csv(DATA_PATH, index=False)


def train_model():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("AQI", axis=1)
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_s, y_train)

    joblib.dump(
        {"model": model, "scaler": scaler, "features": list(X.columns)},
        MODEL_PATH
    )

    y_pred = model.predict(X_test_s)

    return {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }



def aqi_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy (Sensitive)"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

# ---------------- UI ----------------
st.title("🌍 Air Quality Index Prediction System")
st.caption("Dataset-driven ML model with Streamlit dashboard")

# -------- DATASET --------
st.header("1️⃣ Dataset")

if not os.path.exists(DATA_PATH):
    st.warning("Dataset not found.")
    if st.button("Generate Dataset"):
        generate_dataset()
        st.success("Dataset created successfully.")
        st.experimental_rerun()
else:
    st.success("Dataset found.")
    df = pd.read_csv(DATA_PATH)
    st.dataframe(df.head())

# -------- TRAINING --------
st.header("2️⃣ Model Training")

if not os.path.exists(MODEL_PATH):
    st.info("Model not trained yet.")
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            metrics = train_model()
        st.success("Model trained and saved.")
        st.json(metrics)
        st.experimental_rerun()
else:
    st.success("Trained model found.")

# -------- PREDICTION --------
if os.path.exists(DATA_PATH) and os.path.exists(MODEL_PATH):
    st.header("3️⃣ AQI Prediction")

    data = joblib.load(MODEL_PATH)
    model = data["model"]
    scaler = data["scaler"]
    features = data["features"]

    inputs = {}
    cols = st.columns(2)
    for i, f in enumerate(features):
        with cols[i % 2]:
            inputs[f] = st.number_input(f, value=float(df[f].median()))

    if st.button("Predict AQI"):
        X_input = scaler.transform(np.array(list(inputs.values())).reshape(1, -1))
        pred = model.predict(X_input)[0]

        st.metric("Predicted AQI", f"{pred:.2f}")
        st.success(aqi_category(pred))

    # -------- METRICS --------
    st.header("4️⃣ Model Performance")

    X_all = scaler.transform(df[features])
    y_all = df["AQI"]
    y_pred_all = model.predict(X_all)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE", f"{mean_squared_error(y_all, y_pred_all):.2f}")
    c2.metric("MAE", f"{mean_absolute_error(y_all, y_pred_all):.2f}")
    c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_all, y_pred_all)):.2f}")
    c4.metric("R²", f"{r2_score(y_all, y_pred_all):.2f}")

    # -------- SCATTER --------
    st.header("5️⃣ Actual vs Predicted AQI")

    fig, ax = plt.subplots()
    ax.scatter(y_all, y_pred_all, alpha=0.5)
    ax.plot([0, 500], [0, 500], linestyle="--")
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    st.pyplot(fig)

    # -------- FEATURE IMPORTANCE --------
    st.header("6️⃣ Feature Importance")

    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=features)
        st.bar_chart(fi.sort_values(ascending=False))
