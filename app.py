from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__, static_folder=".")
CORS(app)  # still enabled (safe)

df = None
model = None
scaler = None
label_encoders = {}
feature_columns = []
trained = False


# ===================== LOAD DATA =====================
def load_data():
    global df
    path = os.path.join(os.path.dirname(__file__), "retail_store_inventory.csv")
    df = pd.read_csv(path)
    print("Dataset loaded:", df.shape)


# ===================== PREPROCESS =====================
def preprocess():
    global df, model, scaler, label_encoders, feature_columns, trained

    data = df.copy()

    # DROP useless columns
    data.drop(["Store ID", "Product ID"], axis=1, inplace=True)

    # DATE FEATURES
    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Quarter"] = data["Date"].dt.quarter
    data.drop("Date", axis=1, inplace=True)

    # 🔴 REMOVE leakage column (IMPORTANT FIX)
    if "Units Sold" in data.columns:
        data.drop("Units Sold", axis=1, inplace=True)

    # ENCODE categorical
    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    y = data["Demand Forecast"]
    X = data.drop("Demand Forecast", axis=1)

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, max_depth=20)
    model.fit(X_train, y_train)

    trained = True
    print("Model trained ✅")


# ===================== FEATURE PREP =====================
def prepare_input(data):
    input_df = pd.DataFrame([data])

    for col in label_encoders:
        if col in input_df:
            input_df[col] = label_encoders[col].transform(input_df[col])
        else:
            input_df[col] = 0

    for col in feature_columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)

    return input_scaled


# ===================== ROUTES =====================

@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "trained": trained})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = prepare_input(data)
        prediction = model.predict(features)[0]

        return jsonify({
            "success": True,
            "predicted_demand": round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================== RUN =====================
if __name__ == "__main__":
    load_data()
    preprocess()
    app.run(debug=True, port=5000)