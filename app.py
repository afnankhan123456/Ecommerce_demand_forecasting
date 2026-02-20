from flask import Flask, request, render_template, send_file
import numpy as np
import tensorflow as tf
import joblib
import os
import pandas as pd
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("demand_forecast_v1.keras")

# Load scaler
feature_scaler = joblib.load("feature_scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    recommendation = None
    demand_status = None
    batch_processed = False

    if request.method == "POST":

        # 🔥 ---- SINGLE INPUT PREDICTION (Existing logic preserved) ----
        if "price" in request.form:

            features = [
                float(request.form["price"]),
                float(request.form["discount_percent"]),
                float(request.form["ad_spend"]),
                float(request.form["page_views"]),
                float(request.form["cart_additions"]),
                float(request.form["avg_session_time"]),
                float(request.form["competitor_price"]),
                float(request.form["seasonality_index"]),
                float(request.form["day_of_week"])
            ]

            arr = np.array(features).reshape(1, -1)

            arr_scaled = feature_scaler.transform(arr)

            result = model.predict(arr_scaled)
            prediction = round(float(result[0][0]), 2)

            if prediction > 3000:
                demand_status = "High Demand"
                recommendation = [
                    "Increase inventory stock",
                    "Increase advertising budget",
                    "Monitor supply chain to prevent stock-outs"
                ]

            elif prediction > 1500:
                demand_status = "Moderate Demand"
                recommendation = [
                    "Maintain current inventory levels",
                    "Monitor competitor pricing",
                    "Optimize discount strategy"
                ]

            else:
                demand_status = "Low Demand"
                recommendation = [
                    "Increase discount offers",
                    "Improve marketing campaigns",
                    "Re-evaluate pricing strategy"
                ]

        # 🔥 ---- BATCH FILE PREDICTION (New Feature Added) ----
        if "file" in request.files:

            file = request.files["file"]

            if file.filename != "":
                df = pd.read_csv(file)

                required_columns = [
                    "price",
                    "discount_percent",
                    "ad_spend",
                    "page_views",
                    "cart_additions",
                    "avg_session_time",
                    "competitor_price",
                    "seasonality_index",
                    "day_of_week"
                ]

                X = df[required_columns]

                X_scaled = feature_scaler.transform(X)

                predictions = model.predict(X_scaled)
                predictions = np.round(predictions.flatten(), 2)

                df["predicted_units_sold"] = predictions

                output = io.StringIO()
                df.to_csv(output, index=False)
                output.seek(0)

                return send_file(
                    io.BytesIO(output.getvalue().encode()),
                    mimetype="text/csv",
                    as_attachment=True,
                    download_name="predicted_output.csv"
                )

    return render_template(
        "index.html",
        prediction=prediction,
        recommendation=recommendation,
        demand_status=demand_status
    )


# ✅ RENDER SAFE START
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
