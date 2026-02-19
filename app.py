from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib

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

    if request.method == "POST":
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

        # 🔥 SCALE INPUT (VERY IMPORTANT)
        arr_scaled = feature_scaler.transform(arr)

        result = model.predict(arr_scaled)
        prediction = float(result[0][0])

        # 🔥 Business Logic Recommendation
        if prediction > 3000:
            demand_status = "High Demand"
            recommendation = """
            • Increase inventory stock
            • Increase ad budget
            • Monitor supply chain to avoid stock-out
            """

        elif prediction > 1500:
            demand_status = "Moderate Demand"
            recommendation = """
            • Maintain current inventory levels
            • Monitor competitor pricing
            • Optimize discount strategy
            """

        else:
            demand_status = "Low Demand"
            recommendation = """
            • Increase discount offers
            • Improve marketing campaigns
            • Re-evaluate pricing strategy
            """

    return render_template(
        "index.html",
        prediction=prediction,
        recommendation=recommendation,
        demand_status=demand_status
    )

if __name__ == "__main__":
    app.run()
