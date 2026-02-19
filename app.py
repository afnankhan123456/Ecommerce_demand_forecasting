from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("demand_forecast_v1.keras")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

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
        result = model.predict(arr)
        prediction = float(result[0][0])

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()
