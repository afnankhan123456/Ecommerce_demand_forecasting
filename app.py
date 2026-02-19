from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("demand_forecast_v1.keras")

@app.route("/")
def home():
    return "Ecommerce Demand Forecast Model is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    
    arr = np.array(data).reshape(1, -1)
    prediction = model.predict(arr)
    
    return jsonify({
        "prediction": float(prediction[0][0])
    })

if __name__ == "__main__":
    app.run()
