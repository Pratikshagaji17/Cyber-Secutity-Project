from flask import Flask, render_template, request
import numpy as np
from joblib import load

app = Flask(__name__)
model = load("random_forest_model.joblib")  # Make sure this file is in the same directory

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # For now, just dummy input (replace with real form inputs)
        input_data = np.array([[1.0]*83])  # Your model expects 83 features
        
        prediction = model.predict(input_data)[0]
        return render_template("index.html", prediction=prediction)
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
