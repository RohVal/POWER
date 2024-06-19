import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

## Using the same parameters and values as the gradio model 
# # Load models
# with open(r"C:\Users\rexxt\POWER\models\ebm_model.pkl", "rb") as f: 
#     model_grid = pickle.load(f)
# with open(r"C:\Users\rexxt\POWER\models\ebm_model.pkl", "rb") as f:
#     model_xgboost = pickle.load(f)
# with open(r"C:\Users\rexxt\POWER\models\ebm_model.pkl", "rb") as f:
#     model_ebm = pickle.load(f)

# Constants
CONTINUOUS_FEATURES = ["Wind speed (m/s)", "Wind speed - Maximum (m/s)", "Wind speed - Minimum (m/s)",
                       "Nacelle ambient temperature (°C)"]

WIND_DIRECTION_FEATURES = ["E/NE", "E/SE", "N", "N/NE", "N/NW", "NE", "NW", "S", "S/SE", "S/SW", "SE", "SW", "W", "W/NW", "W/SW"]

PREDICTION_NAME = "Power (kW)(generated in one hour)"
LOGICAL_BOUNDS = {
    "Wind speed (m/s)": (0, 27),
    "Wind speed - Maximum (m/s)": (0, 27),
    "Wind speed - Minimum (m/s)": (0, 27),
    "Nacelle ambient temperature (°C)": (-20, 42)
}

def predict_power(model, wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction):
    data = {
        "Wind speed (m/s)": wind_speed,
        "Wind speed - Maximum (m/s)": wind_speed_max,
        "Wind speed - Minimum (m/s)": wind_speed_min,
        "Nacelle ambient temperature (°C)": nacelle_temp,
    }

    for direction in WIND_DIRECTION_FEATURES:
        data[direction] = 0
    data[wind_direction] = 1

    for feature in CONTINUOUS_FEATURES:
        if data[feature] < LOGICAL_BOUNDS[feature][0] or data[feature] > LOGICAL_BOUNDS[feature][1]:
            return "Invalid input"

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')
    # if model_type == 'Grid':
    #     model = model_grid
    # elif model_type == 'XGBoost':
    #     model = model_xgboost
    # elif model_type == 'EBM':
    #     model = model_ebm
    # else:
    #     return jsonify({"error": "Invalid model type"}), 400

    # prediction = predict_power(model, wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction)
    # return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
