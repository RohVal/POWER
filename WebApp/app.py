import pandas as pd
from pickle import load
from flask import Flask, request, jsonify, render_template
from interpret import show, preserve

import os

# Get the absolute path of the current script
script_path = os.path.realpath(__file__)

# Get the directory of the current script
current_directory = os.path.dirname(script_path)

# Get the parent directory of the current directory
parent_directory = os.path.dirname(current_directory)

app = Flask(__name__)

# Using the same parameters and values as the gradio model 
ebm_path = os.path.join(parent_directory, 'models', 'ebm.pkl')
xgboost_path = os.path.join(parent_directory, 'models', 'xgboost.pkl')
lstm_path = os.path.join(parent_directory, 'models', 'lstm.pkl')
# igann_path = os.path.join(parent_directory, 'models', 'igann.pkl')

with open(ebm_path, "rb") as f: 
    model_ebm = load(f)
with open(xgboost_path, "rb") as f:
    model_xgboost = load(f)
# with open(r"/WebApp/models/ebm_model.pkl", "rb") as f:
#     model_ebm = load(f)

# Constants
CONTINUOUS_FEATURES = ["Wind speed (m/s)", "Wind speed - Maximum (m/s)", "Wind speed - Minimum (m/s)",
                       "Nacelle ambient temperature (°C)"]

WIND_DIRECTION_FEATURES = [ "E/NE", "E/SE", "N", "N/NE", "N/NW", "NE", "NW", "S", "S/SE", "S/SW", "SE", "SW", "W", "W/NW", "W/SW"]

PREDICTION_NAME = "Power (kW)(generated in one hour)"
LOGICAL_BOUNDS = {
    "Wind speed (m/s)": (0, 27),
    "Wind speed - Maximum (m/s)": (0, 27),
    "Wind speed - Minimum (m/s)": (0, 27),
    "Nacelle ambient temperature (°C)": (-20, 42)
}

def predict_power(model, wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction):

    if model != model_ebm:
        data = {
            "Wind speed (m/s)": wind_speed,
            "Wind speed - Maximum (m/s)": wind_speed_max,
            "Wind speed - Minimum (m/s)": wind_speed_min,
            "Nacelle ambient temperature (°C)": nacelle_temp,
        }

        for direction in WIND_DIRECTION_FEATURES:
            data[direction] = 0
        if wind_direction != 'E':
            data[wind_direction] = 1    

        for feature in CONTINUOUS_FEATURES:
            if data[feature] < LOGICAL_BOUNDS[feature][0] or data[feature] > LOGICAL_BOUNDS[feature][1]:
                return "Invalid input"

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return prediction
    
    else : 
        data = {
            "Wind speed (m/s)": wind_speed,
            "Wind speed - Maximum (m/s)": wind_speed_max,
            "Wind speed - Minimum (m/s)": wind_speed_min,
            "Nacelle ambient temperature (°C)": nacelle_temp,
            "WindDirection": wind_direction
        }
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        # preserve(model.explain_global(), file_name = "xyz.html")
        # preserve(show(model.explain_global()), file_name = "shobc")
        return prediction




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():

    # wind_speed = int(request.form.get("wspeed"))
    # wind_speed_max = int(request.form.get("wspeed_max"))
    # wind_speed_min = int(request.form.get("wspeed_min"))
    # nacelle_temp = int(request.form.get("ntemp"))
    # wind_direction = (request.form.get("wind_direction"))

    if request.method == "POST":

        wind_speed = int(request.form.get("wspeed"))
        wind_speed_max = int(request.form.get("wspeed_max"))
        wind_speed_min = int(request.form.get("wspeed_min"))
        nacelle_temp = int(request.form.get("ntemp"))
        wind_direction = (request.form.get("wind_direction"))
        model_type = request.form.get("model")
        
        if model_type == 'XGBoost':
            model = model_xgboost
        elif model_type == 'EBM':
            model = model_ebm
        elif model_type == 'All':
            models = {
                'XGBoost': model_xgboost,
                'EBM': model_ebm
                # 'IGANN' : model_igann
                # 'LSTM' : model_lstm
            }
            predictions = {}
            for name, model in models.items():
                prediction = predict_power(model, wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction)
                predictions[name] = prediction
          
            return render_template('predict.html', result = predictions, model = 'all')
        else:
            return jsonify({"error": "Invalid model type"}), 400
            
        

        prediction = predict_power(model, wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction)

        if model_type == 'EBM':
            return render_template('predict.html', result = prediction, model = 'ebm' )
        
        return render_template('predict.html', result = prediction)
    return render_template('predict.html')    

if __name__ == '__main__':
    app.run(debug=True)
