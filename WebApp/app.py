import pandas as pd
from pickle import load
from flask import Flask, request, jsonify, render_template
from interpret import show, preserve
import numpy as np
from igann import IGANN
from sklearn.preprocessing import StandardScaler
from igann_helper import transform_features, scale_features, make_prediction

# from src.igann_visualization import *
import tempfile
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
igann_path = os.path.join(parent_directory, 'models', 'igann.pkl')
fscaler_path = os.path.join(parent_directory, 'models', 'feature_scaler.pkl')
tscaler_path = os.path.join(parent_directory, 'models', 'target_scaler.pkl')

with open(ebm_path, "rb") as f:
    model_ebm = load(f)
with open(xgboost_path, "rb") as f:
    model_xgboost = load(f)
with open(igann_path, "rb") as f:
    model_igann = load(f)
with open(fscaler_path, "rb") as f:
    feature_scaler = load(f)
with open(tscaler_path, "rb") as f:
    target_scaler = load(f)


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


def predict_igann(model: IGANN, feature_scaler: StandardScaler, target_scaler: StandardScaler, features: dict) -> str:
    """
    Predict the power using the IGANN model. The prediction is the inverse transformed value. The shape functions will
    be saved in a directory.

    Args:
        model (IGANN): the IGANN model
        feature_scaler (StandardScaler): the feature scaler
        target_scaler (StandardScaler): the target scaler
        features (dict): the features

    Returns:
        float: the power prediction
    """

    # transform the features into a DataFrame
    feature_df = transform_features(features)

    # scale the features
    scaled_features = scale_features(feature_scaler, feature_df)

    # get the shape functions
    shape_functions = model.get_shape_functions_as_dict()

    # make the prediction
    prediction = make_prediction(clear_plots_dir=True, features=scaled_features, model=model, shape_functions=shape_functions)

    # inverse transform the prediction
    prediction = target_scaler.inverse_transform([[prediction]])

    # return prediction as string
    return f"{prediction[0][0]:.2f}"


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

        if model != model_igann:

            prediction = model.predict(df)[0]
            return prediction
        else :
            # FIX later
            prediction = model.predict(df)[0]

            # inverse transform the prediction
            prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))
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
        return prediction


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():

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
        elif model_type == 'IGANN':
            features = {
                "wind_speed": wind_speed,
                "wind_speed_max": wind_speed_max,
                "wind_speed_min": wind_speed_min,
                "nacelle_temp": nacelle_temp,
                "wind_direction": wind_direction
            }

            prediction = predict_igann(model=model_igann, feature_scaler=feature_scaler, target_scaler=target_scaler, features=features)

            return render_template('predict.html', result=prediction, model='igann')

        elif model_type == 'All':
            models = {
                'XGBoost': model_xgboost,
                'EBM': model_ebm,
            }
            predictions = {}
            for name, model in models.items():
                prediction = predict_power(model, wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction)
                predictions[name] = prediction

            # make the prediction using the IGANN model
            features = {
                "wind_speed": wind_speed,
                "wind_speed_max": wind_speed_max,
                "wind_speed_min": wind_speed_min,
                "nacelle_temp": nacelle_temp,
                "wind_direction": wind_direction
            }

            prediction = predict_igann(model=model_igann, feature_scaler=feature_scaler, target_scaler=target_scaler, features=features)
            predictions['IGANN'] = prediction

            return render_template('predict.html', result = predictions, model = 'all')
        else:
            return jsonify({"error": "Invalid model type"}), 400



        prediction = predict_power(model, wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction)

        if model_type == 'EBM':
            return render_template('predict.html', result = prediction, model = 'ebm' )

        return render_template('predict.html', result = prediction)
    return render_template('predict.html')

@app.route('/explain', methods=['GET'])
def explain():

    explanation = model_ebm.explain_global()
    print("length = " + str(len(explanation.data())))

    plot_htmls = []
    for i in range(5):  # just the feature graphs
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
            preserve(explanation, selector_key=i, file_name=temp_file.name)
            plot_path = temp_file.name

        # read HTML file
        with open(plot_path, 'r') as file:
            plot_html = file.read()

        # Append to the list
        plot_htmls.append(plot_html)

        # Remove the temporary file
        os.remove(plot_path)

    return render_template('explain.html', plot_htmls=plot_htmls)


@app.route('/local', methods = ['GET','POST'])
def elocal():
    if request.method == "POST":

        wind_speed = int(request.form.get("wspeed"))
        wind_speed_max = int(request.form.get("wspeed_max"))
        wind_speed_min = int(request.form.get("wspeed_min"))
        nacelle_temp = int(request.form.get("ntemp"))
        wind_direction = (request.form.get("wind_direction"))

        # input sample

        arr = {
        "Wind speed (m/s)": wind_speed,
        "Wind speed - Maximum (m/s)": wind_speed_max,
        "Wind speed - Minimum (m/s)": wind_speed_min,
        "Nacelle ambient temperature (°C)": nacelle_temp,
        "WindDirection": wind_direction}

        predict = predict_power(model_ebm, wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction)
        X_sample = np.array([[wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction]])
        # local explanation
        local_explanation = model_ebm.explain_local(X_sample)
        print(local_explanation)

        #  preserve to save the plot to an HTML file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
            preserve(local_explanation, 0,file_name=temp_file.name)
            plot_path = temp_file.name

        #preserve(local_explanation, file_name="ghj.html")

        # Read the contents of the HTML file
        with open(plot_path, 'r') as file:
            plot_html = file.read()

        # Remove the temporary file
        os.remove(plot_path)
        return render_template('local.html', plot_html = plot_html, result = True, pred = predict)

    return render_template('local.html' )

if __name__ == '__main__':
    app.run(debug=True)
