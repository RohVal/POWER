import gradio as gr
from utils import load_model
import pandas as pd

model_xgboost = load_model("xgboost.pkl")
model_grid = load_model("xgboost-grid.pkl")

CONTINUOUS_FEATURES = ["Wind speed (m/s)", "Wind speed - Maximum (m/s)", "Wind speed - Minimum (m/s)",
                       "Nacelle ambient temperature (°C)"]

WIND_DIRECTION_FEATURES = ["E/NE", "E/SE", "N", "N/NE", "N/NW", "NE", "NW", "S", "S/SE", "S/SW", "SE", "SW", "W", "W/NW", "W/SW"]

PREDICTION_NAME = "Power (kW)"

LOGICAL_BOUNDS = {
    "Wind speed (m/s)": (0, 27),
    "Wind speed - Maximum (m/s)": (0, 27),
    "Wind speed - Minimum (m/s)": (0, 27),
    "Nacelle ambient temperature (°C)": (-20, 42)
}

def predict_power(wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction):
    data = {
        "Wind speed (m/s)": wind_speed,
        "Wind speed - Maximum (m/s)": wind_speed_max,
        "Wind speed - Minimum (m/s)": wind_speed_min,
        "Nacelle ambient temperature (°C)": nacelle_temp,
    }

    # setting the value for all directions in the matrix to 0
    for direction in WIND_DIRECTION_FEATURES:
        data[direction] = 0

    # Changing the value of the selected direction to 1
    data[wind_direction] = 1

    for feature in CONTINUOUS_FEATURES:
        if data[feature] < LOGICAL_BOUNDS[feature][0] or data[feature] > LOGICAL_BOUNDS[feature][1]:
            return "Invalid input"

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return prediction

# Create the interface
inputs = [
    gr.Slider(minimum=LOGICAL_BOUNDS[feature][0], maximum=LOGICAL_BOUNDS[feature][1], label=feature)
    for feature in CONTINUOUS_FEATURES
]

# Directions in dropdown list
inputs.append(gr.Dropdown(choices=WIND_DIRECTION_FEATURES, label="Wind direction"))

## Directions in radio buttons format
# inputs.append(gr.Radio(choices=WIND_DIRECTION_FEATURES, label="Wind direction"))

outputs = gr.Textbox(label=PREDICTION_NAME)

## Some example Inputs to check the 
examples = [
    [7.26, 9.6, 5.28, 12.56, 'S'],
    [7.92, 9.54, 5.53, 13.19, 'S/SW'],
    [12, 15, 10, 25, 'N'],
    [8, 10, 6, 18, 'E/NE'],
    [20, 22, 18, 30, 'S'],
    [5, 7, 4, 10, 'W'],
]

# # Launch the interface- original 
# gr.Interface(fn=predict_power, inputs=inputs, outputs=outputs, examples=examples).launch()

# Tabs for different models
with gr.Blocks() as demo:
    with gr.Tab(label="Grid Model"):
        gr.Interface(fn=lambda wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction: predict_power(model_grid, wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction), 
                     inputs=inputs, outputs=outputs, examples=examples)
    with gr.Tab(label="XGBoost Model"):
        gr.Interface(fn=lambda wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction: predict_power(model_xgboost, wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction), 
                     inputs=inputs, outputs=outputs, examples=examples)

# Launch the interface
demo.launch()