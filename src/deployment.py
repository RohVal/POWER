import gradio as gr
from utils import load_model
import pandas as pd

model = load_model("xgboost.pkl")

CONTINUOUS_FEATURES = ["Wind speed (m/s)", "Wind speed - Maximum (m/s)", "Wind speed - Minimum (m/s)",
                       "Nacelle ambient temperature (°C)", "Wind direction (°)"]

PREDICTION_NAME = "Power (kW)"
LOGICAL_BOUNDS = {
    "Wind speed (m/s)": (0, 27),
    "Wind speed - Maximum (m/s)": (0, 27),
    "Wind speed - Minimum (m/s)": (0, 27),
    "Nacelle ambient temperature (°C)": (-20, 42),
    "Wind direction (°)": (0, 360)
}


def predict_power(wind_speed, wind_speed_max, wind_speed_min, nacelle_temp, wind_direction):
    data = {
        "Wind direction (°)": wind_direction,
        "Wind speed (m/s)": wind_speed,
        "Wind speed - Maximum (m/s)": wind_speed_max,
        "Wind speed - Minimum (m/s)": wind_speed_min,
        "Nacelle ambient temperature (°C)": nacelle_temp,
    }

    for feature in CONTINUOUS_FEATURES:
        if data[feature] < LOGICAL_BOUNDS[feature][0] or data[feature] > LOGICAL_BOUNDS[feature][1]:
            return "Invalid input"

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return prediction


# Create the interface
inputs = [gr.Slider(minimum=LOGICAL_BOUNDS[feature][0], maximum=LOGICAL_BOUNDS[feature]
                    [1], label=feature) for feature in CONTINUOUS_FEATURES]
outputs = gr.Textbox(label=PREDICTION_NAME)
gr.Interface(fn=predict_power, inputs=inputs, outputs=outputs).launch()
