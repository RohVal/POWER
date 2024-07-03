import matplotlib
matplotlib.use('Agg')

import pandas as pd
from typing import Any
from os import listdir, remove
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from igann import IGANN
import numpy as np
from typing import Dict

FILENAMES = {
    "Wind speed (m/s)": "wind_speed",
    "Wind speed - Maximum (m/s)": "wind_speed_max",
    "Wind speed - Minimum (m/s)": "wind_speed_min",
    "Nacelle ambient temperature (°C)": "temperature",
    "N/NE": "n_ne",
    "E/NE": "e_ne",
    "E/SE": "e_se",
    "S/SE": "s_se",
    "S/SW": "s_sw",
    "W/SW": "w_sw",
    "W/NW": "w_nw",
    "N/NW": "n_nw",
}


def clear_directory(directory: str) -> None:
    # remove all .png files from the directory
    for file in listdir(directory):
        if file.endswith(".png"):
            remove(join(directory, file))


def generate_feature_plot(feature: str, feature_val: float, shape_func: Any, y_val: float) -> Figure:
    fig, ax = plt.subplots()
    sns.lineplot(x=shape_func["x"], y=shape_func["y"], ax=ax, linewidth=2, color="darkblue")

    # Add a marker at the input value with annotation
    ax.plot(feature_val, y_val, marker="s", markersize=8, color="black")
    ax.annotate(f"({feature_val:.2f}, {y_val:.4f})", (feature_val, y_val),
                textcoords="offset points", xytext=(10, 10), ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # Customize plot appearance to match the reference
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)  # Horizontal line at y=0
    ax.set_xlabel(feature)  # X-axis label
    ax.set_ylabel("")   # Remove y-axis label
    ax.set_title(f"{feature}:\n{shape_func['avg_effect']:.2f}%")  # Add title
    ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust tick label size
    sns.despine(left=True, bottom=True)  # Remove top and right spines

    # Add gridlines
    ax.grid(axis='both', linestyle='-', linewidth=0.5, color='lightgray')

    return fig


def make_prediction(clear_plots_dir: bool, features: Dict[str, float], model: IGANN, shape_functions: Any) -> float:

    shape_function_values = []

    if clear_plots_dir:
        # Clear the plots directory
        clear_directory("./static/plots")

    for feature, value in features.items():
        # Find the shape function for the feature
        shape_func = next(filter(lambda x: x["name"] == feature, shape_functions), None)
        if shape_func is None:
            continue
        # Find the y-value for the input value using interpolation
        y_val = np.interp(value, shape_func["x"], shape_func["y"])
        # Add the y-value to the list
        shape_function_values.append(y_val)

        # generate the plot
        fig = generate_feature_plot(feature=feature, feature_val=value, shape_func=shape_func, y_val=y_val)

        # Save the plot
        filename = FILENAMES.get(feature, feature.lower())
        fig.savefig(f"./static/plots/{filename}.png")
        plt.close(fig)

    return sum(shape_function_values) + model.init_classifier.intercept_


def scale_features(scaler: Any, features: pd.DataFrame) -> dict:
    features_scaled = features.copy()
    features_scaled.iloc[:, :4] = scaler.transform(features_scaled.iloc[:, :4])
    features_as_dict = features_scaled.to_dict()
    features = dict(map(lambda item: (item[0], list(item[1].values())[0]), features_as_dict.items()))

    return features


def transform_features(features: dict) -> pd.DataFrame:
    wind_directions = ["E/NE", "E/SE", "N", "N/NE", "N/NW", "NE", "NW", "S", "S/SE", "S/SW", "SE", "SW", "W", "W/NW", "W/SW"]
    data = {
        "Wind speed (m/s)": features["wind_speed"],
        "Wind speed - Maximum (m/s)": features["wind_speed_max"],
        "Wind speed - Minimum (m/s)": features["wind_speed_min"],
        "Nacelle ambient temperature (°C)": features["nacelle_temp"],
    }

    for direction in wind_directions:
        if direction != features["wind_direction"]:
            data[direction] = 0
            continue
        data[direction] = 1

    return pd.DataFrame([data])
