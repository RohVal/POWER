from typing import Any
from utils import load_model, split_data
from preprocessing import preprocess_data
import pandas as pd
from os import listdir, remove
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from igann import IGANN
from utils import load_model, evaluate_model
from sklearn.preprocessing import StandardScaler
from igann_model import scale_data

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
    """
    Remove all .png files from the directory.

    Args:
        directory: str: The directory to remove the .png files from.

    Returns:
        None
    """

    # remove all .png files from the directory
    for file in listdir(directory):
        if file.endswith(".png"):
            remove(join(directory, file))


def generate_feature_plot(feature: str, feature_val: float, shape_func: Any, y_val: float) -> Figure:
    """
    Generate a plot of the shape function for a given feature, with a marker at the input value. The plot will also
    include an annotation with the input value and the corresponding output value.

    Args:
        feature: str: The name of the feature.
        feature_val: float: The input value for the feature.
        shape_func: Any: The shape function for the feature.
        y_val: float: The output value for the feature.

    Returns:
        Figure: The plot of the shape function.
    """

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


def load_scalers() -> tuple:
    """
    Loads the feature and target scalers from the saved files.

    Returns:
        tuple: The feature and target scalers.
    """

    feature_scaler = load_model("feature_scaler.pkl")
    target_scaler = load_model("target_scaler.pkl")

    return feature_scaler, target_scaler


def make_prediction(clear_plots_dir: bool, features: dict[str, float], model: IGANN, shape_functions: Any) -> float:
    """
    Make a prediction using the IGANN model and the input features. The function will generate a plot for each feature
    and save it in the plots directory. The function will also return the prediction value.

    Args:
        clear_plots_dir: bool: Whether to clear the plots directory before generating the plots.
        features: dict[str, float]: The input features.
        model: IGANN: The IGANN model.
        shape_functions: Any: The shape functions for the features.

    Returns:
        float: The prediction value.
    """

    shape_function_values = []

    if clear_plots_dir:
        # Clear the plots directory
        clear_directory("../plots")

    for feature, value in features.items():
        # Find the shape function for the feature
        shape_func = next(x for x in shape_functions if x["name"] == feature)
        # Find the y-value for the input value using interpolation
        y_val = np.interp(value, shape_func["x"], shape_func["y"])
        # Add the y-value to the list
        shape_function_values.append(y_val)

        # generate the plot
        fig = generate_feature_plot(feature=feature, feature_val=value, shape_func=shape_func, y_val=y_val)

        # save the plot
        if FILENAMES.get(feature):
            fig.savefig(f"../plots/{FILENAMES[feature]}.png")
            continue

        fig.savefig(f"../plots/{feature.lower()}.png")

    return sum(shape_function_values) + model.init_classifier.intercept_


def scale_features(scaler: Any, features: pd.DataFrame) -> dict:
    """
    Scale the input features using the provided scaler.

    Args:
        scaler: Any: The scaler to use for scaling the features.
        features: pd.DataFrame: The input features.

    Returns:
        dict: The scaled features.
    """

    features_scaled = features.copy()
    features_scaled.iloc[:, :4] = scaler.transform(features_scaled.iloc[:, :4])
    features_as_dict = features_scaled.to_dict()
    features = dict(map(lambda item: (item[0], list(item[1].values())[0]), features_as_dict.items()))

    return features


def generate_better_plots(model: IGANN, features: dict[str, float], shape_functions: Any) -> list:
    """
    Generate a plot of the shape function for a given feature, with a marker at the input value. The plot will also
    include an annotation with the input value and the corresponding output value. This method is deprecated and
    should not be used.

    Args:
        model: IGANN: The IGANN model.
        features: dict[str, float]: The input features.
        shape_functions: Any: The shape functions for the features.

    Returns:
        list: The list of values for each feature.
    """

    values = []

    for feature, value in features.items():
        shape_func = next(x for x in shape_functions if x["name"] == feature)
        fig, ax = plt.subplots()
        sns.lineplot(x=shape_func["x"], y=shape_func["y"], ax=ax, linewidth=2, color="darkblue")

        # Find the y-value for the input value using interpolation
        y_val = np.interp(value, shape_func["x"], shape_func["y"])
        values.append(y_val)

        # Add a marker at the input value with annotation
        ax.plot(value, y_val, marker="s", markersize=8, color="black")
        ax.annotate(f"({value:.1f}, {y_val:.4f})", (value, y_val),
                    textcoords="offset points", xytext=(10, 10), ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(feature)
        ax.set_ylabel("")
        ax.set_title(f"{feature}:\n{shape_func['avg_effect']:.2f}%")
        ax.tick_params(axis='both', which='major', labelsize=10)
        sns.despine(left=True, bottom=True)

        # Add gridlines
        ax.grid(axis='both', linestyle='-', linewidth=0.5, color='lightgray')

    return values


if __name__ == "__main__":

    # load all pkl files
    model = load_model("igann.pkl")
    feature_scaler, target_scaler = load_scalers()
    xgboost = load_model("xgboost.pkl")

    # load dataset
    df = preprocess_data(
        path="../dataset/Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv")
    df = df.drop(columns=["Date and time"])
    X_train, X_test, y_train, y_test = split_data(df=df)
    first_entry = X_test.iloc[[2]]

    # scale the features
    features = scale_features(scaler=feature_scaler, features=first_entry)

    # get the shape functions
    shape_functions = model.get_shape_functions_as_dict()

    # make the prediction
    prediction = make_prediction(clear_plots_dir=True, features=features, model=model, shape_functions=shape_functions)

    # inverse transform the prediction
    prediction = target_scaler.inverse_transform([[prediction]])
    print(f"Predicted value: {prediction}")


    features_as_dict = pd.DataFrame(features, index=[0])

    prediction = model.predict(features_as_dict)

    # inverse transform the prediction
    prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))
    print(f"Predicted value: {prediction}")

    pred_xgboost = xgboost.predict(first_entry)
    print(f"Predicted value using XGBoost: {pred_xgboost}")
