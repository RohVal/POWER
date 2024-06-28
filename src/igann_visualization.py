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
from utils import load_model


def load_scalers() -> tuple:
    feature_scaler = load_model("feature_scaler.pkl")
    target_scaler = load_model("target_scaler.pkl")

    return feature_scaler, target_scaler


def scale_features(scaler: any, features: pd.DataFrame) -> dict:
    features_scaled = features.copy()
    features_scaled.iloc[:, :4] = scaler.transform(features_scaled.iloc[:, :4])
    features_as_dict = features_scaled.to_dict()
    features = dict(map(lambda item: (item[0], list(item[1].values())[0]), features_as_dict.items()))

    return features


def generate_better_plots(model: IGANN, features: dict[str, float], shape_functions: any) -> list:

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

        # Customize plot appearance to match the reference
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)  # Horizontal line at y=0
        ax.set_xlabel(feature)  # X-axis label
        ax.set_ylabel("")   # Remove y-axis label
        ax.set_title(f"{feature}:\n{shape_func['avg_effect']:.2f}%")  # Add title
        ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust tick label size
        sns.despine(left=True, bottom=True)  # Remove top and right spines

        # Add gridlines
        ax.grid(axis='both', linestyle='-', linewidth=0.5, color='lightgray')

        # plt.show()

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

    # do the plotting
    values = generate_better_plots(model=model, features=features, shape_functions=shape_functions)


    features_as_dict = pd.DataFrame(features, index=[0])

    prediction = model.predict(features_as_dict)

    # inverse transform the prediction
    prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))
    print(f"Predicted value: {prediction}")

    pred_xgboost = xgboost.predict(first_entry)
    print(f"Predicted value using XGBoost: {pred_xgboost}")


    first_row = X_train
    print(first_row.describe())
