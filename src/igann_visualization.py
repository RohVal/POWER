from utils import load_model, split_data
from preprocessing import preprocess_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from igann import IGANN


def generate_better_plots(model: IGANN, features: dict[str, float]) -> None:
    shape_func_list = model.get_shape_functions_as_dict()

    for feature, value in features.items():
        shape_func = next(x for x in shape_func_list if x["name"] == feature)
        sns.set(style="whitegrid")
        fig, ax = plt.subplots()
        sns.lineplot(x=shape_func["x"], y=shape_func["y"], ax=ax, linewidth=2, color="darkblue")

        y_val = np.interp(value, shape_func["x"], shape_func["y"])

        # Add a marker at the input value with annotation
        ax.plot(value, y_val, marker="s", markersize=8, color="black")
        ax.annotate(f"({value:.1f}, {y_val:.4f})", (value, y_val),
                    textcoords="offset points", xytext=(10, 10), ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))


        # Customize plot appearance to match the reference
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)  # Horizontal line at y=0
        ax.set_xlabel("x")  # X-axis label
        ax.set_ylabel("")   # Remove y-axis label
        ax.set_title(f"{feature}:\n{shape_func['avg_effect']:.2f}%")  # Add title
        ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust tick label size
        sns.despine(left=True, bottom=True)  # Remove top and right spines

        # Add gridlines
        ax.grid(axis='both', linestyle='-', linewidth=0.5, color='lightgray')

        plt.show()


# def magic_stuff(model: any, scaler: any)->None:
#     shape_func_list = model.get_shape_functions_as_dict()

#     shape_func = list(filter(lambda x: x["name"] == "Wind speed (m/s)", shape_func_list))[0]

#     # scaler_dict = {
#     #     "Wind speed (m/s)": scaler,
#     #     "Wind speed - Maximum (m/s)": scaler,
#     #     "Wind speed - Minimum (m/s)": scaler,
#     #     "Nacelle ambient temperature (°C)": scaler
#     # }


#     # Input value to highlight
#     value = 1.575

#     # Create the line plot
#     fig, ax = plt.subplots()
#     g = sns.lineplot(
#         x=shape_func["x"], y=shape_func["y"], ax=ax, linewidth=2, color="darkblue"
#     )

#     # Find the y-value for the input value using interpolation
#     y_val = np.interp(input_value, shape_func["x"], shape_func["y"])

#     # Add a vertical line and a marker at the input value
#     ax.axvline(x=input_value, color="red", linestyle="--", label=f"Input Value: {input_value:.2f}")  # Vertical line
#     ax.plot(input_value, y_val, marker="o", markersize=8, color="red")  # Marker
#     # Add horizontal line at the output value (y_val)
#     ax.axhline(y=y_val, color="green", linestyle=":", label=f"Output Value: {y_val:.2f}")

#     # Optional: Add a text label to show the y-value (can be customized)
#     ax.text(input_value, y_val, f"y={y_val:.2f}", color="red", va="bottom")

#     # name the axes
#     ax.set_xlabel("Wind speed (m/s)")

#     plt.legend()  # Show the legend (to explain the red line)
#     plt.show()




def magic_stuff(model: any, scaler: any) -> None:
    shape_func_list = model.get_shape_functions_as_dict()
    shape_func = next(x for x in shape_func_list if x["name"] == "Wind speed (m/s)")
    input_value = 1.575

    # Create the line plot with seaborn
    # sns.set(style="whitegrid")  # Set the plot style
    fig, ax = plt.subplots()
    sns.lineplot(x=shape_func["x"], y=shape_func["y"], ax=ax, linewidth=2, color="darkblue")

    # Find the y-value for the input value using interpolation
    y_val = np.interp(input_value, shape_func["x"], shape_func["y"])

    # Add a marker at the input value with annotation
    ax.plot(input_value, y_val, marker="s", markersize=8, color="black")
    ax.annotate(f"({input_value:.1f}, {y_val:.4f})", (input_value, y_val),
                textcoords="offset points", xytext=(10, 10), ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))


    # Customize plot appearance to match the reference
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)  # Horizontal line at y=0
    ax.set_xlabel("x")  # X-axis label
    ax.set_ylabel("")   # Remove y-axis label
    ax.set_title("OverTime:\n2.12%")  # Add title
    ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust tick label size
    sns.despine(left=True, bottom=True)  # Remove top and right spines



    plt.show()

if __name__ == "__main__":
    # load the model
    model = load_model("igann.pkl")
    scaler = load_model("scaler.pkl")

    # magic stuff
    # magic_stuff(model, scaler)

    # model.plot_single(show_n=4)

    xgboost = load_model("xgboost.pkl")

    # load dataset
    df = preprocess_data(
        path="../dataset/Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv")

    df = df.drop(columns=["Date and time"])
    X_train, X_test, y_train, y_test = split_data(df=df)

    first_entry = X_test.iloc[[0]]
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()

    # scale the data
    first_entry_scaled = first_entry.copy()
    first_entry_scaled.iloc[:, :4] = scaler.transform(first_entry_scaled.iloc[:, :4])

    # print(first_entry_scaled.to_dict())
    baum = first_entry_scaled.to_dict()
    features = dict(map(lambda item: (item[0], list(item[1].values())[0]), baum.items()))
    generate_better_plots(model, features)

    print(first_entry_scaled)

    first_row = X_train
    print(first_row.describe())

    # scale first row
    first_row_scaled = first_row.copy()
    first_row_scaled.iloc[:, :4] = scaler.transform(first_row_scaled.iloc[:, :4])

    # predict
    pred_xgboost = xgboost.predict(first_row)
    pred_igann_scaled = model.predict(first_row_scaled)
    pred_igann = pred_igann_scaled * y_train_std + y_train_mean

    test = -0.9624
    test = test * y_train_std + y_train_mean

    print(pred_xgboost)
    print(pred_igann)
    print(test)


    # print(first_row_scaled)
    #
    dict = {
        'Wind speed (m/s)': 12,
        'Wind speed - Maximum (m/s)': 15,
        'Wind speed - Minimum (m/s)': 10,
        'Nacelle ambient temperature (°C)': 25,
        'E/NE': 0,
        'E/SE': 0,
        'N': 1,
        'N/NE': 0,
        'N/NW': 0,
        'NE': 0,
        'NW': 0,
        'S': 0,
        'S/SE': 0,
        'S/SW': 0,
        'SE': 0,
        'SW': 0,
        'W': 0,
        'W/NW': 0,
        'W/SW': 0
    }

    # tada = pd.DataFrame([dict])

    # # scale tada
    # tada_scaled = tada.copy()
    # tada_scaled.iloc[:, :4] = scaler.transform(tada_scaled.iloc[:, :4])

    # # predict
    # # print(first_row["Nacelle ambient temperature (°C)"])
    # val = model.predict(tada_scaled)
    # print(val)

    # # create empty dataframe

    # # unscale the prediction
    # val = val * y_train.std() + y_train.mean()
    # print(val)

    # print(model._get_feature_importance())

    #
    # get the shape functions
    # print(model.get_shape_functions_as_dict())
