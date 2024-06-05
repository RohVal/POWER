"""
This script is used to benchmark the performance of the differnt models.
Each model is implemented twice. Once with the default parameters and
once with the best parameters found in the hyperparameter tuning.

All models are trained on the same data and the performance is compared
using the MSE, RMSE and R2 score. The models are trained on the data of
turbine 1.

There are two benchmarks:
    - Benchmark 1: The models are tested on the data of turbine 1
    - Benchmark 2: The models are tested on the data of turbine 2

The random state is set to 42 for all models.
"""


from igann import IGANNRegressor
from interpret.glassbox import ExplainableBoostingRegressor as EBMRegressor
from preprocessing import preprocess_data
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from utils import split_data
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_STATE = 42


def benchmark(X: pd.DataFrame, y: pd.Series, models: dict) -> dict:
    """
    This function benchmarks the performance of the models on the data.

    Args:
        X_test (pd.DataFrame): The features of the data
        y_test (pd.Series): The target values of the data
        models (dict): The trained models

    Returns:
        pd.DataFrame: The performance of the models
    """

    results = {}

    for model_name, model in models.items():
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        results[model_name] = {
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        }

    return results


def ebm_model_default(X_train: pd.DataFrame, y_train: pd.Series) -> EBMRegressor:
    """
    This function trains an EBM model with the default parameters on the
    given data.

    Args:
        X_train (pd.DataFrame): The features of the training data
        y_train (pd.Series): The target values of the training data

    Returns:
        EBMRegressor: The trained EBM model
    """

    model = EBMRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model


def ebm_model_tuned(X_train: pd.DataFrame, y_train: pd.Series) -> EBMRegressor:
    """
    This function trains an EBM model with the best parameters found in
    the hyperparameter tuning on the given data.

    Args:
        X_train (pd.DataFrame): The features of the training data
        y_train (pd.Series): The target values of the training data

    Returns:
        EBMRegressor: The trained EBM model
    """

    # TODO add best parameters
    model = EBMRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model


def igann_model_default(X_train: pd.DataFrame, y_train: pd.Series) -> IGANNRegressor:
    """
    This function trains an IGANN model with the default parameters on the
    given data.

    Args:
        X_train (pd.DataFrame): The features of the training data
        y_train (pd.Series): The target values of the training data

    Returns:
        IGANNRegressor: The trained IGANN model
    """

    model = IGANNRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model


def igann_model_tuned(X_train: pd.DataFrame, y_train: pd.Series) -> IGANNRegressor:
    """
    This function trains an IGANN model with the best parameters found in
    the hyperparameter tuning on the given data.

    Args:
        X_train (pd.DataFrame): The features of the training data
        y_train (pd.Series): The target values of the training data

    Returns:
        IGANNRegressor: The trained IGANN model
    """

    model = IGANNRegressor(act="elu", boost_rate=0.2, early_stopping=50, elm_alpha=1,
                           elm_scale=1, init_reg=10, n_estimators=5000, n_hid=20,
                           random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model


def plot_results(benchmark_1: dict, benchmark_2: dict, metric="RMSE") -> None:
    """
    Visualizes a side-by-side comparison of two benchmark results as a grouped bar chart.

    Args:
        benchmark_1 (dict): Dictionary containing the results of the first benchmark (model names as keys, metric values as values).
        benchmark_2 (dict): Dictionary containing the results of the second benchmark (model names as keys, metric values as values).
        metric (str, optional): The metric to plot ("RMSE", "MSE", "R2"). Defaults to "RMSE".
    """

    # Ensure consistent model ordering
    models = list(benchmark_1.keys())
    assert set(benchmark_1.keys()) == set(benchmark_2.keys()), "Benchmarks must have the same models."

    # Prepare data for plotting
    metric_values_1 = [benchmark_1[model][metric] for model in models]
    metric_values_2 = [benchmark_2[model][metric] for model in models]
    x_positions = np.arange(len(models))

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set bar width and calculate positions for grouped bars (side-by-side)
    bar_width = 0.4
    set_offsets = [-bar_width / 2, bar_width / 2]

    # Plot the bars for each set
    bars1 = ax.bar(x_positions + set_offsets[0], metric_values_1, width=bar_width, label="Benchmark 1")
    bars2 = ax.bar(x_positions + set_offsets[1], metric_values_2, width=bar_width, label="Benchmark 2")

    # Add labels to the bars (rounded to three decimal places)
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Add labels, title, and legend
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f"Side-by-Side Benchmark Comparison ({metric})")
    ax.legend()

    plt.tight_layout()
    plt.show()


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> tuple:
    """
    This function scales the data using the StandardScaler.

    Args:
        X_train (pd.DataFrame): The features of the training data
        X_test (pd.DataFrame): The features of the test data
        y_train (pd.Series): The target values of the training data
        y_test (pd.Series): The target values of the test data

    Returns:
        tuple: The scaled data (X_train, X_test, y_train, y_test) and the
               StandardScaler
    """

    scaler = StandardScaler()

    # make a copy of the data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # scale the features
    X_train_scaled.iloc[:, :4] = scaler.fit_transform(X_train_scaled.iloc[:, :4])
    X_test_scaled.iloc[:, :4] = scaler.transform(X_test_scaled.iloc[:, :4])

    # scale the target values as well because IGANN requires it
    y_train_scaled = (y_train - y_train.mean()) / y_train.std()
    y_test_scaled = (y_test - y_train.mean()) / y_train.std()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler


def xgboost_model_default(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    This function trains an XGBoost model with the default parameters on the
    given data.

    Args:
        X_train (pd.DataFrame): The features of the training data
        y_train (pd.Series): The target values of the training data

    Returns:
        XGBRegressor: The trained XGBoost model
    """

    model = XGBRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model


def xgboost_model_tuned(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    This function trains an XGBoost model with the best parameters found in
    the hyperparameter tuning on the given data.

    Args:
        X_train (pd.DataFrame): The features of the training data
        y_train (pd.Series): The target values of the training data

    Returns:
        XGBRegressor: The trained XGBoost model
    """

    model = XGBRegressor(alpha=0.2, booster="gbtree", colsample_bytree=0.8, gamma=0.7,
                         learning_rate=0.005, max_depth=6, min_child_weight=2,
                         n_estimators=1000, reg_lambda=3, subsample=0.8,
                         random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    df_turbine_1 = preprocess_data(
        path="../dataset/Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv")
    df_turbine_2 = preprocess_data(
        path="../dataset/Turbine_Data_Kelmarsh_2_2022-01-01_-_2023-01-01_229.csv")

    # remove date and time column
    df_turbine_1 = df_turbine_1.drop(columns=['Date and time'])
    df_turbine_2 = df_turbine_2.drop(columns=['Date and time'])

    # X_train, X_test, y_train, y_test
    X_train_1, X_test_1, y_train_1, y_test_1 = split_data(df=df_turbine_1)
    X_2, y_2 = df_turbine_2.drop(columns=['Power (kW)']), pd.Series(df_turbine_2['Power (kW)'])

    # scale the data
    # scale the data for turbine 1
    X_train_1, X_test_1, y_train_1, y_test_1, scaler = scale_data(
        X_train=X_train_1, X_test=X_test_1, y_train=y_train_1, y_test=y_test_1)

    # scale the data for turbine 2
    X_2.iloc[:, :4], y_2 = scaler.transform(X_2.iloc[:, :4]), (y_2 - y_2.mean()) / y_2.std()

    # create all models
    # xgboost models
    xgb_default = xgboost_model_default(X_train=X_train_1, y_train=y_train_1)
    xgb_tuned = xgboost_model_tuned(X_train=X_train_1, y_train=y_train_1)

    # igann models
    igann_default = igann_model_default(X_train=X_train_1, y_train=y_train_1)
    igann_tuned = igann_model_tuned(X_train=X_train_1, y_train=y_train_1)

    # ebm models
    ebm_default = ebm_model_default(X_train=X_train_1, y_train=y_train_1)
    # ebm_tuned = ebm_model_tuned(X_train=X_train_1, y_train=y_train_1)

    # run the benchmark
    # TODO add EBM Tuned
    models = {
        "XGBoost Default": xgb_default,
        "XGBoost Tuned": xgb_tuned,
        "IGANN Default": igann_default,
        "IGANN Tuned": igann_tuned,
        "EBM Default": ebm_default
    }

    # benchmark results for all models on the test data of turbine 1
    benchmark_results_turbine_1 = benchmark(models=models, X=X_test_1, y=y_test_1)

    # benchmark results for all models on the data of turbine 2
    benchmark_results_turbine_2 = benchmark(models=models, X=X_2, y=y_2)

    print(benchmark_results_turbine_1)
    print(benchmark_results_turbine_2)

    plot_results(benchmark_results_turbine_1, benchmark_results_turbine_2, "R2")
