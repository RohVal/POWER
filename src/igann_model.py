from preprocessing import preprocess_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from igann import IGANN, IGANNRegressor
from utils import split_data, cross_validation, evaluate_model, log_metrics
from sklearn.model_selection import GridSearchCV
import torch


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Perform hyperparameter tuning using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Train DataFrame.
        y_train (pd.Series): Train target.

    Returns:
        dict: Best hyperparameters.
    """

    # default parameters
    # task="classification", n_hid=10, n_estimators=5000, boost_rate=0.1, init_reg=1,
    # elm_scale=1, elm_alpha=1, sparse=0, act="elu", early_stopping=50, device="cpu",
    # random_state=1, optimize_threshold=False, verbose=0

    model = IGANNRegressor(random_state=42)

    params = {
        "act": [torch.nn.Tanh()],
        "boost_rate": [0.2],
        "early_stopping": [50],
        "elm_alpha": [1],
        "elm_scale": [1],
        "init_reg": [10],
        "n_estimators": [5000],
        "n_hid": [20],
        "random_state": [42],
    }

    # grid search
    grid_search = GridSearchCV(
        model, param_grid=params, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=3)

    grid_search.fit(X_train, y_train)

    return grid_search.best_params_



def scale_data(X_test: pd.DataFrame, X_train: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Scale the data using StandardScaler.

    Args:
        X_test (pd.DataFrame): Test DataFrame.
        X_train (pd.DataFrame): Train DataFrame.

    Returns:
        tuple[StandardScaler, pd.DataFrame, pd.DataFrame]: Tuple containing the scaler and the scaled data.
    """

    # do not scale the wind direction because it is one hot encoded
    # scale the data
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled.iloc[:, :4] = scaler.fit_transform(X_train_scaled.iloc[:, :4])
    X_test_scaled.iloc[:, :4] = scaler.transform(X_test_scaled.iloc[:, :4])

    return scaler, X_train, X_test


if __name__ == "__main__":
    # preprocess the data
    df = preprocess_data(
        path="../dataset/Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv")

    # drop the date column
    df = df.drop(columns=["Date and time"])

    # split the data
    X_train, X_test, y_train, y_test = split_data(df=df)

    # scale the data
    scaler, X_train, X_test = scale_data(X_test=X_test, X_train=X_train)
    y_train = (y_train - y_train.mean()) / y_train.std()
    y_test = (y_test - y_test.mean()) / y_test.std()

    # hyperparameter tuning
    # params = hyperparameter_tuning(X_train, y_train)
    # print(params)

    # best parameters for now
    params = {
        "act": torch.nn.Tanh(),
        "boost_rate": 0.2,
        "early_stopping": 50,
        "elm_alpha": 1,
        "elm_scale": 1,
        "init_reg": 10,
        "n_estimators": 5000,
        "n_hid": 70,
        "random_state": 42,
    }

    # create the model
    model = IGANNRegressor(**params)

    # fit the model
    model.fit(X_train, y_train)
    model.plot_single()

    # log metrics
    metrics = cross_validation(model, X_train, y_train)
    log_metrics(hyperparameters=params, metrics=metrics, model_type="igann")

    # evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics)
