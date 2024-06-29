from sklearn.model_selection import cross_val_score, RepeatedKFold
import pandas as pd
from preprocessing import preprocess_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV
from xgboost import XGBRegressor, plot_tree
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from utils import save_model, split_data, cross_validation, log_metrics, evaluate_model


def bayesian_optimization() -> dict:
    """
    Perform hyperparameter tuning using Bayesian Optimization.

    Returns:
        dict: the best hyperparameters
    """

    best_model = XGBRegressor(booster="gbtree", colsample_bytree=0.8, learning_rate=0.01,
                              max_depth=7, min_child_weight=3, n_estimators=800, subsample=0.7, gamma=0.5, reg_lambda=2, alpha=0.5)

    parameter_space = {
        "colsample_bytree": (0.6, 0.9),
        "learning_rate": (0.005, 0.015),
        "max_depth": (3, 7),
        "min_child_weight": (1, 5),
        "n_estimators": (600, 1200),
        "subsample": (0.6, 0.9),
        "gamma": (0.3, 0.7),
        "reg_lambda": (0.5, 5.0),
        "alpha": (0.1, 1.0)
    }

    optimizer = BayesianOptimization(
        f=objective_function, pbounds=parameter_space)

    optimizer.maximize(init_points=10, n_iter=100)

    # Get the best hyperparameters
    best_hyperparams = optimizer.max['params']

    return best_hyperparams


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """P
    Perform hyperparameter tuning using Grid Search.

    Args:
        X_train (pd.DataFrame): the training data
        y_train (pd.Series): the target data

    Returns:
        dict: the best hyperparameters
    """

    model = XGBRegressor(booster="gbtree", colsample_bytree=0.8, learning_rate=0.01,
                         max_depth=7, min_child_weight=3, n_estimators=800, subsample=0.7, gamma=0.5, reg_lambda=2, alpha=0.5)

    params = {
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [5, 6, 7, 8, 9],
        "min_child_weight": [1, 2, 3, 4, 5],
        "n_estimators": [400, 800, 1000],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "alpha": [0.1, 0.5, 1.0]
    }

    # grid search
    grid_search = GridSearchCV(
        model, param_grid=params, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=3)

    print(grid_search.best_params_)

    return grid_search.best_params_


def objective_function(colsample_bytree, learning_rate, max_depth, min_child_weight, n_estimators, subsample, gamma, reg_lambda, alpha):
    """
    Objective function for hyperparameter tuning using Bayesian Optimization.

    Args:
        colsample_bytree (float): the fraction of features to be randomly sampled for each tree
        learning_rate (float): the learning rate
        max_depth (float): the maximum depth of the tree
        min_child_weight (float): the minimum sum of instance weight needed in a child
        n_estimators (float): the number of trees
        subsample (float): the fraction of samples to be randomly sampled for each tree
        gamma (float): the minimum loss reduction required to make a further partition on a leaf node of the tree
        reg_lambda (float): the L2 regularization term on weights
        alpha (float): the L1 regularization term on weights

    Returns:
        float: the mean of the negative mean squared error scores
    """

    model = XGBRegressor(booster="gbtree", colsample_bytree=colsample_bytree, learning_rate=learning_rate,
                         max_depth=int(max_depth), min_child_weight=min_child_weight, n_estimators=int(n_estimators), subsample=subsample, gamma=gamma, reg_lambda=reg_lambda, alpha=alpha)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    mse_scores = cross_val_score(
        model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

    return -mse_scores.mean()


def plot_feature_importance(model: XGBRegressor, X: pd.DataFrame) -> None:
    """
    Plot the feature importance of the model.

    Args:
        model (XGBRegressor): the trained model
        X (pd.DataFrame): the features

    Returns:
        None
    """

    feature_importance = model.feature_importances_
    feature_names = X.columns
    plt.bar(feature_names, feature_importance)
    plt.show()


def plot_temperature_vs_power(path: str) -> None:
    """
    Plot the temperature vs power.

    Args:
        path (str): the path to the dataset

    Returns:
        None
    """

    df = pd.read_csv(path)
    df = df[df['Power (kW)'] < 0]
    plt.scatter(df['Nacelle ambient temperature (°C)'], df['Power (kW)'])
    plt.xlabel("Nacelle ambient temperature (°C)")
    plt.ylabel("Power (kW)")
    plt.title("Temperature vs Power")
    plt.show()


def plot_wind_speed_vs_power(path: str) -> None:
    """
    Plot the wind speed vs power.

    Args:
        path (str): the path to the dataset

    Returns:
        None
    """

    df = pd.read_csv(path)
    plt.scatter(df['Wind speed (m/s)'], df['Power (kW)'])
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power (kW)")
    plt.title("Wind Speed vs Power")
    plt.show()


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    Train the model.

    Args:
        X_train (pd.DataFrame): the training data
        y_train (pd.Series): the target data

    Returns:
        XGBRegressor: the trained model
    """

    # model = XGBRegressor(colsample_bytree=1, gamma=0.5, learning_rate=0.01,
    #                      max_depth=7, n_estimators=1350, subsample=0.7)

    model = XGBRegressor(booster="gbtree", colsample_bytree=0.7, learning_rate=0.01,
                         max_depth=5, min_child_weight=2, n_estimators=900, subsample=0.9)
    model.fit(X_train, y_train, )

    return model


if __name__ == "__main__":
    df = preprocess_data(
        path="../dataset/Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv")

    # remove date and time column
    df = df.drop(columns=['Date and time'])

    # global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = split_data(df=df)

    params = {
        "alpha": 0,
        "booster": "gbtree",
        "colsample_bytree": 1,
        "gamma": 0,
        "learning_rate": 0.01,
        "max_depth": 7,
        "min_child_weight": 2,
        "n_estimators": 1000,
        "reg_lambda": 2,
        "subsample": 0.7
    }

    model = XGBRegressor(random_state=42, **params)
    model.fit(X_train, y_train)

    metrics = cross_validation(model, X_train, y_train)
    log_metrics(params, metrics, "xgboost")

    scores = evaluate_model(model, X_test, y_test)
    print(scores)
