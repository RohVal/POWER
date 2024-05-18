from sklearn.model_selection import cross_val_score, RepeatedKFold
import pandas as pd
from preprocessing import preprocess_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from utils import save_model


def bayesian_optimization() -> dict:
    """Perform hyperparameter tuning using Bayesian Optimization"""

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
    print(best_hyperparams)


def evaluate_model(model: XGBRegressor, X: pd.DataFrame, y: pd.Series) -> float:
    """Evaluate the model using various metrics"""
    model.fit(X, y)
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def evaluate_model_cross_validation(model, X, y):
    """Evaluate the model using cross-validation and various metrics"""
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    mse_scores = cross_val_score(
        model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    rmse_scores = cross_val_score(
        model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    mae_scores = cross_val_score(
        model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)

    return {
        "MSE": -mse_scores.mean(),
        "RMSE": -rmse_scores.mean(),
        "MAE": -mae_scores.mean(),
        "R2": r2_scores.mean()
    }


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Perform hyperparameter tuning"""

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

    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

    return grid_search.best_params_


def objective_function(colsample_bytree, learning_rate, max_depth, min_child_weight, n_estimators, subsample, gamma, reg_lambda, alpha):
    model = XGBRegressor(booster="gbtree", colsample_bytree=colsample_bytree, learning_rate=learning_rate,
                         max_depth=int(max_depth), min_child_weight=min_child_weight, n_estimators=int(n_estimators), subsample=subsample, gamma=gamma, reg_lambda=reg_lambda, alpha=alpha)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    mse_scores = cross_val_score(
        model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

    return -mse_scores.mean()


def plot_feature_importance(model: XGBRegressor, X: pd.DataFrame) -> None:
    """Plot feature importance as bar chart"""

    feature_importance = model.feature_importances_
    feature_names = X.columns
    plt.bar(feature_names, feature_importance)
    plt.show()


def plot_wind_speed_vs_power(path: str):
    df = pd.read_csv(path)
    plt.scatter(df['Wind speed (m/s)'], df['Power (kW)'])
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power (kW)")
    plt.title("Wind Speed vs Power")
    plt.show()


def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                  pd.Series, pd.Series]:
    """Split the data into training and testing sets"""

    X = df.drop(columns=['Power (kW)'])
    y = df['Power (kW)']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Train the model"""

    # model = XGBRegressor(colsample_bytree=1, gamma=0.5, learning_rate=0.01,
    #                      max_depth=7, n_estimators=1350, subsample=0.7)

    model = XGBRegressor(booster="gbtree", colsample_bytree=0.7, learning_rate=0.01,
                         max_depth=5, min_child_weight=2, n_estimators=900, subsample=0.9)
    model.fit(X_train, y_train, )

    return model


if __name__ == "__main__":
    df = preprocess_data(
        path="./dataset/Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv")

    # remove date and time column
    df = df.drop(columns=['Date and time'])

    # global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = split_data(df=df)

    # best model I've found using grid search and some manual tuning
    best_model = XGBRegressor(booster="gbtree", colsample_bytree=0.8, learning_rate=0.01,
                              max_depth=7, min_child_weight=3, n_estimators=800, subsample=0.7, gamma=0.5, reg_lambda=2, alpha=0.5)

    best_model.fit(X_train, y_train)
    best_metrics = evaluate_model_cross_validation(best_model, X_test, y_test)
    print(best_metrics)

    # bayesian_optimization()

    # best model I've found using bayesian optimization and some manual tuning
    bayesian_model = XGBRegressor(booster="gbtree", alpha=0.2, colsample_bytree=0.8, gamma=0.7, learning_rate=0.005,
                                  max_depth=6, min_child_weight=2, n_estimators=1000, reg_lambda=3, subsample=0.8)
    bayesian_model.fit(X_train, y_train)
    metrics = evaluate_model_cross_validation(bayesian_model, X_test, y_test)
    print(metrics)

    # plot feature importance
    plot_feature_importance(bayesian_model, X_train)

    # save the model
    save_model(bayesian_model, "xgboost.pkl")