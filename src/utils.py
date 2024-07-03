from pickle import dump, load
from typing import Any
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, RepeatedKFold
import json


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate the model using the metrics MSE, RMSE, MAE, and R2.

    Args:
        model (Any): The model to evaluate.
        X (pd.DataFrame): The features to evaluate the model on.
        y (pd.Series): The target variable to evaluate the model on.

    Returns:
        dict: The metrics MSE, RMSE, MAE, and R2.
    """

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def cross_validation(model: Any, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate the model using cross-validation. Returns the mean of the metrics MSE, RMSE, MAE, and R2.

    Args:
        model (Any): The model to evaluate.
        X (pd.DataFrame): The features to evaluate the model on.
        y (pd.Series): The target variable to evaluate the model on.

    Returns:
        dict: The mean of the metrics MSE, RMSE, MAE, and R2.
    """

    # define the cross-validation procedure (10 folds, 3 repeats)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    mse_scores = cross_val_score(
        model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    rmse_scores = cross_val_score(
        model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, verbose=1)
    mae_scores = cross_val_score(
        model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1, verbose=1)

    return {
        "MSE": -mse_scores.mean(),
        "RMSE": -rmse_scores.mean(),
        "MAE": -mae_scores.mean(),
        "R2": r2_scores.mean()
    }


def load_model(name: str) -> Any:
    """
    Load a model from a file inside the models directory.

    Args:
        name (str): The name of the file to load the model from.

    Returns:
        any: The model loaded from the file.
    """

    with open(f"../models/{name}", 'rb') as file:
        return load(file)


def log_metrics(hyperparameters: dict, metrics: dict, model_type: str) -> None:
    """
    Log the metrics and hyperparameters to a csv file.

    Args:
        hyperparameters (dict): The hyperparameters used to train the model.
        metrics (dict): The metrics calculated for the model.
        model_type (str): The type of model to log the metrics for.

    Returns:
        None
    """

    with open(f"../logs/{model_type}_metrics.csv", "a") as file:
        params = ','.join([f"{value}" for value in hyperparameters.values()])
        metrics = ','.join([f"{value}" for value in metrics.values()])
        file.write(f"{params},{metrics}\n")

    print("Metrics and hyperparameters logged successfully!")


def log_metrics_json(hyperparameters: dict, metrics: dict, model_type: str) -> None:
    """
    Log the metrics and hyperparameters to a json file.

    Args:
        hyperparameters (dict): The hyperparameters used to train the model.
        metrics (dict): The metrics calculated for the model.
        model_type (str): The type of model to log the metrics for.

    Returns:
        None
    """

    with open(f"../logs/{model_type}_metrics.json", "r") as file:
        # load the json file
        data = json.load(file)

        # append the new data
        data.append({
            "hyperparameters": hyperparameters,
            "metrics": metrics
        })

    with open(f"../logs/{model_type}_metrics.json", "w") as file:
        # write the new data
        json.dump(data, file)


def remove_duplicates_from_log(model_type: str) -> None:
    """
    Remove duplicates from the log file.

    Args:
        model_type (str): The type of model to remove duplicates from.
            Used to determine the log file to read from.

    Returns:
        None
    """

    with open(f"../logs/{model_type}_metrics.csv", "r") as file:
        lines = file.readlines()
        header, lines = lines[0], lines[1:]

    with open(f"../logs/{model_type}_metrics.csv", "w") as file:
        file.write(header)
        file.writelines(list(set(lines)))

    print("Duplicates removed successfully!")


def save_model(model: Any, name: str) -> None:
    """
    Save a model to a file inside the models directory.

    Args:
        model (any): The model to save.
        name (str): The name of the file to save the model to.

    Returns:
        None
    """

    with open(f"../models/{name}", 'wb') as file:
        dump(model, file)


def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                  pd.Series, pd.Series]:
    """
    Split the data into training and testing sets. Default test size is 0.2.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        test_size (float): The proportion of the data to include in the test set.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The training and testing
            sets, and the target variables for both.
    """

    X = df.drop(columns=['Power (kW)'])
    y = df['Power (kW)']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test
