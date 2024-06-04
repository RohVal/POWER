from typing import Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from preprocessing import preprocess_data
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from keras import Sequential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import log_metrics_json


def create_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple[Any, Any, dict]:
    """
    Create a LSTM model.

    Args:
        X_train (pd.DataFrame): the training data
        y_train (pd.DataFrame): the target data

    Returns:
        tuple: a tuple containing the model, the history of the model and the hyperparameters
    """

    params_first_layer = {
        "units": 50,
        "dropout": 0,
        "return_sequences": True,
        "input_shape": (X_train.shape[1], X_train.shape[2]),
    }

    print(X_train.shape[1], X_train.shape[2])

    params_second_layer = {
        "units": 20,
        "dropout": 0,
        "return_sequences": False
    }

    compile_params = {
        "optimizer": Adam(learning_rate=0.01),
        "loss": 'mean_squared_error'
    }

    fit_params = {
        "validation_split": 0.3,
        "shuffle": False,
        "epochs": 1000,
        "batch_size": 16,
        "verbose": 1,
        "callbacks": [EarlyStopping(patience=10, monitor='loss')]
    }

    model = Sequential()
    model.add(LSTM(**params_first_layer))
    model.add(LSTM(**params_second_layer))
    model.add(Dense(units=1))
    model.compile(**compile_params)

    history = model.fit(X_train, y_train, **fit_params)

    hyperparameters = {
        "params_first_layer": params_first_layer,
        "params_second_layer": params_second_layer,
        "compile_params": {"optimizer": "Adam", "loss": "mean_squared_error"},
        "fit_params": {"validation_split": fit_params["validation_split"], "shuffle": fit_params["shuffle"],
                          "epochs": fit_params["epochs"], "batch_size": fit_params["batch_size"], "verbose": fit_params["verbose"],
                          "callbacks": "EarlyStopping(patience=10, monitor='loss')"}
    }

    return model, history, hyperparameters


def scale_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Scale the data using the StandardScaler.

    Args:
        train (pd.DataFrame): the training data
        test (pd.DataFrame): the testing data

    Returns:
        tuple: a tuple containing the scaler, the scaled training data and the scaled testing data
    """

    # scale the data
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled.iloc[:, 1:5] = scaler.fit_transform(train_scaled.iloc[:, 1:5])
    test_scaled.iloc[:, 1:5] = scaler.transform(test_scaled.iloc[:, 1:5])

    return scaler, train_scaled, test_scaled


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the data into training and testing sets. The training set contains of the first
    70% of the data and the testing set contains the remaining 30%.

    Args:
        df (pd.DataFrame): the data to split

    Returns:
        tuple: a tuple containing the training and testing data
    """

    # split the data
    X, y = df.drop(columns=["Power (kW)"]), df["Power (kW)"]
    X_train, X_test = X[:int(0.7 * len(X))], X[int(0.7 * len(X)):]
    y_train, y_test = y[:int(0.7 * len(y))], y[int(0.7 * len(y)):]

    return X_train, y_train, X_test, y_test


def transform(dataset: pd.DataFrame, timestep: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform the data into a format that can be used by the LSTM model. The data is transformed
    into a 3D array. The first dimension represents the number of samples, the second dimension
    represents the number of timesteps and the third dimension represents the number of features.

    Args:
        dataset (pd.DataFrame): the data to transform
        timestep (int): the number of timesteps to use

    Returns:
        tuple: a tuple containing the transformed data
    """

    X, y = [], []
    for i in range(len(dataset)):
        target_value = i + timestep
        if target_value == len(dataset):
            break
        feature_chunk, target = dataset.iloc[i:target_value, :-1],   dataset.iloc[target_value, -1]
        X.append(feature_chunk)
        y.append(target)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # preprocess the data
    df = preprocess_data(path="../dataset/Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv")

    # reorganize the columns
    df = df.drop(columns=["Power (kW)"]).join(df["Power (kW)"])

    # remove the date column
    df = df.drop(columns=["Date and time"])

    # split the data into training and testing sets
    train, test = df[:int(0.7 * len(df))], df[int(0.7 * len(df)):]

    # scale the data
    scaler, train, test = scale_data(train, test)

    # transform the data
    X_train, y_train = transform(train)
    X_test, y_test = transform(test)

    # create the model
    model, history, hyperparams = create_model(X_train, y_train)

    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.show()

    # evaluate the model on the training data, MSE, R2
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    print(f"Mean Squared Error on training data: {mse_train}")
    print(f"R2 on training data: {r2_train}")

    # make predictions
    y_pred = model.predict(X_test)

    # visualize the predicted values and the actual values in form of a plot
    plt.plot(y_test, label="Actual values")
    plt.plot(y_pred, label="Predicted values")
    plt.legend()
    plt.show()

    # calculate the metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    hyperparams["timestep"] = 6

    log_metrics_json(hyperparameters=hyperparams, metrics={
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }, model_type="lstm")

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")


# sources
# https://medium.com/swlh/using-deep-learning-to-forecast-a-wind-turbines-power-output-e87b37b9a50e
# (https://github.com/Sk70249/Wind-Energy-Analysis-and-Forecast-using-Deep-Learning-LSTM/blob/master/Wind%20Energy%20Analysis%20and%20Prediction%20using%20LSTM%20(2).ipynb) not really but pretty similar
# https://www.youtube.com/watch?v=tepxdcepTbY
# worth checking out: https://medium.com/swlh/using-deep-learning-to-forecast-a-wind-turbines-power-output-e87b37b9a50e
