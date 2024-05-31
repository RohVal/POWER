import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def combine_values_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    The data is recorded every 10 minutes. We have to combine the values to hourly values.
    """

    # convert the date and time column to a datetime object
    df['Date and time'] = pd.to_datetime(df['Date and time'])

    # we have to set an index to resample the data
    df = df.set_index('Date and time')
    df = df.resample('h').mean()
    df.reset_index(inplace=True)

    return df


def encode_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the wind direction column to numerical values using one-hot encoding
    """

    # encode the wind direction column, drop the first column to avoid multicollinearity
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='first').set_output(transform="pandas")

    # fit and transform the column
    encoded_columns = encoder.fit_transform(df[['Wind direction (°)']])

    # name the new columns by the cell values without 'Wind direction' in front
    encoded_columns.columns = [col.split('Wind direction (°)_')[1].strip() for col in encoded_columns.columns]

    # drop the original column and concatenate the encoded columns
    df = pd.concat([df, encoded_columns], axis=1).drop(columns=['Wind direction (°)'])

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataframe
    """

    # drop rows with missing power values, as they are the target values
    df = df.dropna(subset=['Power (kW)'])

    # fill missing values in the columns with the rolling mean
    columns = ["Wind speed (m/s)", "Wind speed - Maximum (m/s)",
               "Wind speed - Minimum (m/s)", "Nacelle ambient temperature (°C)"]

    for column in columns:
        df.loc[:, column] = df[column].fillna(df[column].rolling(
            window=10, min_periods=2, center=True).mean())

    return df


def map_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the wind direction to the corresponding cardinal direction
    """

    directions = {
        (11.25, 33.75): "N/NE",
        (33.75, 56.25): "NE",
        (56.25, 78.75): "E/NE",
        (78.75, 101.25): "E",
        (101.25, 123.75): "E/SE",
        (123.75, 146.25): "SE",
        (146.25, 168.75): "S/SE",
        (168.75, 191.25): "S",
        (191.25, 213.75): "S/SW",
        (213.75, 236.25): "SW",
        (236.25, 258.75): "W/SW",
        (258.75, 281.25): "W",
        (281.25, 303.75): "W/NW",
        (303.75, 326.25): "NW",
        (326.25, 348.75): "N/NW"
    }

    df['Wind direction (°)'] = df['Wind direction (°)'].apply(
        lambda x: next((v for (k1, k2), v in directions.items() if k1 <= x < k2), "N"))

    return df


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from the given path
    """
    return pd.read_csv(path)


def preprocess_data(path: str) -> pd.DataFrame:
    """
    Preprocess the data
    """

    # Load the data
    df = load_data(path=path)

    columns = ["Date and time", "Wind direction (°)", "Wind speed (m/s)",
               "Wind speed - Maximum (m/s)", "Wind speed - Minimum (m/s)",
               "Nacelle ambient temperature (°C)", "Power (kW)"]
    df = remove_columns_except(columns=columns, df=df)
    df = combine_values_hourly(df=df)
    df = handle_missing_values(df=df)
    df = map_wind_direction(df=df)
    df = encode_wind_direction(df=df)
    df = winsorize_power(df=df)

    return df


def remove_columns_except(columns: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all columns from the dataframe except the given columns

    args:
    columns: list[str]
    df: pd.DataFrame

    returns:
    pd.DataFrame
    """
    return df[columns]


def winsorize_power(df, max_power=2050):
    """
    Winsorizes wind power generation data within a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing wind power data.
        column_name (str): Name of the column with wind power data.
        max_power (float): Maximum power output of the wind turbine.

    Returns:
        pd.DataFrame: DataFrame with winsorized data.
    """

    df["Power (kW)"] = df["Power (kW)"].clip(lower=0, upper=max_power)

    return df
# TODO instead of rolling mean, try using interpolation
