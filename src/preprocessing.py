import pandas as pd
from pandas.io.formats.style import plt
from pandas.io.formats.style import plt
from sklearn.preprocessing import OneHotEncoder


def combine_values_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the values in the DataFrame to hourly values. The values are averaged over the hour.

    Args:
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        pd.DataFrame: DataFrame with the values averaged over the hour
    Combine the values in the DataFrame to hourly values. The values are averaged over the hour.

    Args:
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        pd.DataFrame: DataFrame with the values averaged over the hour
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
    Encode the wind direction column using one-hot encoding

    Args:
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        pd.DataFrame: DataFrame with the wind direction column encoded"""

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
    Handle missing values in the DataFrame by filling them with the rolling mean

    Args:
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        pd.DataFrame: DataFrame with missing values filled with the rolling mean
    """

    # drop rows with missing power values, as they are the target values
    df = df.dropna(subset=['Power (kW)'])

    # fill missing values in the columns with the rolling mean
    columns = ["Wind speed (m/s)", "Wind speed - Maximum (m/s)",
               "Wind speed - Minimum (m/s)"]

    for column in columns:
        df.loc[:, column] = df[column].fillna(df[column].rolling(
            window=10, min_periods=2, center=True).mean())

    return df


def handle_missing_temperatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing temperature values in the DataFrame by using interpolation.

    Args:
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        pd.DataFrame: DataFrame with missing temperature values filled with interpolation
    """

    df['Nacelle ambient temperature (°C)'] = df['Nacelle ambient temperature (°C)'].interpolate()

    return df


def map_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the wind direction values to the nearest cardinal direction

    Args:
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        pd.DataFrame: DataFrame with the wind direction values mapped to the nearest cardinal direction
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
    Load the data from the given path

    Args:
        path (str): Path to the data file

    Returns:
        pd.DataFrame: DataFrame containing the data
    """

    return pd.read_csv(path)


def preprocess_data(path: str) -> pd.DataFrame:
    """
    Preprocess the data for the wind turbine power prediction model

    Args:
        path (str): Path to the data file

    Returns:
        pd.DataFrame: Preprocessed DataFrame
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
    df = remove_outliers(df=df, cut_in_speed=3.0)

    return df


def remove_columns_except(columns: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns from the DataFrame that are not in the list of columns to keep.

    Args:
        columns (list[str]): List of columns to keep.
        df (pd.DataFrame): DataFrame containing data.

    Returns:
        pd.DataFrame: DataFrame with columns not in the list removed.
    """

    return df[columns]


def remove_outliers(df, cut_in_speed=3.0) -> pd.DataFrame:
    """
    Filters out rows with zero power output but wind speed above the cut-in speed.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        cut_in_speed (float): Minimum wind speed (m/s) for power generation.

    Returns:
        pd.DataFrame: Filtered DataFrame with rows meeting the criteria removed.
    """

    # Filter rows where 'Power' is zero and 'Wind Speed' is greater than the cut-in speed
    filter_condition = (df["Power (kW)"] == 0) & (df["Wind speed (m/s)"] > cut_in_speed)

    # Invert the filter to select rows that do not meet the condition
    filtered_df = df[~filter_condition]

    return filtered_df


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


def winsorize_wind_speed(df: pd.DataFrame):
    """
    Winsorizes wind speed data (wind speed, wind speed maximum, wind speed minimum)
    within a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing wind speed data.

    Returns:
        pd.DataFrame: DataFrame with winsorized data.
    """

    df["Wind speed (m/s)"] = df["Wind speed (m/s)"].clip(lower=0)
    df["Wind speed - Maximum (m/s)"] = df["Wind speed - Maximum (m/s)"].clip(lower=0)
    df["Wind speed - Minimum (m/s)"] = df["Wind speed - Minimum (m/s)"].clip(lower=0)

    return df

# TODO instead of rolling mean, try using interpolation
