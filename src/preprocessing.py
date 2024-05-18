import pandas as pd


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

    return df

# TODO instead of rolling mean, try using interpolation
