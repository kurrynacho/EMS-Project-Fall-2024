import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import set_style
from sklearn.preprocessing import MinMaxScaler
from dask.dataframe import DataFrame as DaskDataFrame

def subset_data(data, column, value):
    return data[data[column]==value]

def get_time_series(df):
    df=df.copy()

    if isinstance(df, DaskDataFrame):
        df=df.compute()
    # Convert 'eTimes_03' to datetime, handling errors
    df.loc[:, 'DateTime'] = pd.to_datetime(df['eTimes_03'], exact=False, errors='coerce')

    # After 2019, switches datetime format, fix NaT errors
    df.loc[df['DateTime'].isna(), 'DateTime'] = pd.to_datetime(
        df.loc[df['DateTime'].isna(), 'eTimes_03'], 
        format='%d%b%Y:%H:%M:%S', 
        exact=False, 
        errors='coerce'
    )

    # Create a 'Date' column (only date, without time)
    df.loc[:, 'Date'] = df['DateTime'].apply(lambda x: x.date())
    series=df.Date.value_counts().sort_index().reset_index().rename(columns={'index':'Value','Date':'date'})
    return series


def remove_outliers(data, column):
    # Calculate outliers
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    abnormal_dates = data[
        (data[column] < q1 - 1.5 * iqr) | (data[column] > q3 + 1.5 * iqr)
    ].index

    # Ensure abnormal_dates are unique
    abnormal_dates = abnormal_dates.drop_duplicates()

    # Replace outliers with NaN
    data.loc[abnormal_dates, column] = np.nan

    # Replace NaN with mean
    avg = data[column].mean()
    data.loc[:, column] = data[column].fillna(avg)

    return data

def scale_data(data, column):
    scaler=MinMaxScaler()
    scaler.fit(data[column].values.reshape(-1, 1))
    scaler.transform(data[column].values.reshape(-1, 1)).flatten()
    data[column] = scaler.transform(data[column].values.reshape(-1, 1)).flatten()
    return data, scaler

def drop_zeros(data, column):
    data = data.drop(data[data[column] == 0].index)
    return data

def get_processed_series(data):
    series = get_time_series(data)
    series = remove_outliers(series, 'count')
    series, scaler= scale_data(series, 'count')
    series = drop_zeros(series, 'count')
    return series, scaler

def time_series_split(data, train_ratio=0.9):
    """
    Splits a time series dataset into train and test sets, preserving temporal order.

    Parameters:
        data (pd.DataFrame or np.ndarray): The time series dataset to split.
        train_ratio (float): Proportion of the dataset to include in the train set (default is 0.8).

    Returns:
        tuple: (train_set, test_set)
    """
    # Calculate split index
    split_index = int(len(data) * train_ratio)
    
    # Split data
    train_set = data.iloc[:split_index] if hasattr(data, 'iloc') else data[:split_index]
    test_set = data.iloc[split_index:] if hasattr(data, 'iloc') else data[split_index:]
    
    return train_set, test_set

def convert_to_weekly(data):
    df = data.reset_index()
    df['date']=pd.to_datetime(df['date'])

    df.set_index('date', inplace=True)
    weekly_data = df.resample('W').sum()
    return weekly_data.reset_index()

def convert_to_monthly(data):
    df = data.reset_index()
    df['date']=pd.to_datetime(df['date'])

    df.set_index('date', inplace=True)
    weekly_data = df.resample('M').sum()
    return weekly_data.reset_index()

def get_trend(data, window_size=300):
    trend = (
        data['count']
        .rolling(window=window_size, center=False)
        .mean()
        .interpolate(method='linear')
        .bfill()
        .ffill()
    )
    return trend

