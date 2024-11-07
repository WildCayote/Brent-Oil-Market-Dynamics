import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import stats

data_path = '../../data/data.csv'

def load_price_data():
    """
    Loads the price data from a CSV file.

    The function reads the CSV file located at 'data_path', parses the 'Date' column to datetime format, 
    and sets it as the index of the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the price data with 'Date' as the index.
    """
    data = pd.read_csv(data_path, parse_dates=['Date'])
    data['Date'] = pd.to_datetime(data['Date'], format='mixed')
    data.set_index('Date', inplace=True)
    return data

def get_prices_around_event(event_date, data, days_before=30, days_after=30):
    """
    Gets the prices around a specific event date.

    Parameters
    ----------
    event_date : datetime
        The date of the event.
    data : pd.DataFrame
        The DataFrame containing the price data.
    days_before : int, optional
        Number of days before the event date to include (default is 30).
    days_after : int, optional
        Number of days after the event date to include (default is 30).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the price data within the specified range around the event date.
    """
    before_date = event_date - timedelta(days=days_before)
    after_date = event_date + timedelta(days=days_after)
    return data[(data.index >= before_date) & (data.index <= after_date)]

def calculate_analysis_metrics(data):
    """
    Calculates various analysis metrics for the price data.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the price data.

    Returns
    -------
    dict
        Dictionary containing the calculated metrics including volatility, average price change, 
        minimum price, maximum price, total price change, and correlation.
    """
    volatility = np.std(data['Price']) / np.mean(data['Price'])
    avg_price_change = data['Price'].diff().abs().mean()
    min_price = data['Price'].min()
    max_price = data['Price'].max()
    total_price_change = data['Price'].iloc[-1] - data['Price'].iloc[0]
    correlation = data['Price'].corr(data['Date'].apply(lambda x: x.toordinal()))
    
    return {
        "volatility": round(volatility, 4),
        "average_price_change": round(avg_price_change, 2),
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "total_price_change": round(total_price_change, 2),
        "correlation": round(correlation, 4),
        "model_accuracy": {
            "RMSE": 2.3,  # Placeholder
            "MAE": 1.5    # Placeholder
        }
    }

def calculate_event_impact(event, date, price_data):
    """
    Calculates the impact of an event on price data over different time periods.

    Parameters
    ----------
    event : str
        The name or description of the event.
    date : str
        The date of the event in a recognizable datetime format.
    price_data : pd.DataFrame
        The DataFrame containing the price data with a 'Price' column.

    Returns
    -------
    dict
        Dictionary containing the event's impact metrics including 1-month, 3-month, and 6-month price changes,
        cumulative returns before and after the event, and t-test statistics.
    """
    event_date = pd.to_datetime(date)
    prices_around_event = get_prices_around_event(event_date, price_data, days_before=180, days_after=180)
    change_1m, change_3m, change_6m = None, None, None

    try:
        price_before_1m = price_data.loc[event_date - timedelta(days=30), 'Price']
        price_after_1m = price_data.loc[event_date + timedelta(days=30), 'Price']
        change_1m = ((price_after_1m - price_before_1m) / price_before_1m) * 100
    except KeyError:
        change_1m = None

    try:
        price_before_3m = price_data.loc[event_date - timedelta(days=90), 'Price']
        price_after_3m = price_data.loc[event_date + timedelta(days=90), 'Price']
        change_3m = ((price_after_3m - price_before_3m) / price_before_3m) * 100
    except KeyError:
        change_3m = None

    try:
        price_before_6m = price_data.loc[event_date - timedelta(days=180), 'Price']
        price_after_6m = price_data.loc[event_date + timedelta(days=180), 'Price']
        change_6m = ((price_after_6m - price_before_6m) / price_before_6m) * 100
    except KeyError:
        change_6m = None

    cum_return_before = prices_around_event.loc[:event_date]['Price'].pct_change().add(1).cumprod().iloc[-1] - 1
    cum_return_after = prices_around_event.loc[event_date:]['Price'].pct_change().add(1).cumprod().iloc[-1] - 1
    before_prices = prices_around_event.loc[:event_date]['Price']
    after_prices = prices_around_event.loc[event_date:]['Price']
    t_stat, p_val = stats.ttest_ind(before_prices, after_prices, nan_policy='omit')

    return {
        "Event": event,
        "Date": date,
        "Change_1M": change_1m,
        "Change_3M": change_3m,
        "Change_6M": change_6m,
        "Cumulative Return Before": float(cum_return_before),
        "Cumulative Return After": float(cum_return_after),
        "T-Statistic": float(t_stat),
        "P-Value": float(p_val)
    }

def calculate_price_trends(data):
    """
    Extracts price trends from the data.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the price data.

    Returns
    -------
    dict
        Dictionary containing lists of prices and corresponding dates formatted as year strings.
    """
    return {
        'prices': data['Price'].tolist(),
        'dates': data.index.strftime('%Y').tolist()  # Format dates as year strings
    }

def calculate_price_distribution(data, bin_size=5):
    """
    Calculates the distribution of prices within specified bins.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the price data.
    bin_size : int, optional
        The size of the bins for price distribution (default is 5).

    Returns
    -------
    list
        List of dictionaries containing price ranges and their frequencies.
    """
    # Calculate the minimum and maximum prices to determine the range
    min_price = data['Price'].min()
    max_price = data['Price'].max()

    # Create bins from min to max price
    bins = list(range(int(min_price), int(max_price) + bin_size, bin_size))
    
    # Use pd.cut to create a Series that indicates which bin each price belongs to
    data['PriceRange'] = pd.cut(data['Price'], bins=bins, right=False)

    # Group by the PriceRange and count the occurrences
    distribution = data.groupby('PriceRange').size().reset_index(name='Frequency')

    # Format the distribution for visualization
    distribution['PriceRange'] = distribution['PriceRange'].astype(str)  # Convert bins to string for better visualization
    return distribution.to_dict(orient='records')  # Return as a list of dictionaries

def calculate_yearly_average_price(data):
    """
    Calculates the yearly average price.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the price data.

    Returns
    -------
    list
        List of dictionaries containing years and their corresponding average prices.
    """
    yearly_avg = data['Price'].resample('YE').mean()
    return yearly_avg.reset_index().rename(columns={'Date': 'Year', 'Price': 'Average_Price'}).to_dict(orient='records')