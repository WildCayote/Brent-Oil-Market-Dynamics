import wbdata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Analyzer:
    """
    A class used to analyze economic indicators and their relationships with oil prices.

    Methods
    -------
    fetch_data(indicator_code, indicator_name, country='WLD', start_date=None, end_date=None):
        Fetches data for a given economic indicator from the World Bank API.
    
    clean_data(data):
        Cleans the fetched data by resetting the index, renaming columns, dropping NA values, and converting dates to datetime.
    
    convert_to_daily(data):
        Converts the given data to daily frequency and interpolates missing values.
    
    analyz_and_plot(indicator_data, indicator_name, oil_data, x_label, color):
        Merges the indicator data with oil data, calculates the correlation, and creates a scatter plot.
    
    analyz_indicators(gdp_data, inflation_data, unemployment_data, exchange_rate_data, oil_data):
        Analyzes and plots the relationships between multiple economic indicators and oil prices.
    """

    @staticmethod
    def fetch_data(indicator_code, indicator_name, country='WLD', start_date=None, end_date=None):
        """
        Fetches data for a given economic indicator from the World Bank API.

        Parameters
        ----------
        indicator_code : str
            The code of the economic indicator.
        indicator_name : str
            The name of the economic indicator.
        country : str, optional
            The country code for which data is fetched (default is 'WLD' for World).
        start_date : str, optional
            The start date for fetching data.
        end_date : str, optional
            The end date for fetching data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the fetched data.
        """
        data = wbdata.get_dataframe({indicator_code: indicator_name}, country=country, date=(start_date, end_date))
        return data
    
    @staticmethod
    def clean_data(data):
        """
        Cleans the fetched data by resetting the index, renaming columns, dropping NA values, and converting dates to datetime.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the fetched data.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with proper date format and no missing values.
        """
        if data is not None and not data.empty:
            data.reset_index(inplace=True)
            data.columns = ['date', data.columns[1]]
            data.dropna(inplace=True)
            data['date'] = pd.to_datetime(data['date'])
            return data
        return pd.DataFrame()

    @staticmethod
    def convert_to_daily(data):
        """
        Converts the given data to daily frequency and interpolates missing values.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the data to be converted.

        Returns
        -------
        pd.DataFrame
            DataFrame with daily frequency and interpolated missing values.
        """
        full_index = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D')
        data_daily = data.set_index('date').reindex(full_index)
        data_daily.interpolate(method='time', inplace=True)
        data_daily.reset_index(inplace=True)
        data_daily.rename(columns={'index': 'Date'}, inplace=True)
        return data_daily

    @staticmethod
    def analyz_and_plot(indicator_data, indicator_name, oil_data, x_label, color):
        """
        Merges the indicator data with oil data, calculates the correlation, and creates a scatter plot.

        Parameters
        ----------
        indicator_data : pd.DataFrame
            The DataFrame containing the economic indicator data.
        indicator_name : str
            The name of the economic indicator.
        oil_data : pd.DataFrame
            The DataFrame containing the oil price data.
        x_label : str
            The label for the x-axis in the plot.
        color : str
            The color for the scatter plot points.

        Returns
        -------
        None
        """
        merged_data = pd.merge(indicator_data, oil_data.reset_index(), on='Date')
        
        # Drop NaN values to ensure correlation calculation is valid
        merged_data.dropna(inplace=True)
        correlation = merged_data[indicator_name].corr(merged_data['Price'])
        print(f"Correlation between {indicator_name} and oil prices: {correlation}")

        # Scatter plot
        plt.figure(figsize=(10, 4))
        sns.scatterplot(data=merged_data, x=indicator_name, y='Price', color=color)
        plt.title(f'{indicator_name} vs Brent Oil Prices')
        plt.xlabel(x_label)
        plt.ylabel('Brent Oil Price ($)')
        plt.show()

    @staticmethod
    def analyz_indicators(gdp_data, inflation_data, unemployment_data, exchange_rate_data, oil_data):
        """
        Analyzes and plots the relationships between multiple economic indicators and oil prices.

        Parameters
        ----------
        gdp_data : pd.DataFrame
            The DataFrame containing GDP data.
        inflation_data : pd.DataFrame
            The DataFrame containing inflation data.
        unemployment_data : pd.DataFrame
            The DataFrame containing unemployment data.
        exchange_rate_data : pd.DataFrame
            The DataFrame containing exchange rate data.
        oil_data : pd.DataFrame
            The DataFrame containing oil price data.

        Returns
        -------
        None
        """
        # Analyze GDP
        Analyzer.analyz_and_plot(gdp_data, 'GDP', oil_data, 'GDP Growth Rate (%)', color='SteelBlue')

        # Analyze Inflation
        Analyzer.analyz_and_plot(inflation_data, 'CPI', oil_data, 'Inflation Rate (%)', color='DarkOrange')

        # Analyze Unemployment
        Analyzer.analyz_and_plot(unemployment_data, 'Unemployment Rate', oil_data, 'Unemployment Rate (%)', color='SeaGreen')

        # Analyze Exchange Rate
        Analyzer.analyz_and_plot(exchange_rate_data, 'Exchange Rate', oil_data, 'Exchange Rate (USD to Local Currency)', color='FireBrick')
