import wbdata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Analyzer:
    @staticmethod
    def fetch_data(indicator_code, indicator_name, country='WLD', start_date=None, end_date=None):
        data = wbdata.get_dataframe({indicator_code: indicator_name}, country=country, date=(start_date, end_date))
        return data
    
    @staticmethod
    def clean_data(data):
        if data is not None and not data.empty:
            data.reset_index(inplace=True)
            data.columns = ['date', data.columns[1]]
            data.dropna(inplace=True)
            data['date'] = pd.to_datetime(data['date'])
            return data
        return pd.DataFrame()

    @staticmethod
    def convert_to_daily(data):
        full_index = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D')
        data_daily = data.set_index('date').reindex(full_index)
        data_daily.interpolate(method='time', inplace=True)
        data_daily.reset_index(inplace=True)
        data_daily.rename(columns={'index': 'Date'}, inplace=True)
        return data_daily

    @staticmethod
    def analyz_and_plot(indicator_data, indicator_name, oil_data, x_label, color):
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
        
        # Analyze GDP
        Analyzer.analyz_and_plot(gdp_data, 'GDP', oil_data, 'GDP Growth Rate (%)', color='SteelBlue')

        # Analyze Inflation
        Analyzer.analyz_and_plot(inflation_data, 'CPI', oil_data, 'Inflation Rate (%)', color='DarkOrange')

        # Analyze Unemployment
        Analyzer.analyz_and_plot(unemployment_data, 'Unemployment Rate', oil_data, 'Unemployment Rate (%)', color='SeaGreen')

        # Analyze Exchange Rate
        Analyzer.analyz_and_plot(exchange_rate_data, 'Exchange Rate', oil_data, 'Exchange Rate (USD to Local Currency)', color='FireBrick')
