import wbdata
import pandas as pd

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
