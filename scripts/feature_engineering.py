import pandas as pd

class FeatureEngineering:
    @staticmethod
    def create_pct_change(data:pd.DataFrame):
        data['Pct_Change'] = data['Price'].pct_change()
        return data
    
    @staticmethod
    def create_rolling_avg(data:pd.DataFrame):
        data['7D_MA'] = data['Price'].rolling(window=7).mean()
        data['30D_MA'] = data['Price'].rolling(window=30).mean()
        return data

    @staticmethod
    def create_rolling_volatility(data:pd.DataFrame):
        data['7D_Volatility'] = data['Price'].rolling(window=7).std()
        data['30D_Volatility'] = data['Price'].rolling(window=30).std()
        return data

    @staticmethod
    def create_price_momentum(data:pd.DataFrame):
        data['7D_Change'] = data['Price'] - data['Price'].shift(7)
        data['30D_Change'] = data['Price'] - data['Price'].shift(30)
        return data

    @staticmethod
    def add_features(data:pd.DataFrame):
        data = FeatureEngineering.create_pct_change(data=data)
        data = FeatureEngineering.create_rolling_avg(data=data)
        data = FeatureEngineering.create_rolling_volatility(data=data)
        data = FeatureEngineering.create_price_momentum(data=data)

        return data

    