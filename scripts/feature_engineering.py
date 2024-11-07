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
    def create_lag_features(data:pd.DataFrame):
        data['7D_Change'] = data['Price'].diff(periods=7)
        data['30D_Change'] = data['Price'].diff(periods=30)
        return data

    @staticmethod
    def add_features(data:pd.DataFrame):
        data = data.copy()
        data = FeatureEngineering.create_pct_change(data=data)
        data = FeatureEngineering.create_rolling_avg(data=data)
        data = FeatureEngineering.create_rolling_volatility(data=data)
        data = FeatureEngineering.create_price_momentum(data=data)
        data = FeatureEngineering.create_lag_features(data=data)

        data = data.set_index('Date')

        return data

    @staticmethod
    def handle_missing_values(data:pd.DataFrame):
        data = data.copy()

        # Ensure the date index is properly formatted
        data.index = pd.to_datetime(data.index)
        
        # Create a complete date range and reindex
        full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        data = data.reindex(full_date_range)
        
        # Fill missing values using forward fill
        data = data.dropna()

        return data