import pandas as pd

class FeatureEngineering:
    """
    A class used to perform various feature engineering tasks on a time series DataFrame.

    Methods
    -------
    create_pct_change(data: pd.DataFrame) -> pd.DataFrame
        Calculates the percentage change in the 'Price' column.
    
    create_rolling_avg(data: pd.DataFrame) -> pd.DataFrame
        Calculates the 7-day and 30-day moving averages of the 'Price' column.
    
    create_rolling_volatility(data: pd.DataFrame) -> pd.DataFrame
        Calculates the 7-day and 30-day rolling volatility (standard deviation) of the 'Price' column.
    
    create_price_momentum(data: pd.DataFrame) -> pd.DataFrame
        Calculates the price momentum by finding the difference between the current 'Price' and its value 7 and 30 days ago.
    
    create_lag_features(data: pd.DataFrame) -> pd.DataFrame
        Creates lag features by calculating the difference in 'Price' over 7 and 30 days.
    
    add_features(data: pd.DataFrame) -> pd.DataFrame
        Adds all the above features to the DataFrame and sets 'Date' as the index.
    
    handle_missing_values(data: pd.DataFrame) -> pd.DataFrame
        Handles missing values by ensuring a complete date range and forward filling the missing data.
    """

    @staticmethod
    def create_pct_change(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the percentage change in the 'Price' column.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the 'Price' column.
        
        Returns
        -------
        pd.DataFrame
            The DataFrame with a new column 'Pct_Change' representing the percentage change in 'Price'.
        """
        data['Pct_Change'] = data['Price'].pct_change()
        return data
    
    @staticmethod
    def create_rolling_avg(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the 7-day and 30-day moving averages of the 'Price' column.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the 'Price' column.
        
        Returns
        -------
        pd.DataFrame
            The DataFrame with new columns '7D_MA' and '30D_MA' representing the 7-day and 30-day moving averages of 'Price'.
        """
        data['7D_MA'] = data['Price'].rolling(window=7).mean()
        data['30D_MA'] = data['Price'].rolling(window=30).mean()
        return data

    @staticmethod
    def create_rolling_volatility(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the 7-day and 30-day rolling volatility (standard deviation) of the 'Price' column.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the 'Price' column.
        
        Returns
        -------
        pd.DataFrame
            The DataFrame with new columns '7D_Volatility' and '30D_Volatility' representing the rolling volatility.
        """
        data['7D_Volatility'] = data['Price'].rolling(window=7).std()
        data['30D_Volatility'] = data['Price'].rolling(window=30).std()
        return data

    @staticmethod
    def create_price_momentum(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the price momentum by finding the difference between the current 'Price' and its value 7 and 30 days ago.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the 'Price' column.
        
        Returns
        -------
        pd.DataFrame
            The DataFrame with new columns '7D_Change' and '30D_Change' representing the price momentum.
        """
        data['7D_Change'] = data['Price'] - data['Price'].shift(7)
        data['30D_Change'] = data['Price'] - data['Price'].shift(30)
        return data
    
    @staticmethod
    def create_lag_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lag features by calculating the difference in 'Price' over 7 and 30 days.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the 'Price' column.
        
        Returns
        -------
        pd.DataFrame
            The DataFrame with new columns '7D_Change' and '30D_Change' representing the lag features.
        """
        data['7D_Change'] = data['Price'].diff(periods=7)
        data['30D_Change'] = data['Price'].diff(periods=30)
        return data

    @staticmethod
    def add_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all the above features to the DataFrame and sets 'Date' as the index.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the 'Price' and 'Date' columns.
        
        Returns
        -------
        pd.DataFrame
            The DataFrame with additional feature columns.
        """
        data = data.copy()
        data = FeatureEngineering.create_pct_change(data=data)
        data = FeatureEngineering.create_rolling_avg(data=data)
        data = FeatureEngineering.create_rolling_volatility(data=data)
        data = FeatureEngineering.create_price_momentum(data=data)
        data = FeatureEngineering.create_lag_features(data=data)

        data = data.set_index('Date')

        return data

    @staticmethod
    def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values by ensuring a complete date range and forward filling the missing data.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame with missing values.
        
        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values handled.
        """
        data = data.copy()

        # Ensure the date index is properly formatted
        data.index = pd.to_datetime(data.index)
        
        # Create a complete date range and reindex
        full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        data = data.reindex(full_date_range)
        
        # Fill missing values using forward fill
        data = data.dropna()

        return data
