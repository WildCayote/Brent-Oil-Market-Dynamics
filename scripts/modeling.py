import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.callbacks import EarlyStopping
from keras.api.layers import LSTM, Dense, Input


class VAR_MODEL:
    """
    A class used to represent a Vector AutoRegression (VAR) model for time series prediction.

    Attributes
    ----------
    train : pd.DataFrame
        Training data set.
    test : pd.DataFrame
        Testing data set.

    Methods
    -------
    train_model():
        Trains the VAR model and makes predictions on the test set.
    
    evaluate_model(predictions, actual):
        Evaluates the model's performance using MSE, RMSE, and MAE.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """
        Constructs all the necessary attributes for the VAR_MODEL object.

        Parameters
        ----------
        train : pd.DataFrame
            Training data set.
        test : pd.DataFrame
            Testing data set.
        """
        self.train = train
        self.test = test

    def train_model(self):
        """
        Trains the VAR model on the training data and forecasts future values.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the forecasted values.
        """
        model = VAR(self.train)
        model_fit = model.fit()

        # Get the number of lags
        lag_order = model_fit.k_ar

        # Prepare the last 'lag_order' observations for forecasting
        last_obs = self.train.values[-lag_order:]

        # Forecast the future values
        forecast = model_fit.forecast(last_obs, steps=len(self.test))

        # Create a DataFrame for the predictions
        predictions = pd.DataFrame(forecast, index=self.test.index, columns=self.train.columns)

        # Evaluate the model's performance
        self.evaluate_model(predictions['Price'], self.test['Price'])

        return predictions

    def evaluate_model(self, predictions, actual):
        """
        Evaluates the VAR model's performance using MSE, RMSE, and MAE metrics.

        Parameters
        ----------
        predictions : pd.Series
            The predicted values.
        actual : pd.Series
            The actual values.

        Returns
        -------
        None
        """
        mse = np.mean((predictions - actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual))

        print(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}')

        plt.figure(figsize=(14, 7))
        plt.plot(actual.index, actual, label='Actual Price', color='blue')
        plt.plot(predictions.index, predictions, label='Predicted Price', color='orange')
        plt.title('VAR Model Predictions vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    
class SARIMAX:
    """
    A class used to represent a Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors (SARIMAX) model for time series prediction.

    Attributes
    ----------
    train : pd.DataFrame
        Training data set.
    test : pd.DataFrame
        Testing data set.

    Methods
    -------
    train_model():
        Trains the SARIMAX model and makes predictions on the test set.
    
    evaluate_model(predictions, actual):
        Evaluates the model's performance using MSE, RMSE, and MAE.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """
        Constructs all the necessary attributes for the SARIMAX object.

        Parameters
        ----------
        train : pd.DataFrame
            Training data set.
        test : pd.DataFrame
            Testing data set.
        """
        self.train = train
        self.test = test

    def train_model(self):
        """
        Trains the SARIMAX model on the training data and forecasts future values.

        Returns
        -------
        pd.Series
            Series containing the forecasted values.
        """
        # Define endogenous and exogenous variables
        y_train = self.train['Price']
        exog_train = self.train[['GDP', 'Exchange Rate', 'Pct_Change',
                            '7D_MA', '30D_MA',
                            '7D_Volatility', '30D_Volatility',
                            '7D_Change', '30D_Change']]

        y_test = self.test['Price']
        exog_test = self.test[['GDP', 'Exchange Rate', 'Pct_Change',
                          '7D_MA', '30D_MA',
                          '7D_Volatility', '30D_Volatility',
                          '7D_Change', '30D_Change']]

        try:
            # Fit the SARIMAX model with adjusted parameters
            model = sm.tsa.SARIMAX(y_train,
                                    exog=exog_train,
                                    order=(1, 1, 1),  
                                    seasonal_order=(1, 1, 1, 12), 
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

            results = model.fit(maxiter=1000, disp=False, pgtol=1e-4)

            # Forecasting the next values
            forecast = results.get_forecast(steps=len(self.test), exog=exog_test)
            predictions = forecast.predicted_mean

            # Evaluate the model's performance
            self.evaluate_model(predictions, y_test)

            return predictions

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def evaluate_model(self, predictions, actual):
        """
        Evaluates the SARIMAX model's performance using MSE, RMSE, and MAE metrics.

        Parameters
        ----------
        predictions : pd.Series
            The predicted values.
        actual : pd.Series
            The actual values.

        Returns
        -------
        None
        """
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)

        print(f'MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}')

        plt.figure(figsize=(14, 7))
        plt.plot(actual.index, actual, label='Actual Price', color='blue')
        plt.plot(actual.index, predictions, label='Predicted Price', color='orange')
        plt.title('Model Predictions vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

class LSTM_Model:
    """
    A class used to represent a Long Short-Term Memory (LSTM) model for time series prediction.

    Attributes
    ----------
    train : pd.DataFrame
        Training data set.
    test : pd.DataFrame
        Testing data set.
    validation : pd.DataFrame
        Validation data set.

    Methods
    -------
    prepare_data(data):
        Prepares the data by scaling and creating sequences.
    
    create_lstm_model():
        Creates and compiles the LSTM model.
    
    train_model():
        Trains the LSTM model on the training data.
    
    test_model():
        Tests the LSTM model on the testing data and plots the predictions.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame) -> None:
        """
        Constructs all the necessary attributes for the LSTM_Model object.

        Parameters
        ----------
        train : pd.DataFrame
            Training data set.
        test : pd.DataFrame
            Testing data set.
        validation : pd.DataFrame
            Validation data set.
        """
        self.train = train
        self.test = test
        self.validation = validation
    
    def prepare_data(self, data):
        """
        Prepares the data by scaling the 'Price' values and creating sequences.

        Parameters
        ----------
        data : pd.DataFrame
            The input data set containing 'Price' column.

        Returns
        -------
        np.array
            Arrays for training/testing the model.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

        # Create sequences
        X, y = [], []
        for i in range(30, len(scaled_data)):
            X.append(scaled_data[i-30:i])
            y.append(scaled_data[i])

        # save the scaler 
        self.scaler = scaler

        return np.array(X), np.array(y), scaler
    
    def create_lstm_model(self):
        """
        Creates and compiles the LSTM model.

        Returns
        -------
        keras.models.Sequential
            The compiled LSTM model.
        """
        model = Sequential()
        model.add(Input(shape=(30, 1)))  # 30 time steps and 1 feature
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model
    
    def train_model(self):
        """
        Trains the LSTM model on the training data.

        Returns
        -------
        keras.models.Sequential
            The trained LSTM model.
        keras.callbacks.History
            The history object containing details about the training process.
        """
        # prepare the data for lstm
        X_train, y_train, scaler = self.prepare_data(self.train)
        X_val, y_val, _ = self.prepare_data(self.validation)

        # create the model
        model = self.create_lstm_model()

        # Define early stopping to protect overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',  # The early stopping should use validation loss as its measurement
            patience=7,          # Number of epochs to wait before stopping training if there is no progress
            verbose=1,           # Set the logging to verbose
            mode='min',          # Select the mode as 'min' to notify that we are trying to minimize the loss function
            restore_best_weights=True  # Restore the best weights from training after early stopping
        )

        # train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                          validation_data=(X_val, y_val), callbacks=[early_stopping])
        
        # save the model
        self.model = model
        
        return model, history
    
    def test_model(self):
        """
        Tests the LSTM model on the testing data and plots the predictions.

        Returns
        -------
        None
        """
        scaler = self.scaler

        # Prepare test data
        X_test, y_test, _ = self.prepare_data(self.test)

        # Predictions
        predictions_lstm = self.model.predict(X_test)
        predictions_lstm = scaler.inverse_transform(predictions_lstm)

        # Calculate evaluation metrics
        mse = mean_squared_error(self.test['Price'][30:], predictions_lstm)
        mae = mean_absolute_error(self.test['Price'][30:], predictions_lstm)

        print(f'MSE: {mse:.2f}, MAE: {mae:.2f}')

        # Visualizing LSTM Predictions
        plt.figure(figsize=(14, 7))
        plt.plot(self.test.index, self.test['Price'], label='Actual Price', color='blue')
        plt.plot(self.test.index[30:], predictions_lstm, label='Predicted Price (LSTM)', color='orange')
        plt.title('LSTM Model Predictions vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()