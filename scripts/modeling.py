import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error


class VAR_MODEL:
    def __init__(self, train:pd.DataFrame, test:pd.DataFrame) -> None:
        self.train = train
        self.test = test

    def train_model(self):
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
    def __init__(self, train:pd.DataFrame, test:pd.DataFrame) -> None:
        self.train = train
        self.test = test
    
    def train_model(self):
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