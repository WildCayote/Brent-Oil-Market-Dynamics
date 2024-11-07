import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR


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

    # Define the evaluation function
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