import math, warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ruptures as rpt
import pymc as pm
import arviz as az
import numpy as np

warnings.simplefilter(action="ignore")

sns.set_theme()

class TimeSeriesAnalyzer:
    """
    A class used to analyze time series data.

    Attributes
    ----------
    data : pd.DataFrame
        The time series data.
    date_col : str
        The column name for dates in the time series data.
    """

    def __init__(self, time_series_data: pd.DataFrame, date_column: str) -> None:
        """
        Initializes the TimeSeriesAnalyzer with time series data and a date column.

        Parameters
        ----------
        time_series_data : pd.DataFrame
            The time series data.
        date_column : str
            The column name for dates in the time series data.
        """
        self.data = time_series_data
        self.date_col = date_column

    def analyze_trend(self, col: str, label: str, title: str):
        """
        Plots the trend of a specified column over time.

        Parameters
        ----------
        col : str
            The column to analyze.
        label : str
            The label for the plot.
        title : str
            The title for the plot.
        """
        plt.plot(self.data[self.date_col], self.data[col], label=label, color='blue')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(col)
        # plt.axvline(x='Date', color='red', linestyle='--', label='Event Marker')
        plt.legend()
        plt.show()

    def analyze_seasonality(self):
        """
        Analyzes the seasonality in the data by plotting average yearly prices.
        """
        # Extract the year from the Date
        self.data['Year'] = self.data['Date'].dt.year

        # Calculate average price per year
        yearly_avg = self.data.groupby('Year')['Price'].mean().reset_index()

        # Plot yearly average prices
        plt.figure(figsize=(12, 4))
        sns.barplot(x='Year', y='Price', data=yearly_avg, hue='Year', legend=False, palette='husl')
        plt.title('Average Yearly Oil Prices', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Average Price', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def cummlative_sum_analysis(self):
        """
        Performs Cumulative Sum (CUSUM) analysis on the price data and plots the results.
        """
        # Calculate CUSUM (Cumulative Sum of Deviations from Mean)
        mean_price = self.data['Price'].mean()
        cusum = (self.data['Price'] - mean_price).cumsum()

        # Plotting the CUSUM line plot
        plt.figure(figsize=(14, 5))
        plt.plot(self.data['Date'], cusum, label='CUSUM of Price', color='green')

        # Enhancements for better readability
        plt.title('CUSUM Line Plot of Brent Oil Price')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Sum of Deviations')
        plt.legend()
        plt.grid()
        plt.show()

    def event_specific_impact(self):
        """
        Detects change points in the price data using a CUSUM-based method and plots the results.
        """
        # Extract the price series for change point detection
        price_series = self.data['Price'].values

        # Apply the CUSUM-based method for change point detection
        algo = rpt.Binseg(model="l2").fit(price_series)
        change_points = algo.predict(n_bkps=5)  # Adjust n_bkps for more or fewer breakpoints

        # Extract and print the year of each change point
        change_years = [self.data['Date'].iloc[cp].year for cp in change_points[:-1]]  
        print("Detected change point years:", change_years)

        # Plotting the Brent Oil Price with change points
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Date'], self.data['Price'], label='Brent Oil Price', color='blue')

        # Overlay detected change points with year annotations
        for cp in change_points[:-1]: 
            year = self.data['Date'].iloc[cp].year
            plt.axvline(self.data['Date'].iloc[cp], color='red', linestyle='--')
            plt.text(self.data['Date'].iloc[cp], self.data['Price'].iloc[cp], str(year), color="red", fontsize=10)

        # Enhancements
        plt.title('Brent Oil Prices with CUSUM Change Points and Years')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()
        plt.show()

    def baysian_changepoint_analysis(self):
        """
        Performs Bayesian change point analysis on the price data and plots the results.

        Uses a model with informative priors for the change point detection.
        """
        analysis_data = self.data['Price'].values

        # Set informative mean priors based on your analysisd
        prior_mu = np.mean(analysis_data)  # Prior mean for the first segment

        with pm.Model() as model:
            change_point = pm.DiscreteUniform('change_point', lower=0, upper=len(self.data) - 1)
            mu1 = pm.Normal('mu1', mu=prior_mu, sigma=5)
            mu2 = pm.Normal('mu2', mu=prior_mu, sigma=5)
            sigma1 = pm.HalfNormal('sigma1', sigma=5)
            sigma2 = pm.HalfNormal('sigma2', sigma=5)

            likelihood = pm.Normal(
                'likelihood',
                mu=pm.math.switch(change_point >= np.arange(len(analysis_data)), mu1, mu2),
                sigma=pm.math.switch(change_point >= np.arange(len(analysis_data)), sigma1, sigma2),
                observed=analysis_data
            )

            trace = pm.sample(1000, tune=1000, chains=2, random_seed=42)

            az.plot_trace(trace)
            plt.show()

            s_posterior = trace.posterior['change_point'].values.flatten()
            change_point_estimate = int(np.median(s_posterior))
            change_point_date = self.data.iloc[change_point_estimate]['Date']

            print(f"Estimated Change Point Date: {change_point_date}")
        