import math, warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore")

sns.set_theme()

class TimeSeriesAnalyzer:
    def __init__(self, time_series_data: pd.DataFrame, date_column: str) -> None:
        self.data = time_series_data
        self.date_col = date_column
    
    def analyze_trend(self, col : str, label: str , title: str):
        plt.plot(self.data[self.date_col], self.data[col], label=label, color='blue')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(col)
        # plt.axvline(x='Date', color='red', linestyle='--', label='Event Marker')
        plt.legend()
        plt.show()
    
    def analyze_seasonality(self):
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