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
    
    def visualize_value(self, col : str, label: str , title: str):
        plt.plot(self.data[self.date_col], self.data[col], label=label, color='blue')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(col)
        # plt.axvline(x='Date', color='red', linestyle='--', label='Event Marker')
        plt.legend()
        plt.show()