import math, warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore")

sns.set_theme()

class EDAAnalyzer:
    """
    A class for organizing functions/methods for performing EDA on bank transaction data.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the EDAAnalyzer class

        Args:
            data(pd.DataFrame): the dataframe that contains bank transactional data
        """
        self.data = data
    
    def basic_overview(self) -> None:
        """
        A function that creates basic overview of the data like - data type of columns, the shape of the data(i.e the number of rows and columns) 
        """
        # print out the shape
        print(f"The data has a shape of: {self.data.shape}")

        # print out the column info
        print(self.data.info())
    
    def summary_statistics(self) -> None:
        """
        A function that generates 5 number summary(descriptive statistics) of the dataframe
        """
        print(self.data.describe())
    
    def missing_values(self) -> None:
        """
        A function that checks for columns with missing value and then returns ones with greater than 0 with the percentage of missing values.
        """

        # obtain missing value percentage
        missing = self.data.isna().mean() * 100
        missing = missing [missing > 0]
        
        # print out the result
        print(f"These are columns with missing values greater than 0%:\n{missing}")

    def determine_duplicate(self) -> None:
        """
        A function that finds the percentage of duplicate data/rows in the given data set. It prints the percentage of that duplicate data
        """

        # obtain ratio of dublicate data
        duplicate_ratio = self.data.duplicated().mean()

        # calcualte the percentage from ration
        percentage = duplicate_ratio * 100

        # print out the result
        print(f"The data has {percentage}% duplicate data.")
    
    def outlire_detection(self) -> None:
        """
        A function that performs outlire detection by plotting a box plot.
        """
        # create the box plots of the numeric data
        ax = sns.boxplot(data=self.data, palette='husl')
        ax.set_title("Box-plot of Categorical Variables", pad=30, fontweight='bold')
        ax.set_xlabel("Numerical Columns", fontweight='bold', labelpad=10)
        ax.set_ylabel("Values", fontweight='bold', labelpad=10)
    
    def count_outliers(self) -> None:
        """
        A function that counts the number of outliers in numerical columns. The amount of data that are outliers and also gives the cut-off point.
        The cut off points being defined as:
            - lowerbound = Q1 - 1.5 * IQR
            - upperbound = Q3 + 1.5 * IQR
        """
        # get the numeric data
        numerical_columns = list(self.data._get_numeric_data().columns)
        numerical_data = self.data[numerical_columns]

        # obtain the Q1, Q3 and IQR(Inter-Quartile Range)
        quartile_one = numerical_data.quantile(0.25)
        quartile_three = numerical_data.quantile(0.75)
        iqr = quartile_three - quartile_one

        # obtain the upperbound and lowerbound values for each column
        upper_bound = quartile_three + 1.5 * iqr
        lower_bound = quartile_one - 1.5 * iqr

        # count all the outliers for the respective columns
        outliers = {"Columns" : [], "Num. of Outliers": []}
        for column in lower_bound.keys():
            column_outliers = self.data[(self.data[column] < lower_bound[column]) | (self.data[column] > upper_bound[column])]
            count = column_outliers.shape[0]

            outliers["Columns"].append(column)
            outliers["Num. of Outliers"].append(count)

        outliers = pd.DataFrame.from_dict(outliers).sort_values(by='Num. of Outliers')
        ax = sns.barplot(outliers, x='Columns', y='Num. of Outliers', palette='husl')
        ax.set_title("Plot of Skewness values of Numerical Columns", pad=20)
        ax.set_xlabel("Numerical Columns", weight='bold')
        ax.set_ylabel("Num. of Outliers", weight="bold")
        ax.tick_params(axis='x', labelrotation=45)

        columns = outliers['Columns'].unique()
        for idx, patch in enumerate(ax.patches):
            # get the corrdinates to write the values
            x_coordinate = patch.get_x() + patch.get_width() / 2
            y_coordinate = patch.get_height()

            # get the value of the coordinate
            value = outliers[outliers['Columns'] == columns[idx]]['Num. of Outliers'].values[0]
            ax.text(x=x_coordinate, y=y_coordinate, s=value, ha='center', va='bottom', weight='bold')

    def merge_event(self, events_data: pd.DataFrame) -> pd.DataFrame:
        """
        A function that mergres historical event data with price data

        Args:
            events_data(pd.DataFrame): the dataframe which contains the historical events
        
        Returns:
            merged_data(pd.DataFrane): a new data frame with the datasets merged
        """
        merged_data = pd.DataFrame()
        merged_data['Date'] = pd.to_datetime(self.data['Date'])
        events_data['Start'] = pd.to_datetime(events_data['Start'])
        events_data['End'] = pd.to_datetime(events_data['End'])

        events_expanded = pd.DataFrame({
            'Date': pd.date_range(start=events_data['Start'].min(), end=events_data['End'].max(), freq='D')
        })

        merged_data = pd.merge_asof(
            events_expanded.sort_values('Date'), 
            events_data.sort_values('Start'), 
            left_on='Date', 
            right_on='Start', 
            direction='backward'
        )

        merged_data = merged_data.merge(merged_data[['Date', 'Event', 'Category']], on='Date', how='left')

        merged_data.fillna({'Event': 'No Event', 'Category': 'No Category'}, inplace=True)

        return merged_data