import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import missingno as msno

warnings.filterwarnings('ignore')

# Load the dataset
def load_data(filepath):
    """Load the dataset from the provided filepath."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Overview of the data
def data_overview(df):
    """Print the shape and data types of the dataframe."""
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    print(f"Column Data Types:\n{df.dtypes}")
    return df.head()

# Summary statistics for numerical columns
def summary_statistics(df):
    """Generate summary statistics for numerical columns."""
    print(df.describe())

# Distribution of numerical features
def plot_numerical_distribution(df):
    """Plot histograms and boxplots for numerical features."""
    df.hist(figsize=(10, 8))
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df.select_dtypes(include='number'))
    plt.title('Boxplot for Numerical Features')
    plt.show()

# Distribution of categorical features
def plot_categorical_distribution(df):
    """Plot countplots for top 10 categories in categorical features."""
    for col in df.select_dtypes(include='object').columns:
        top_categories = df[col].value_counts().nlargest(10).index
        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, data=df[df[col].isin(top_categories)])
        plt.title(f'Top 10 Categories of {col}')
        plt.xticks(rotation=45)  # Rotate x-axis labels if needed
        plt.show()

# Correlation analysis
def plot_correlation_matrix(df):
    """Plot the correlation matrix for numerical features."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

# Missing values analysis
def check_missing_values(df):
    """Check and print the count of missing values in the dataframe."""
    missing_values = df.isnull().sum()
    print(f"Missing Values:\n{missing_values[missing_values > 0]}")

    # Visualize missing values
    plt.figure(figsize=(10, 6))
    msno.heatmap(df)
    plt.show()

# Outlier detection
def detect_outliers(df, columns):
    """Plot boxplots for outlier detection in specified columns."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[columns])
    plt.title(f'Outlier Detection for {", ".join(columns)}')
    plt.show()

# Main function to run all steps
def main():
    filepath = '../data/data.csv'  # Change the path as needed
    df = load_data(filepath)

    if df is not None:
        # Data overview and summary
        data_overview(df)
        summary_statistics(df)

        # Plot numerical and categorical distributions
        plot_numerical_distribution(df)
        plot_categorical_distribution(df)

        # Correlation analysis
        plot_correlation_matrix(df)

        # Missing values analysis
        check_missing_values(df)

        # Outlier detection for 'Amount' and 'Value' columns
        detect_outliers(df, ['Amount', 'Value'])

if __name__ == "__main__":
    main()
