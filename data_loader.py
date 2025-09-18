"""
Data Loader for LSTM Time Series Prediction
Handles loading and preprocessing of various time series datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class TimeSeriesLoader:
    """Utility class for loading and preprocessing time series data."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
    
    def load_stock_data(self, filename="stock_data.csv", target_column="Close"):
        """Load stock price data."""
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        return df[target_column].values.reshape(-1, 1)
    
    def load_weather_data(self, filename="weather_data.csv", target_column="Temperature"):
        """Load weather temperature data."""
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        return df[target_column].values.reshape(-1, 1)
    
    def load_sales_data(self, filename="sales_data.csv", target_column="Sales"):
        """Load sales data."""
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        return df[target_column].values.reshape(-1, 1)
    
    def load_energy_data(self, filename="energy_data.csv", target_column="Energy_Consumption_kWh"):
        """Load energy consumption data."""
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.set_index('DateTime').sort_index()
        return df[target_column].values.reshape(-1, 1)
    
    def load_custom_csv(self, filename, date_column=None, target_column=None):
        """Load custom CSV data."""
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath)
        
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column).sort_index()
        
        if target_column and target_column in df.columns:
            return df[target_column].values.reshape(-1, 1)
        else:
            # Return the first numeric column if no target specified
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols[0]].values.reshape(-1, 1)
            else:
                raise ValueError("No numeric columns found in the dataset")
    
    def get_available_datasets(self):
        """Get list of available datasets."""
        datasets = {}
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.csv'):
                    name = filename.replace('.csv', '').replace('_data', '')
                    datasets[name] = filename
        return datasets
    
    def preprocess_data(self, data, fit_scaler=True):
        """Normalize the data using MinMaxScaler."""
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)
        return scaled_data
    
    def inverse_transform(self, scaled_data):
        """Inverse transform the scaled data back to original scale."""
        return self.scaler.inverse_transform(scaled_data)
    
    def create_sequences(self, data, window_size):
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)
    
    def prepare_data_for_lstm(self, dataset_type="stock", window_size=10, test_size=0.2):
        """Complete data preparation pipeline for LSTM."""
        # Load data based on type
        if dataset_type == "stock":
            data = self.load_stock_data()
        elif dataset_type == "weather":
            data = self.load_weather_data()
        elif dataset_type == "sales":
            data = self.load_sales_data()
        elif dataset_type == "energy":
            data = self.load_energy_data()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Preprocess data
        scaled_data = self.preprocess_data(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, window_size)
        
        # Train/test split
        split_idx = int((1 - test_size) * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        return X_train, X_test, y_train, y_test
