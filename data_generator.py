"""
Mock Database Generator for Time Series Data
Generates various types of time series datasets for LSTM training and testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class TimeSeriesDatabase:
    """Mock database class for generating and storing time series data."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def generate_stock_data(self, symbol="MOCK", days=365, start_price=100):
        """Generate mock stock price data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Generate realistic stock price movements
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        prices = [start_price]
        
        for i in range(1, days):
            price = prices[-1] * (1 + returns[i])
            prices.append(max(price, 1))  # Ensure price doesn't go negative
        
        df = pd.DataFrame({
            'Date': dates,
            'Symbol': symbol,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, days)
        })
        
        return df
    
    def generate_weather_data(self, location="MockCity", days=365):
        """Generate mock weather temperature data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Seasonal temperature pattern
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonal_temp = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Add daily variation and noise
        daily_variation = 5 * np.sin(2 * np.pi * np.arange(days) / 1) 
        noise = np.random.normal(0, 3, days)
        temperature = seasonal_temp + daily_variation + noise
        
        df = pd.DataFrame({
            'Date': dates,
            'Location': location,
            'Temperature': temperature,
            'Humidity': np.random.uniform(30, 90, days),
            'Pressure': np.random.normal(1013, 10, days)
        })
        
        return df
    
    def generate_sales_data(self, product="MockProduct", months=24):
        """Generate mock sales data with trend and seasonality."""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=months, freq='M')
        
        # Trend component
        trend = np.linspace(1000, 2000, months)
        
        # Seasonal component (higher sales in certain months)
        seasonal = 200 * np.sin(2 * np.pi * np.arange(months) / 12)
        
        # Random noise
        noise = np.random.normal(0, 100, months)
        
        sales = trend + seasonal + noise
        sales = np.maximum(sales, 0)  # Ensure non-negative sales
        
        df = pd.DataFrame({
            'Date': dates,
            'Product': product,
            'Sales': sales,
            'Units_Sold': (sales / np.random.uniform(10, 50, months)).astype(int),
            'Revenue': sales * np.random.uniform(0.8, 1.2, months)
        })
        
        return df
    
    def generate_energy_consumption(self, building="MockBuilding", hours=8760):
        """Generate mock energy consumption data (hourly for a year)."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=hours, freq='H')
        
        # Daily pattern (higher consumption during day)
        hour_of_day = np.array([d.hour for d in dates])
        daily_pattern = 50 + 30 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Weekly pattern (lower on weekends)
        day_of_week = np.array([d.weekday() for d in dates])
        weekly_pattern = np.where(day_of_week < 5, 1.0, 0.7)
        
        # Seasonal pattern (higher in summer/winter for AC/heating)
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonal_pattern = 1 + 0.3 * (np.sin(2 * np.pi * (day_of_year - 80) / 365) ** 2)
        
        # Random noise
        noise = np.random.normal(1, 0.1, hours)
        
        consumption = daily_pattern * weekly_pattern * seasonal_pattern * noise
        consumption = np.maximum(consumption, 10)  # Minimum consumption
        
        df = pd.DataFrame({
            'DateTime': dates,
            'Building': building,
            'Energy_Consumption_kWh': consumption,
            'Temperature_C': 20 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2, hours),
            'Occupancy': np.random.randint(0, 100, hours)
        })
        
        return df
    
    def save_dataset(self, df, filename):
        """Save dataset to CSV file."""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")
        return filepath
    
    def load_dataset(self, filename):
        """Load dataset from CSV file."""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    def create_all_datasets(self):
        """Generate and save all mock datasets."""
        datasets = {}
        
        # Generate different types of time series data
        print("Generating mock datasets...")
        
        # Stock data
        stock_df = self.generate_stock_data("MOCK_STOCK", days=500)
        datasets['stock'] = self.save_dataset(stock_df, "stock_data.csv")
        
        # Weather data
        weather_df = self.generate_weather_data("MockCity", days=500)
        datasets['weather'] = self.save_dataset(weather_df, "weather_data.csv")
        
        # Sales data
        sales_df = self.generate_sales_data("MockProduct", months=36)
        datasets['sales'] = self.save_dataset(sales_df, "sales_data.csv")
        
        # Energy consumption data
        energy_df = self.generate_energy_consumption("MockBuilding", hours=2000)
        datasets['energy'] = self.save_dataset(energy_df, "energy_data.csv")
        
        # Save dataset metadata
        metadata = {
            'datasets': datasets,
            'created_at': datetime.now().isoformat(),
            'description': 'Mock time series datasets for LSTM training'
        }
        
        with open(os.path.join(self.data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"All datasets created successfully in '{self.data_dir}' directory!")
        return datasets

if __name__ == "__main__":
    # Create mock database and generate datasets
    db = TimeSeriesDatabase()
    datasets = db.create_all_datasets()
    
    print("\nAvailable datasets:")
    for name, path in datasets.items():
        print(f"- {name}: {path}")
