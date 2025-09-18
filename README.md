# LSTM Time Series Prediction

A comprehensive LSTM-based time series forecasting system with web interface and mock database support.

## Features

- **Advanced LSTM Architecture**: Multi-layer LSTM with dropout, batch normalization, and modern optimization techniques
- **Web Interface**: Beautiful Flask-based UI for training models and making predictions
- **Mock Database**: Generate realistic time series data for stocks, weather, sales, and energy consumption
- **Multiple Datasets**: Support for various time series data types
- **Real-time Training**: Monitor training progress with live metrics and visualizations
- **Future Predictions**: Generate multi-step ahead forecasts
- **Comprehensive Metrics**: RMSE, MAE, R² score, and accuracy measurements

## Supported Data Types

1. **Stock Prices** - Mock stock market data with realistic price movements
2. **Weather Data** - Temperature patterns with seasonal variations
3. **Sales Data** - Business sales with trend and seasonality
4. **Energy Consumption** - Building energy usage with daily/weekly patterns

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 0076_LSTM_for_time_series_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate mock datasets:
```bash
python data_generator.py
```

## Usage

### Command Line Interface

Run the enhanced LSTM model directly:
```bash
python 0076.py
```

### Web Interface

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Use the web interface to:
   - Generate mock datasets
   - Configure model parameters
   - Train LSTM models
   - View training metrics and plots
   - Make future predictions

### Programmatic Usage

```python
from data_loader import TimeSeriesLoader
from data_generator import TimeSeriesDatabase

# Generate mock data
db = TimeSeriesDatabase()
datasets = db.create_all_datasets()

# Load and prepare data
loader = TimeSeriesLoader()
X_train, X_test, y_train, y_test = loader.prepare_data_for_lstm(
    dataset_type="stock", 
    window_size=10
)

# Train model (see 0076.py for complete example)
```

## 📁 Project Structure

```
├── 0076.py              # Enhanced LSTM implementation
├── app.py               # Flask web application
├── data_generator.py    # Mock database generator
├── data_loader.py       # Data loading utilities
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html      # Web interface template
├── data/               # Generated datasets (created automatically)
└── README.md           # This file
```

## 🔧 Configuration

### Model Parameters

- **Window Size**: Number of time steps to look back (default: 10)
- **LSTM Units**: Number of neurons in LSTM layers (default: 50)
- **Epochs**: Maximum training epochs (default: 50)
- **Dropout Rate**: Regularization rate (default: 0.2)

### Dataset Parameters

- **Time Steps**: Length of generated time series
- **Noise Level**: Amount of random variation
- **Seasonal Patterns**: Strength of seasonal components

## Model Architecture

The LSTM model features:

1. **Input Layer**: Sequences of specified window size
2. **LSTM Layer 1**: 50 units with return sequences
3. **Dropout Layer**: 20% dropout for regularization
4. **LSTM Layer 2**: 25 units (half of first layer)
5. **Dropout Layer**: Additional regularization
6. **Batch Normalization**: Stabilize training
7. **Dense Layer**: 25 neurons with ReLU activation
8. **Output Layer**: Single neuron for prediction

## Web Interface Features

- **Modern UI**: Bootstrap-based responsive design
- **Real-time Metrics**: Live training progress and performance metrics
- **Interactive Plots**: Training history and prediction visualizations
- **Easy Configuration**: Intuitive parameter adjustment
- **Future Forecasting**: Multi-step ahead predictions

## Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination
- **Accuracy**: Percentage accuracy based on MAE

## Advanced Features

- **Early Stopping**: Prevent overfitting with patience-based stopping
- **Learning Rate Reduction**: Adaptive learning rate scheduling
- **Validation Split**: Automatic train/validation splitting
- **Residual Analysis**: Visualize prediction errors
- **Multiple Data Sources**: Support for various time series types

## Future Enhancements

- [ ] Multi-variate time series support
- [ ] Attention mechanisms
- [ ] Real-time data streaming
- [ ] Model comparison tools
- [ ] Export trained models
- [ ] API endpoints for predictions

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- Flask 2.3+
- Pandas 2.0+
- Matplotlib 3.7+
- Scikit-learn 1.3+
- Seaborn 0.12+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Flask team for the web framework
- Scikit-learn for preprocessing utilities
- Bootstrap for the UI components

---

**Note**: This project uses mock data for demonstration purposes. For production use, replace the mock data generator with real data sources.
# LSTM-Time-Series-Prediction
