"""
Flask Web Application for LSTM Time Series Prediction
Provides a web interface for training and predicting with LSTM models.
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from data_generator import TimeSeriesDatabase
from data_loader import TimeSeriesLoader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lstm_time_series_prediction'

# Global variables to store model and data
current_model = None
current_scaler = None
current_data = None

def create_plot_base64(fig):
    """Convert matplotlib figure to base64 string."""
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return img_str

def build_lstm_model(window_size, units=50, dropout_rate=0.2):
    """Build LSTM model."""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(dropout_rate),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout_rate),
        BatchNormalization(),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/datasets')
def get_datasets():
    """Get available datasets."""
    try:
        loader = TimeSeriesLoader()
        datasets = loader.get_available_datasets()
        return jsonify({'success': True, 'datasets': datasets})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/generate_data', methods=['POST'])
def generate_data():
    """Generate mock datasets."""
    try:
        db = TimeSeriesDatabase()
        datasets = db.create_all_datasets()
        return jsonify({'success': True, 'message': 'Datasets generated successfully', 'datasets': datasets})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train LSTM model."""
    global current_model, current_scaler, current_data
    
    try:
        data = request.get_json()
        dataset_type = data.get('dataset_type', 'stock')
        window_size = int(data.get('window_size', 10))
        epochs = int(data.get('epochs', 50))
        units = int(data.get('units', 50))
        
        # Load and prepare data
        loader = TimeSeriesLoader()
        X_train, X_test, y_train, y_test = loader.prepare_data_for_lstm(
            dataset_type=dataset_type, 
            window_size=window_size
        )
        
        # Store for later use
        current_scaler = loader.scaler
        current_data = {
            'X_test': X_test,
            'y_test': y_test,
            'dataset_type': dataset_type
        }
        
        # Build and train model
        model = build_lstm_model(window_size, units)
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        current_model = model
        
        # Make predictions for evaluation
        y_pred = model.predict(X_test)
        y_test_inv = current_scaler.inverse_transform(y_test)
        y_pred_inv = current_scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        # Create training history plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        training_plot = create_plot_base64(fig)
        
        # Create prediction plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(y_test_inv, label='Actual', linewidth=2)
        ax1.plot(y_pred_inv, label='Predicted', linestyle='--', linewidth=2)
        ax1.set_title(f'LSTM Predictions - {dataset_type.title()} Data')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        residuals = y_test_inv.flatten() - y_pred_inv.flatten()
        ax2.scatter(range(len(residuals)), residuals, alpha=0.6)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title('Residuals')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Residual')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        prediction_plot = create_plot_base64(fig)
        
        return jsonify({
            'success': True,
            'metrics': {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'accuracy': float((1 - mae/np.mean(np.abs(y_test_inv)))*100)
            },
            'training_plot': training_plot,
            'prediction_plot': prediction_plot,
            'epochs_trained': len(history.history['loss'])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Make predictions with trained model."""
    global current_model, current_scaler
    
    try:
        if current_model is None:
            return jsonify({'success': False, 'error': 'No model trained yet'})
        
        data = request.get_json()
        steps = int(data.get('steps', 10))
        
        # Use last sequence from test data for prediction
        if current_data is None:
            return jsonify({'success': False, 'error': 'No data available for prediction'})
        
        X_test = current_data['X_test']
        last_sequence = X_test[-1:]  # Get last sequence
        
        predictions = []
        current_seq = last_sequence.copy()
        
        # Generate future predictions
        for _ in range(steps):
            pred = current_model.predict(current_seq, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 0] = pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_inv = current_scaler.inverse_transform(predictions)
        
        return jsonify({
            'success': True,
            'predictions': predictions_inv.flatten().tolist(),
            'steps': steps
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
