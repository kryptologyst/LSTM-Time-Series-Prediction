# Project 76. LSTM for time series prediction
# Description:
# An LSTM (Long Short-Term Memory) model is ideal for forecasting time series data where temporal dependencies matter. In this project, we use an LSTM-based neural network built with Keras to predict future values of a univariate time series (e.g., stock price, temperature).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_synthetic_data(time_steps=200):
    """Generates synthetic time series data."""
    np.random.seed(42)
    data = np.sin(np.linspace(0, 20, time_steps)) + np.random.normal(0, 0.2, time_steps)
    return pd.DataFrame({'Value': data})

def create_sequences(data, window_size):
    """Creates sequences from the time series data."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def build_lstm_model(window_size, units=50, dropout_rate=0.2):
    """Builds an enhanced LSTM model with dropout and batch normalization."""
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

def plot_predictions(y_true, y_pred, title="LSTM - Time Series Prediction"):
    """Plots the actual vs. predicted values with enhanced visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Main prediction plot
    ax1.plot(y_true, label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(y_pred, label='Predicted', linestyle='--', linewidth=2, alpha=0.8)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_true.flatten() - y_pred.flatten()
    ax2.scatter(range(len(residuals)), residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax2.set_title("Residuals (Actual - Predicted)", fontsize=12)
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Residual")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_training_history(history):
    """Plots the training history (loss and metrics)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation MAE
    ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig

def main():
    """Main function to run the time series prediction."""
    # Generate and prepare data
    df = generate_synthetic_data()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    window_size = 10
    X, y = create_sequences(scaled_data, window_size)

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build and train the model with callbacks
    model = build_lstm_model(window_size)
    
    # Define callbacks for better training
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    print("Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Make predictions
    y_pred = model.predict(X_test)

    # Inverse scale the data to original representation
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Evaluate the model with multiple metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Model Accuracy: {(1 - mae/np.mean(np.abs(y_test_inv)))*100:.2f}%")
    print("="*50)
    
    # Plot training history
    plot_training_history(history)

    # Plot the results
    plot_predictions(y_test_inv, y_pred_inv)

if __name__ == "__main__":
    main()