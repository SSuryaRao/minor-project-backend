"""
LSTM Model for Sales Forecasting
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import warnings
warnings.filterwarnings('ignore')

class LSTMForecaster:
    def __init__(self, lookback=60, units=100, dropout=0.3, layers=4):
        """
        Initialize LSTM model (OPTIMIZED VERSION)

        Parameters:
        -----------
        lookback : int
            Number of past days to use for prediction - OPTIMIZED to 60 (was 30)
        units : int
            Number of LSTM units - OPTIMIZED to 100 (was 50)
        dropout : float
            Dropout rate - OPTIMIZED to 0.3 (was 0.2) for better regularization
        layers : int
            Number of LSTM layers - OPTIMIZED to 4 (was 3)
        """
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.layers = layers
        self.model = None
        self.scaler = MinMaxScaler()
        self.forecast_results = None
        self.history = None
        self.training_data = None  # Store training data for easy prediction
        self.last_date = None  # Store last date

    def create_sequences(self, data, lookback):
        """
        Create sequences for LSTM

        Parameters:
        -----------
        data : np.array
            Time series data
        lookback : int
            Number of time steps to look back

        Returns:
        --------
        X, y : sequences and targets
        """
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build OPTIMIZED LSTM architecture with deeper network"""
        layers_list = []

        # First LSTM layer
        layers_list.append(LSTM(units=self.units, return_sequences=True, input_shape=input_shape))
        layers_list.append(Dropout(self.dropout))

        # Middle LSTM layers (flexible number)
        for i in range(self.layers - 2):
            layers_list.append(LSTM(units=self.units, return_sequences=True))
            layers_list.append(Dropout(self.dropout))

        # Last LSTM layer (no return_sequences)
        layers_list.append(LSTM(units=self.units))
        layers_list.append(Dropout(self.dropout))

        # Dense layers for better learning
        layers_list.append(Dense(units=50, activation='relu'))  # OPTIMIZED: Added intermediate dense layer
        layers_list.append(Dropout(0.2))
        layers_list.append(Dense(units=25, activation='relu'))  # OPTIMIZED: Added another dense layer
        layers_list.append(Dense(units=1))  # Output layer

        model = Sequential(layers_list)

        # OPTIMIZED: Use Adam with custom learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

        model.compile(
            optimizer=optimizer,
            loss='huber',  # OPTIMIZED: Huber loss is more robust to outliers than MSE
            metrics=['mae', 'mse']
        )

        return model

    def train(self, train_data, target_col='sales', epochs=100, batch_size=16, validation_split=0.15):
        """
        Train LSTM model

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        target_col : str
            Name of target column
        epochs : int
            Number of training epochs - OPTIMIZED to 100 (was 50)
        batch_size : int
            Batch size - OPTIMIZED to 16 (was 32) for better gradient updates
        validation_split : float
            Validation split ratio - OPTIMIZED to 0.15 (was 0.1)
        """
        print("Training LSTM model...")

        # Prepare data
        if 'date' in train_data.columns:
            train_data = train_data.sort_values('date')
            # Store last date for forecasting
            self.last_date = pd.to_datetime(train_data['date'].iloc[-1])

        # Store training data for prediction
        self.training_data = train_data.copy()

        # Get target series
        y = train_data[target_col].values.reshape(-1, 1)

        # Scale data
        y_scaled = self.scaler.fit_transform(y)

        # Create sequences
        X, y = self.create_sequences(y_scaled, self.lookback)

        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        print(f"Training data shape: X={X.shape}, y={y.shape}")

        # Build model
        self.model = self.build_model((X.shape[1], 1))
        print("\nModel Architecture:")
        self.model.summary()

        # OPTIMIZED Callbacks for better training
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,  # OPTIMIZED: Increased patience from 10 to 15
            restore_best_weights=True,
            min_delta=0.0001  # OPTIMIZED: Only stop if improvement is significant
        )

        # Learning rate reducer for fine-tuning
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reduce LR by half
            patience=5,
            min_lr=0.00001,
            verbose=1
        )

        # Train model with OPTIMIZED callbacks
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],  # OPTIMIZED: Added learning rate reducer
            verbose=1
        )

        print("\nModel training completed!")
        return self

    def predict(self, last_n_days, steps=7):
        """
        Make predictions

        Parameters:
        -----------
        last_n_days : pd.DataFrame or np.array
            Last n days of data (n >= lookback)
        steps : int
            Number of days to forecast

        Returns:
        --------
        pd.DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare input data
        if isinstance(last_n_days, pd.DataFrame):
            if 'sales' in last_n_days.columns:
                input_data = last_n_days['sales'].values
            else:
                input_data = last_n_days.values.flatten()
        elif isinstance(last_n_days, pd.Series):
            input_data = last_n_days.values
        else:
            input_data = np.array(last_n_days).flatten()

        # Take last lookback days
        input_data = input_data[-self.lookback:]
        input_data = input_data.reshape(-1, 1)

        # Scale
        input_scaled = self.scaler.transform(input_data)

        # Make predictions
        predictions = []
        current_batch = input_scaled.reshape((1, self.lookback, 1))

        for i in range(steps):
            # Predict next day
            pred = self.model.predict(current_batch, verbose=0)
            predictions.append(pred[0, 0])

            # Update batch - reshape prediction to match batch dimensions
            pred_reshaped = pred.reshape((1, 1, 1))
            current_batch = np.append(current_batch[:, 1:, :], pred_reshaped, axis=1)

        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)

        # Create forecast dataframe with dates if last_date is available
        if hasattr(self, 'last_date') and self.last_date is not None:
            forecast_dates = pd.date_range(
                start=self.last_date + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
            forecast_df = pd.DataFrame({
                'forecast': predictions.flatten(),
                'date': forecast_dates
            })
        else:
            forecast_df = pd.DataFrame({
                'forecast': predictions.flatten()
            })

        self.forecast_results = forecast_df
        return forecast_df

    def predict_simple(self, steps=7, target_col='sales'):
        """
        Simplified prediction using stored training data

        Parameters:
        -----------
        steps : int
            Number of days to forecast
        target_col : str
            Name of target column

        Returns:
        --------
        pd.DataFrame with predictions
        """
        if self.training_data is None:
            raise ValueError("No training data stored. Call train() first.")

        # Use the last lookback days from training data
        return self.predict(self.training_data[target_col], steps=steps)

    def evaluate(self, test_data, target_col='sales'):
        """
        Evaluate model on test data

        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
        target_col : str
            Name of target column

        Returns:
        --------
        dict with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare test data
        if 'date' in test_data.columns:
            test_data = test_data.sort_values('date')

        y_test = test_data[target_col].values.reshape(-1, 1)
        y_test_scaled = self.scaler.transform(y_test)

        # Create sequences
        X_test, y_test = self.create_sequences(y_test_scaled, self.lookback)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Make predictions
        y_pred_scaled = self.model.predict(X_test, verbose=0)

        # Inverse transform
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        y_true = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # MAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        return metrics

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            raise ValueError("Model not trained yet.")

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig

    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Save Keras model
        self.model.save(f"{filepath}.h5")

        # Save scaler and parameters
        joblib.dump({
            'scaler': self.scaler,
            'lookback': self.lookback,
            'units': self.units,
            'dropout': self.dropout
        }, f"{filepath}_params.pkl")

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        # Load Keras model
        self.model = keras.models.load_model(f"{filepath}.h5")

        # Load scaler and parameters
        params = joblib.load(f"{filepath}_params.pkl")
        self.scaler = params['scaler']
        self.lookback = params['lookback']
        self.units = params['units']
        self.dropout = params['dropout']

        print(f"Model loaded from {filepath}")
        return self
