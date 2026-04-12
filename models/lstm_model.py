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
        """Create sequences for LSTM — data shape: (n, n_features)"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])  # target is always first column (sales)
        return np.array(X), np.array(y)

    def _add_seasonal_features(self, dates, sales_scaled):
        """Add sin/cos day-of-week features so LSTM learns weekly seasonality"""
        dow = np.array([d.dayofweek for d in dates])
        sin_dow = np.sin(2 * np.pi * dow / 7).reshape(-1, 1)
        cos_dow = np.cos(2 * np.pi * dow / 7).reshape(-1, 1)
        return np.hstack([sales_scaled, sin_dow, cos_dow])  # (n, 3)

    def build_model(self, input_shape=(30, 3)):
        """Build LSTM architecture"""
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
        sales = train_data[target_col].values.reshape(-1, 1)

        # Scale data
        sales_scaled = self.scaler.fit_transform(sales)

        # Add day-of-week cyclic features
        dates = pd.to_datetime(train_data['date']) if 'date' in train_data.columns else train_data.index
        features = self._add_seasonal_features(dates, sales_scaled)  # (n, 3)

        # Create sequences
        X, y = self.create_sequences(features, self.lookback)

        print(f"Training data shape: X={X.shape}, y={y.shape}")

        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
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
            input_df = last_n_days.copy()
            input_sales = input_df['sales'].values if 'sales' in input_df.columns else input_df.values.flatten()
            input_dates = pd.to_datetime(input_df['date']) if 'date' in input_df.columns else input_df.index
        elif isinstance(last_n_days, pd.Series):
            input_sales = last_n_days.values
            input_dates = last_n_days.index
        else:
            input_sales = np.array(last_n_days).flatten()
            input_dates = pd.date_range(end=self.last_date, periods=len(input_sales), freq='D')

        # Take last lookback days
        input_sales = input_sales[-self.lookback:].reshape(-1, 1)
        input_dates = list(input_dates)[-self.lookback:]

        # Scale sales
        sales_scaled = self.scaler.transform(input_sales)

        # Build initial feature matrix with seasonal features
        features = self._add_seasonal_features(input_dates, sales_scaled)
        current_batch = features.reshape((1, self.lookback, features.shape[1]))

        # Make predictions
        predictions = []
        next_date = self.last_date + pd.Timedelta(days=1)

        for i in range(steps):
            pred = self.model.predict(current_batch, verbose=0)
            predictions.append(pred[0, 0])

            # Build next feature row with day-of-week for the forecast date
            dow = next_date.dayofweek
            sin_dow = np.sin(2 * np.pi * dow / 7)
            cos_dow = np.cos(2 * np.pi * dow / 7)
            next_row = np.array([[pred[0, 0], sin_dow, cos_dow]])  # (1, 3)
            next_row_3d = next_row.reshape((1, 1, 3))
            current_batch = np.append(current_batch[:, 1:, :], next_row_3d, axis=1)
            next_date += pd.Timedelta(days=1)

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
        test_dates = pd.to_datetime(test_data['date']) if 'date' in test_data.columns else test_data.index
        test_features = self._add_seasonal_features(test_dates, y_test_scaled)

        # Create sequences
        X_test, y_test = self.create_sequences(test_features, self.lookback)

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
