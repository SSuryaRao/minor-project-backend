"""
ARIMA Model for Sales Forecasting
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import joblib
import warnings
warnings.filterwarnings('ignore')

class ARIMAForecaster:
    def __init__(self, order=(1, 1, 1), seasonal_order=None):
        """
        Initialize ARIMA model

        Parameters:
        -----------
        order : tuple
            (p, d, q) order for ARIMA
        seasonal_order : tuple or None
            (P, D, Q, s) for SARIMAX if seasonality is needed
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.forecast_results = None
        self.last_date = None  # Store last date from training data

    def check_stationarity(self, series):
        """Check if series is stationary using ADF test"""
        result = adfuller(series.dropna())
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print(f'Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.3f}')

        if result[1] <= 0.05:
            print("Series is stationary")
            return True
        else:
            print("Series is non-stationary")
            return False

    def train(self, train_data, target_col='sales'):
        """
        Train ARIMA model

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with date index and target column
        target_col : str
            Name of target column
        """
        print("Training ARIMA model...")

        # Prepare data
        if not isinstance(train_data.index, pd.DatetimeIndex):
            if 'date' in train_data.columns:
                train_data = train_data.set_index('date')

        # Get target series
        y = train_data[target_col]

        # Store last date for forecasting
        self.last_date = y.index[-1] if isinstance(y.index, pd.DatetimeIndex) else pd.to_datetime(train_data['date'].iloc[-1])

        # Check stationarity
        print("\nChecking stationarity:")
        self.check_stationarity(y)

        # Fit model
        if self.seasonal_order:
            self.model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            self.model = ARIMA(y, order=self.order)

        self.fitted_model = self.model.fit()

        print("\nModel Summary:")
        print(self.fitted_model.summary())

        return self

    def predict(self, steps=7):
        """
        Make predictions

        Parameters:
        -----------
        steps : int
            Number of days to forecast

        Returns:
        --------
        pd.DataFrame with predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Make forecast
        forecast = self.fitted_model.forecast(steps=steps)

        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'forecast': forecast.values,
            'date': forecast_dates
        })

        # Get confidence intervals
        forecast_ci = self.fitted_model.get_forecast(steps=steps).conf_int()
        forecast_df['lower_bound'] = forecast_ci.iloc[:, 0].values
        forecast_df['upper_bound'] = forecast_ci.iloc[:, 1].values

        self.forecast_results = forecast_df
        return forecast_df

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
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare test data
        if not isinstance(test_data.index, pd.DatetimeIndex):
            if 'date' in test_data.columns:
                test_data = test_data.set_index('date')

        y_true = test_data[target_col].values

        # Make predictions
        predictions = self.fitted_model.forecast(steps=len(test_data))
        y_pred = predictions.values

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Handle division by zero in MAPE
        mape = mape if not np.isnan(mape) else 0

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        return metrics

    def save_model(self, filepath):
        """Save trained model"""
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")

        joblib.dump({
            'model': self.fitted_model,
            'order': self.order,
            'seasonal_order': self.seasonal_order
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        data = joblib.load(filepath)
        self.fitted_model = data['model']
        self.order = data['order']
        self.seasonal_order = data['seasonal_order']
        print(f"Model loaded from {filepath}")
        return self
