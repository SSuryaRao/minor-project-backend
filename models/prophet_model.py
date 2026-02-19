"""
Prophet Model for Sales Forecasting
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import warnings
warnings.filterwarnings('ignore')

class ProphetForecaster:
    def __init__(self, growth='linear', seasonality_mode='multiplicative',
                 changepoint_prior_scale=0.1, seasonality_prior_scale=10.0,
                 yearly_fourier=15, weekly_fourier=5):
        """
        Initialize Prophet model (OPTIMIZED VERSION)

        Parameters:
        -----------
        growth : str
            'linear' or 'logistic'
        seasonality_mode : str
            'additive' or 'multiplicative'
        changepoint_prior_scale : float
            Flexibility of trend (0.05=conservative, 0.5=flexible) - OPTIMIZED to 0.1
        seasonality_prior_scale : float
            Strength of seasonality (default 10.0) - OPTIMIZED
        yearly_fourier : int
            Complexity of yearly seasonality - OPTIMIZED to 15
        weekly_fourier : int
            Complexity of weekly seasonality - OPTIMIZED to 5
        """
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_fourier = yearly_fourier
        self.weekly_fourier = weekly_fourier
        self.model = None
        self.forecast_results = None

    def prepare_data(self, data, target_col='sales'):
        """
        Prepare data for Prophet

        Parameters:
        -----------
        data : pd.DataFrame
            Data with 'date' and target column
        target_col : str
            Name of target column

        Returns:
        --------
        pd.DataFrame with 'ds' and 'y' columns
        """
        df = data.copy()

        if 'date' not in df.columns:
            df = df.reset_index()

        # Prophet requires 'ds' and 'y' columns
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y': df[target_col]
        })

        return prophet_df

    def train(self, train_data, target_col='sales', holidays=None, extra_regressors=None):
        """
        Train Prophet model

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        target_col : str
            Name of target column
        holidays : pd.DataFrame
            Holiday data with 'ds' and 'holiday' columns
        extra_regressors : list
            List of additional regressor column names
        """
        print("Training Prophet model...")

        # Prepare data
        prophet_df = self.prepare_data(train_data, target_col)

        # Initialize model with OPTIMIZED parameters
        self.model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_fourier,  # OPTIMIZED: More complex yearly patterns
            weekly_seasonality=self.weekly_fourier,   # OPTIMIZED: Better weekly patterns
            daily_seasonality=False,
            changepoint_prior_scale=self.changepoint_prior_scale,  # OPTIMIZED: More flexible
            seasonality_prior_scale=self.seasonality_prior_scale,  # OPTIMIZED: Stronger seasonality
            holidays_prior_scale=15.0,  # OPTIMIZED: Give more weight to holidays
            interval_width=0.95  # 95% confidence intervals
        )

        # Add holidays if provided
        if holidays is not None:
            self.model.holidays = holidays

        # Add custom seasonalities for better accuracy
        # Monthly seasonality (captures month-end payday effects)
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=8  # OPTIMIZED: Capture complex monthly patterns
        )

        # Quarterly seasonality (captures quarterly business cycles)
        self.model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=5
        )

        # Add extra regressors if provided
        if extra_regressors:
            for regressor in extra_regressors:
                if regressor in train_data.columns:
                    self.model.add_regressor(regressor)
                    prophet_df[regressor] = train_data[regressor].values

        # Fit model
        self.model.fit(prophet_df)

        print("Model training completed!")
        return self

    def predict(self, steps=7, extra_regressors_df=None):
        """
        Make predictions

        Parameters:
        -----------
        steps : int
            Number of days to forecast
        extra_regressors_df : pd.DataFrame
            DataFrame with extra regressors for future dates

        Returns:
        --------
        pd.DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps, freq='D')

        # Add extra regressors if provided
        if extra_regressors_df is not None:
            for col in extra_regressors_df.columns:
                if col != 'ds':
                    future[col] = extra_regressors_df[col].values

        # Make forecast
        forecast = self.model.predict(future)

        # Extract relevant columns
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
        forecast_df.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']

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
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare test data
        test_df = self.prepare_data(test_data, target_col)

        # Make predictions
        predictions = self.model.predict(test_df)

        y_true = test_df['y'].values
        y_pred = predictions['yhat'].values

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # MAPE with handling for zero values
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

    def plot_forecast(self):
        """Plot forecast with Prophet's built-in visualization"""
        if self.model is None or self.forecast_results is None:
            raise ValueError("Model not trained or no forecast available.")

        fig = self.model.plot(self.model.predict(
            self.model.make_future_dataframe(periods=7, freq='D')
        ))
        return fig

    def plot_components(self):
        """Plot forecast components (trend, seasonality)"""
        if self.model is None:
            raise ValueError("Model not trained.")

        forecast = self.model.predict(self.model.make_future_dataframe(periods=7, freq='D'))
        fig = self.model.plot_components(forecast)
        return fig

    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        joblib.dump({
            'model': self.model,
            'growth': self.growth,
            'seasonality_mode': self.seasonality_mode,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'yearly_fourier': self.yearly_fourier,
            'weekly_fourier': self.weekly_fourier
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.growth = data['growth']
        self.seasonality_mode = data['seasonality_mode']
        self.changepoint_prior_scale = data.get('changepoint_prior_scale', 0.1)
        self.seasonality_prior_scale = data.get('seasonality_prior_scale', 10.0)
        self.yearly_fourier = data.get('yearly_fourier', 15)
        self.weekly_fourier = data.get('weekly_fourier', 5)
        print(f"Model loaded from {filepath}")
        return self
