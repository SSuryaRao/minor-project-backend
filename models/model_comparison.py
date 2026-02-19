"""
Model Comparison and Evaluation Module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.arima_model import ARIMAForecaster
from models.prophet_model import ProphetForecaster
from models.lstm_model import LSTMForecaster

class ModelComparator:
    def __init__(self):
        """Initialize model comparator"""
        self.models = {}
        self.results = {}
        self.metrics = {}

    def add_model(self, name, model):
        """Add a model to compare"""
        self.models[name] = model

    def train_all(self, train_data, target_col='sales', **kwargs):
        """Train all models"""
        print("="*60)
        print("TRAINING ALL MODELS")
        print("="*60)

        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}...")
            print(f"{'='*60}")

            try:
                if isinstance(model, LSTMForecaster):
                    model.train(
                        train_data,
                        target_col=target_col,
                        epochs=kwargs.get('epochs', 50),
                        batch_size=kwargs.get('batch_size', 32)
                    )
                else:
                    model.train(train_data, target_col=target_col)

                print(f"\n✓ {name} training completed!")
            except Exception as e:
                print(f"\n✗ Error training {name}: {str(e)}")

    def evaluate_all(self, test_data, target_col='sales'):
        """Evaluate all models"""
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)

        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating {name}...")
            print(f"{'='*60}")

            try:
                metrics = model.evaluate(test_data, target_col=target_col)
                self.metrics[name] = metrics
                print(f"\n✓ {name} evaluation completed!")
            except Exception as e:
                print(f"\n✗ Error evaluating {name}: {str(e)}")
                self.metrics[name] = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

    def compare_metrics(self):
        """Compare metrics across all models"""
        if not self.metrics:
            print("No metrics available. Run evaluate_all() first.")
            return None

        # Create comparison dataframe
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df = metrics_df.round(4)

        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(metrics_df)

        # Identify best model for each metric
        print("\n" + "="*60)
        print("BEST MODELS BY METRIC")
        print("="*60)

        for metric in ['MAE', 'RMSE', 'MAPE']:
            if metric in metrics_df.columns:
                best_model = metrics_df[metric].idxmin()
                best_value = metrics_df[metric].min()
                print(f"{metric}: {best_model} ({best_value:.4f})")

        return metrics_df

    def plot_comparison(self):
        """Plot comparison of models"""
        if not self.metrics:
            print("No metrics available. Run evaluate_all() first.")
            return

        metrics_df = pd.DataFrame(self.metrics).T

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics_to_plot = ['MAE', 'RMSE', 'MAPE']
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        for idx, metric in enumerate(metrics_to_plot):
            if metric in metrics_df.columns:
                ax = axes[idx]
                data = metrics_df[metric].sort_values()

                ax.barh(range(len(data)), data.values, color=colors[idx], alpha=0.7)
                ax.set_yticks(range(len(data)))
                ax.set_yticklabels(data.index)
                ax.set_xlabel(metric)
                ax.set_title(f'{metric} Comparison (Lower is Better)')
                ax.grid(axis='x', alpha=0.3)

                # Add value labels
                for i, v in enumerate(data.values):
                    ax.text(v, i, f' {v:.2f}', va='center')

        plt.tight_layout()
        return fig

    def plot_forecasts(self, actual_data, target_col='sales', steps=7):
        """Plot forecasts from all models"""
        if not self.models:
            print("No models available.")
            return

        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot actual data
        if 'date' in actual_data.columns:
            ax.plot(actual_data['date'], actual_data[target_col],
                   label='Actual', color='black', linewidth=2)
            last_date = actual_data['date'].max()
        else:
            ax.plot(actual_data.index, actual_data[target_col],
                   label='Actual', color='black', linewidth=2)
            last_date = actual_data.index[-1]

        # Plot forecasts from each model
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for idx, (name, model) in enumerate(self.models.items()):
            try:
                if isinstance(model, LSTMForecaster):
                    # LSTM needs last n days of data
                    forecast = model.predict(actual_data.tail(model.lookback), steps=steps)
                else:
                    forecast = model.predict(steps=steps)

                # Create future dates
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=steps,
                    freq='D'
                )

                ax.plot(future_dates, forecast['forecast'],
                       label=f'{name} Forecast', color=colors[idx % len(colors)],
                       linewidth=2, linestyle='--', marker='o')

            except Exception as e:
                print(f"Error plotting {name}: {str(e)}")

        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title('Model Forecasts Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def get_best_model(self, metric='RMSE'):
        """Get the best performing model based on a metric"""
        if not self.metrics:
            print("No metrics available. Run evaluate_all() first.")
            return None

        metrics_df = pd.DataFrame(self.metrics).T

        if metric not in metrics_df.columns:
            print(f"Metric {metric} not found.")
            return None

        best_model_name = metrics_df[metric].idxmin()
        best_model = self.models[best_model_name]

        print(f"\nBest model based on {metric}: {best_model_name}")
        print(f"{metric}: {metrics_df.loc[best_model_name, metric]:.4f}")

        return best_model_name, best_model

    def save_comparison_report(self, filepath):
        """Save comparison report to file"""
        if not self.metrics:
            print("No metrics available. Run evaluate_all() first.")
            return

        metrics_df = pd.DataFrame(self.metrics).T

        with open(filepath, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")

            f.write("Metrics Comparison:\n")
            f.write(metrics_df.to_string())
            f.write("\n\n")

            f.write("="*60 + "\n")
            f.write("BEST MODELS BY METRIC\n")
            f.write("="*60 + "\n")

            for metric in ['MAE', 'RMSE', 'MAPE']:
                if metric in metrics_df.columns:
                    best_model = metrics_df[metric].idxmin()
                    best_value = metrics_df[metric].min()
                    f.write(f"{metric}: {best_model} ({best_value:.4f})\n")

        print(f"Comparison report saved to {filepath}")
