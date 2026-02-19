"""
Test and Compare Optimized vs Original Models
Shows accuracy improvements from fine-tuning
"""

import sys
sys.path.append('.')

from utils.data_loader import DataLoader
from models.prophet_model import ProphetForecaster
from models.lstm_model import LSTMForecaster
import pandas as pd
import time

def print_header(text):
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80 + "\n")

def test_prophet_models():
    """Test original vs optimized Prophet model"""
    print_header("PROPHET MODEL COMPARISON")

    # Load data
    loader = DataLoader()
    loader.load_data().preprocess_data()

    # Get BREAD/BAKERY data
    bread_data = loader.get_product_data('BREAD/BAKERY')
    recent_data = bread_data.tail(365)
    split = int(len(recent_data) * 0.8)
    train_data = recent_data.iloc[:split]
    test_data = recent_data.iloc[split:]

    print(f"Training data: {len(train_data)} records")
    print(f"Test data: {len(test_data)} records")

    # Test ORIGINAL parameters (conservative)
    print("\n" + "-"*80)
    print("ORIGINAL PROPHET MODEL (Conservative Settings)")
    print("-"*80)

    start_time = time.time()
    model_old = ProphetForecaster(
        changepoint_prior_scale=0.05,  # Original conservative value
        seasonality_prior_scale=10.0,
        yearly_fourier=10,
        weekly_fourier=3
    )
    model_old.train(train_data, target_col='sales')
    metrics_old = model_old.evaluate(test_data, target_col='sales')
    time_old = time.time() - start_time

    print(f"\nTraining time: {time_old:.2f} seconds")
    print(f"MAE: {metrics_old['MAE']:.2f}")
    print(f"RMSE: {metrics_old['RMSE']:.2f}")
    print(f"MAPE: {metrics_old['MAPE']:.2f}%")

    # Test OPTIMIZED parameters
    print("\n" + "-"*80)
    print("OPTIMIZED PROPHET MODEL (Enhanced Settings)")
    print("-"*80)

    start_time = time.time()
    model_new = ProphetForecaster(
        changepoint_prior_scale=0.1,   # OPTIMIZED: More flexible
        seasonality_prior_scale=10.0,
        yearly_fourier=15,              # OPTIMIZED: More complex
        weekly_fourier=5                # OPTIMIZED: Better weekly patterns
    )
    model_new.train(train_data, target_col='sales')
    metrics_new = model_new.evaluate(test_data, target_col='sales')
    time_new = time.time() - start_time

    print(f"\nTraining time: {time_new:.2f} seconds")
    print(f"MAE: {metrics_new['MAE']:.2f}")
    print(f"RMSE: {metrics_new['RMSE']:.2f}")
    print(f"MAPE: {metrics_new['MAPE']:.2f}%")

    # Compare
    print("\n" + "="*80)
    print(" IMPROVEMENT SUMMARY - PROPHET")
    print("="*80)

    mae_improvement = ((metrics_old['MAE'] - metrics_new['MAE']) / metrics_old['MAE']) * 100
    rmse_improvement = ((metrics_old['RMSE'] - metrics_new['RMSE']) / metrics_old['RMSE']) * 100
    mape_improvement = ((metrics_old['MAPE'] - metrics_new['MAPE']) / metrics_old['MAPE']) * 100

    print(f"\nMAE Improvement: {mae_improvement:+.2f}%")
    print(f"RMSE Improvement: {rmse_improvement:+.2f}%")
    print(f"MAPE Improvement: {mape_improvement:+.2f}%")

    if mape_improvement > 0:
        print(f"\n[SUCCESS] Optimized model is {abs(mape_improvement):.2f}% MORE ACCURATE!")
    else:
        print(f"\n[INFO] Original model performed better by {abs(mape_improvement):.2f}%")

    return metrics_old, metrics_new

def test_lstm_models():
    """Test original vs optimized LSTM model"""
    print_header("LSTM MODEL COMPARISON")

    # Load data
    loader = DataLoader()
    loader.load_data().preprocess_data()

    # Get BREAD/BAKERY data
    bread_data = loader.get_product_data('BREAD/BAKERY')
    recent_data = bread_data.tail(365)
    split = int(len(recent_data) * 0.8)
    train_data = recent_data.iloc[:split]
    test_data = recent_data.iloc[split:]

    print(f"Training data: {len(train_data)} records")
    print(f"Test data: {len(test_data)} records")

    # Test ORIGINAL parameters
    print("\n" + "-"*80)
    print("ORIGINAL LSTM MODEL (Basic Architecture)")
    print("-"*80)
    print("Settings: lookback=30, units=50, dropout=0.2, epochs=50")

    start_time = time.time()
    model_old = LSTMForecaster(
        lookback=30,   # Original
        units=50,      # Original
        dropout=0.2,   # Original
        layers=3       # Original
    )
    model_old.train(train_data, target_col='sales', epochs=50, batch_size=32)
    metrics_old = model_old.evaluate(test_data, target_col='sales')
    time_old = time.time() - start_time

    print(f"\nTraining time: {time_old:.2f} seconds")

    # Test OPTIMIZED parameters
    print("\n" + "-"*80)
    print("OPTIMIZED LSTM MODEL (Enhanced Architecture)")
    print("-"*80)
    print("Settings: lookback=60, units=100, dropout=0.3, epochs=100")

    start_time = time.time()
    model_new = LSTMForecaster(
        lookback=60,   # OPTIMIZED: Double lookback
        units=100,     # OPTIMIZED: Double units
        dropout=0.3,   # OPTIMIZED: Better regularization
        layers=4       # OPTIMIZED: Deeper network
    )
    model_new.train(train_data, target_col='sales', epochs=100, batch_size=16)
    metrics_new = model_new.evaluate(test_data, target_col='sales')
    time_new = time.time() - start_time

    print(f"\nTraining time: {time_new:.2f} seconds")

    # Compare
    print("\n" + "="*80)
    print(" IMPROVEMENT SUMMARY - LSTM")
    print("="*80)

    mae_improvement = ((metrics_old['MAE'] - metrics_new['MAE']) / metrics_old['MAE']) * 100
    rmse_improvement = ((metrics_old['RMSE'] - metrics_new['RMSE']) / metrics_old['RMSE']) * 100
    mape_improvement = ((metrics_old['MAPE'] - metrics_new['MAPE']) / metrics_old['MAPE']) * 100

    print(f"\nMAE Improvement: {mae_improvement:+.2f}%")
    print(f"RMSE Improvement: {rmse_improvement:+.2f}%")
    print(f"MAPE Improvement: {mape_improvement:+.2f}%")

    if mape_improvement > 0:
        print(f"\n[SUCCESS] Optimized model is {abs(mape_improvement):.2f}% MORE ACCURATE!")
    else:
        print(f"\n[INFO] Original model performed better by {abs(mape_improvement):.2f}%")

    return metrics_old, metrics_new

def main():
    print_header("MODEL OPTIMIZATION TEST SUITE")
    print("This script compares ORIGINAL vs OPTIMIZED model parameters")
    print("to demonstrate the improvements from fine-tuning.\n")

    # Test Prophet
    print("\n[1/2] Testing Prophet Model...")
    prophet_old, prophet_new = test_prophet_models()

    # Test LSTM (note: this will take longer due to neural network training)
    print("\n[2/2] Testing LSTM Model...")
    print("[WARNING] LSTM training will take 2-5 minutes. Please wait...")
    lstm_old, lstm_new = test_lstm_models()

    # Final summary
    print_header("FINAL COMPARISON SUMMARY")

    print("Prophet Model:")
    print(f"  Original MAPE: {prophet_old['MAPE']:.2f}%")
    print(f"  Optimized MAPE: {prophet_new['MAPE']:.2f}%")
    print(f"  Improvement: {((prophet_old['MAPE'] - prophet_new['MAPE']) / prophet_old['MAPE'] * 100):+.2f}%")

    print("\nLSTM Model:")
    print(f"  Original MAPE: {lstm_old['MAPE']:.2f}%")
    print(f"  Optimized MAPE: {lstm_new['MAPE']:.2f}%")
    print(f"  Improvement: {((lstm_old['MAPE'] - lstm_new['MAPE']) / lstm_old['MAPE'] * 100):+.2f}%")

    print("\n" + "="*80)
    print(" OPTIMIZATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
