import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from tensorflow import keras
from keras import layers

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)

print("="*80)
print("LOCALIZED SALES PREDICTION FOR PERISHABLE GOODS")
print("ML Model Training and Evaluation")
print("="*80)

# Load data
print("\n[1/9] Loading data...")
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"   Loaded {len(df)} records")
print(f"   Products: {df['product'].unique()}")

# Focus on one product for demonstration (Milk - most consistent)
product_name = 'Milk'
print(f"\n[2/9] Focusing on product: {product_name}")
df_product = df[df['product'] == product_name].copy()
df_product = df_product.sort_values('date').reset_index(drop=True)

print(f"   Total records for {product_name}: {len(df_product)}")
print(f"   Date range: {df_product['date'].min()} to {df_product['date'].max()}")
print(f"   Average daily sales: {df_product['sales_quantity'].mean():.2f}")
print(f"   Sales range: {df_product['sales_quantity'].min()} - {df_product['sales_quantity'].max()}")

# Train-test split (80-20)
train_size = int(len(df_product) * 0.8)
train_data = df_product[:train_size].copy()
test_data = df_product[train_size:].copy()

print(f"\n[3/9] Data split:")
print(f"   Training set: {len(train_data)} days ({train_data['date'].min()} to {train_data['date'].max()})")
print(f"   Test set: {len(test_data)} days ({test_data['date'].min()} to {test_data['date'].max()})")

# Store results
results = {}

# =============================================
# MODEL 1: FACEBOOK PROPHET
# =============================================
print("\n[4/9] Training Prophet model...")
try:
    # Prepare data for Prophet
    prophet_train = train_data[['date', 'sales_quantity']].copy()
    prophet_train.columns = ['ds', 'y']

    # Add regressors
    prophet_train['temperature'] = train_data['temperature'].values
    prophet_train['is_festival'] = train_data['is_festival'].values
    prophet_train['is_weekend'] = train_data['is_weekend'].values

    # Initialize and train Prophet
    model_prophet = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model_prophet.add_regressor('temperature')
    model_prophet.add_regressor('is_festival')
    model_prophet.add_regressor('is_weekend')

    model_prophet.fit(prophet_train)

    # Prepare test data
    prophet_test = test_data[['date', 'sales_quantity']].copy()
    prophet_test.columns = ['ds', 'y']
    prophet_test['temperature'] = test_data['temperature'].values
    prophet_test['is_festival'] = test_data['is_festival'].values
    prophet_test['is_weekend'] = test_data['is_weekend'].values

    # Make predictions
    forecast = model_prophet.predict(prophet_test)
    prophet_predictions = forecast['yhat'].values

    # Calculate metrics
    mae = mean_absolute_error(test_data['sales_quantity'], prophet_predictions)
    rmse = np.sqrt(mean_squared_error(test_data['sales_quantity'], prophet_predictions))
    mape = np.mean(np.abs((test_data['sales_quantity'] - prophet_predictions) / test_data['sales_quantity'])) * 100
    r2 = r2_score(test_data['sales_quantity'], prophet_predictions)

    results['Prophet'] = {
        'predictions': prophet_predictions,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

    print(f"   Prophet trained successfully")
    print(f"   MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")
except Exception as e:
    print(f"   Prophet training failed: {e}")
    results['Prophet'] = None

# =============================================
# MODEL 2: ARIMA
# =============================================
print("\n[5/9] Training ARIMA model...")
try:
    # ARIMA works best with univariate time series
    arima_model = ARIMA(train_data['sales_quantity'], order=(5, 1, 2))
    arima_fit = arima_model.fit()

    # Make predictions
    arima_predictions = arima_fit.forecast(steps=len(test_data))

    # Calculate metrics
    mae = mean_absolute_error(test_data['sales_quantity'], arima_predictions)
    rmse = np.sqrt(mean_squared_error(test_data['sales_quantity'], arima_predictions))
    mape = np.mean(np.abs((test_data['sales_quantity'] - arima_predictions) / test_data['sales_quantity'])) * 100
    r2 = r2_score(test_data['sales_quantity'], arima_predictions)

    results['ARIMA'] = {
        'predictions': arima_predictions,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

    print(f"   ARIMA trained successfully")
    print(f"   MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")
except Exception as e:
    print(f"   ARIMA training failed: {e}")
    results['ARIMA'] = None

# =============================================
# MODEL 3: XGBOOST
# =============================================
print("\n[6/9] Training XGBoost model...")
try:
    # Feature engineering for XGBoost
    def create_features(df):
        df = df.copy()
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Lag features
        df['lag_1'] = df['sales_quantity'].shift(1)
        df['lag_7'] = df['sales_quantity'].shift(7)
        df['lag_14'] = df['sales_quantity'].shift(14)

        # Rolling average
        df['rolling_mean_7'] = df['sales_quantity'].shift(1).rolling(window=7).mean()
        df['rolling_mean_14'] = df['sales_quantity'].shift(1).rolling(window=14).mean()

        return df

    train_xgb = create_features(train_data)
    test_xgb = create_features(test_data)

    # Drop rows with NaN (from lag features)
    train_xgb = train_xgb.dropna()
    test_xgb = test_xgb.dropna()

    # Select features
    feature_cols = ['temperature', 'is_festival', 'is_weekend', 'day_of_week',
                    'month', 'day', 'week_of_year', 'lag_1', 'lag_7', 'lag_14',
                    'rolling_mean_7', 'rolling_mean_14', 'rainfall']

    X_train = train_xgb[feature_cols]
    y_train = train_xgb['sales_quantity']
    X_test = test_xgb[feature_cols]
    y_test = test_xgb['sales_quantity']

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Make predictions
    xgb_predictions = xgb_model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, xgb_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    mape = np.mean(np.abs((y_test - xgb_predictions) / y_test)) * 100
    r2 = r2_score(y_test, xgb_predictions)

    results['XGBoost'] = {
        'predictions': xgb_predictions,
        'actual': y_test.values,
        'test_dates': test_xgb['date'].values,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'feature_importance': dict(zip(feature_cols, xgb_model.feature_importances_))
    }

    print(f"   XGBoost trained successfully")
    print(f"   MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")
except Exception as e:
    print(f"   XGBoost training failed: {e}")
    results['XGBoost'] = None

# =============================================
# MODEL 4: LSTM
# =============================================
print("\n[7/9] Training LSTM model...")
try:
    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Use multiple features
    lstm_features = ['sales_quantity', 'temperature', 'is_festival', 'is_weekend', 'rainfall']
    train_lstm = train_data[lstm_features].values
    test_lstm = test_data[lstm_features].values

    # Scale data
    train_scaled = scaler.fit_transform(train_lstm)
    test_scaled = scaler.transform(test_lstm)

    # Create sequences
    def create_sequences(data, seq_length=7):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 0])  # Predict sales_quantity
        return np.array(X), np.array(y)

    seq_length = 7
    X_train_lstm, y_train_lstm = create_sequences(train_scaled, seq_length)
    X_test_lstm, y_test_lstm = create_sequences(test_scaled, seq_length)

    # Build LSTM model
    lstm_model = keras.Sequential([
        layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, len(lstm_features))),
        layers.Dropout(0.2),
        layers.LSTM(50, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(25, activation='relu'),
        layers.Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train with less verbose output
    history = lstm_model.fit(
        X_train_lstm, y_train_lstm,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        verbose=0
    )

    # Make predictions
    lstm_predictions_scaled = lstm_model.predict(X_test_lstm, verbose=0)

    # Inverse transform predictions
    # Create dummy array with same shape for inverse transform
    dummy = np.zeros((len(lstm_predictions_scaled), len(lstm_features)))
    dummy[:, 0] = lstm_predictions_scaled.flatten()
    lstm_predictions = scaler.inverse_transform(dummy)[:, 0]

    # Inverse transform actual values
    dummy_actual = np.zeros((len(y_test_lstm), len(lstm_features)))
    dummy_actual[:, 0] = y_test_lstm
    y_test_actual = scaler.inverse_transform(dummy_actual)[:, 0]

    # Calculate metrics
    mae = mean_absolute_error(y_test_actual, lstm_predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
    mape = np.mean(np.abs((y_test_actual - lstm_predictions) / y_test_actual)) * 100
    r2 = r2_score(y_test_actual, lstm_predictions)

    results['LSTM'] = {
        'predictions': lstm_predictions,
        'actual': y_test_actual,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'training_history': history.history
    }

    print(f"   LSTM trained successfully")
    print(f"   MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")
except Exception as e:
    print(f"   LSTM training failed: {e}")
    results['LSTM'] = None

# =============================================
# VISUALIZATIONS
# =============================================
print("\n[8/9] Creating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# Plot 1: Prophet Predictions
if results['Prophet']:
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(test_data['date'], test_data['sales_quantity'], label='Actual', marker='o', linewidth=2)
    ax1.plot(test_data['date'], results['Prophet']['predictions'], label='Prophet Prediction', marker='s', linewidth=2)
    ax1.set_title(f'Prophet Model\nMAPE: {results["Prophet"]["mape"]:.2f}%', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales Quantity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

# Plot 2: ARIMA Predictions
if results['ARIMA']:
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(test_data['date'], test_data['sales_quantity'], label='Actual', marker='o', linewidth=2)
    ax2.plot(test_data['date'], results['ARIMA']['predictions'], label='ARIMA Prediction', marker='s', linewidth=2)
    ax2.set_title(f'ARIMA Model\nMAPE: {results["ARIMA"]["mape"]:.2f}%', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sales Quantity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

# Plot 3: XGBoost Predictions
if results['XGBoost']:
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(results['XGBoost']['test_dates'], results['XGBoost']['actual'], label='Actual', marker='o', linewidth=2)
    ax3.plot(results['XGBoost']['test_dates'], results['XGBoost']['predictions'], label='XGBoost Prediction', marker='s', linewidth=2)
    ax3.set_title(f'XGBoost Model\nMAPE: {results["XGBoost"]["mape"]:.2f}%', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Sales Quantity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

# Plot 4: LSTM Predictions
if results['LSTM']:
    ax4 = plt.subplot(2, 3, 4)
    test_dates_lstm = test_data['date'].values[seq_length:]
    ax4.plot(test_dates_lstm, results['LSTM']['actual'], label='Actual', marker='o', linewidth=2)
    ax4.plot(test_dates_lstm, results['LSTM']['predictions'], label='LSTM Prediction', marker='s', linewidth=2)
    ax4.set_title(f'LSTM Model\nMAPE: {results["LSTM"]["mape"]:.2f}%', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Sales Quantity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

# Plot 5: Model Comparison
ax5 = plt.subplot(2, 3, 5)
models = []
mapes = []
for model_name, result in results.items():
    if result:
        models.append(model_name)
        mapes.append(result['mape'])

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = ax5.bar(models, mapes, color=colors[:len(models)])
ax5.set_title('Model Comparison (MAPE)', fontsize=14, fontweight='bold')
ax5.set_ylabel('MAPE (%)')
ax5.set_ylim(0, max(mapes) * 1.2)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Feature Importance (XGBoost)
if results['XGBoost']:
    ax6 = plt.subplot(2, 3, 6)
    importance = results['XGBoost']['feature_importance']
    features = list(importance.keys())
    importances = list(importance.values())

    # Sort by importance
    sorted_idx = np.argsort(importances)
    features = [features[i] for i in sorted_idx[-10:]]  # Top 10
    importances = [importances[i] for i in sorted_idx[-10:]]

    ax6.barh(features, importances, color='#3498db')
    ax6.set_title('Top 10 Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Importance')
    ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
print("   Saved: model_predictions.png")

# =============================================
# RESULTS SUMMARY
# =============================================
print("\n[9/9] Model Performance Summary")
print("="*80)

summary_df = pd.DataFrame({
    'Model': [],
    'MAE': [],
    'RMSE': [],
    'MAPE (%)': [],
    'R2 Score': []
})

for model_name, result in results.items():
    if result:
        summary_df = pd.concat([summary_df, pd.DataFrame({
            'Model': [model_name],
            'MAE': [f"{result['mae']:.2f}"],
            'RMSE': [f"{result['rmse']:.2f}"],
            'MAPE (%)': [f"{result['mape']:.2f}"],
            'R2 Score': [f"{result['r2']:.4f}"]
        })], ignore_index=True)

print(summary_df.to_string(index=False))

# Find best model
best_model = min(results.items(), key=lambda x: x[1]['mape'] if x[1] else float('inf'))
print(f"\nBest Model: {best_model[0]} (MAPE: {best_model[1]['mape']:.2f}%)")

print("\n" + "="*80)
print("Training completed successfully!")
print("="*80)

# Save results to CSV
summary_df.to_csv('model_comparison.csv', index=False)
print("\nFiles generated:")
print("  - model_predictions.png (Visualizations)")
print("  - model_comparison.csv (Performance metrics)")
