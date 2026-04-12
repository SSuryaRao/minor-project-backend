"""
Flask Backend API for Sales Forecasting System
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

from config.config import Config
from utils.data_loader import DataLoader
from utils.api_integrations import WeatherAPI, EventsAPI, ExternalDataIntegrator
from utils.inventory_optimizer import InventoryOptimizer
from utils.upload_handler import (
    parse_uploaded_file, detect_columns, validate_and_prepare,
    store_upload, get_upload
)
from models.arima_model import ARIMAForecaster
from models.prophet_model import ProphetForecaster
from models.lstm_model import LSTMForecaster
from utils.gemini_insights import forecast_summary, inventory_alert, anomaly_explain

# Initialize Flask app (API only - no templates)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB upload limit
CORS(app, resources={r"/api/*": {
    "origins": "*",
    "allow_headers": ["Content-Type", "X-Internal-Secret"],
}})

# Shared secret validation — requests must come from Next.js
@app.before_request
def check_internal_secret():
    if request.method == 'OPTIONS':
        return None
    if request.path == '/api/health':
        return None
    secret = os.getenv('INTERNAL_API_SECRET', '')
    if not secret:
        return None  # Secret not configured — skip enforcement (dev convenience)
    if request.headers.get('X-Internal-Secret') != secret:
        return jsonify({'error': 'Unauthorized', 'message': 'Missing or invalid internal secret'}), 401

# Initialize components
config = Config()
data_loader = DataLoader()
weather_api = WeatherAPI()
events_api = EventsAPI()
inventory_optimizer = InventoryOptimizer()

# Global storage for models (in production, use proper storage/database)
models_cache = {}
upload_models_cache = {}

@app.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        'message': 'Sales Forecasting API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'products': '/api/products',
            'data_summary': '/api/data/summary',
            'sales_data': '/api/data/sales',
            'train_model': '/api/forecast/train',
            'predict': '/api/forecast/predict',
            'inventory': '/api/inventory/recommendations',
            'weather': '/api/weather/current',
            'events': '/api/events/holidays',
            'upload_parse': '/api/upload/parse',
            'upload_train': '/api/upload/train',
            'upload_predict': '/api/upload/predict'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get list of perishable products"""
    return jsonify({
        'products': config.PERISHABLE_PRODUCTS
    })

@app.route('/api/data/summary', methods=['GET'])
def get_data_summary():
    """Get dataset summary statistics"""
    try:
        if data_loader.train_df is None:
            data_loader.load_data().preprocess_data()

        stats = data_loader.get_summary_stats()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/sales', methods=['POST'])
def get_sales_data():
    """Get sales data for a specific product"""
    try:
        params = request.json
        product_family = params.get('product_family', 'BREAD/BAKERY')
        store_nbr = params.get('store_nbr', None)
        start_date = params.get('start_date', None)
        end_date = params.get('end_date', None)

        if data_loader.train_df is None:
            data_loader.load_data().preprocess_data()

        # Get product data
        df = data_loader.get_product_data(product_family, store_nbr)

        # Filter by date if provided
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        # Convert to JSON-serializable format (NaN → null via pandas JSON encoder)
        df['date'] = df['date'].astype(str)
        result = json.loads(df.to_json(orient='records'))

        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/forecast/train', methods=['POST'])
def train_model():
    """Train a forecasting model"""
    try:
        params = request.json
        product_family = params.get('product_family', 'BREAD/BAKERY')
        model_type = params.get('model_type', 'prophet')  # arima, prophet, lstm
        store_nbr = params.get('store_nbr', None)

        if data_loader.train_df is None:
            data_loader.load_data().preprocess_data()

        # Get training data
        train_data = data_loader.get_product_data(product_family, store_nbr)
        train_data = train_data.sort_values('date')

        # Split train/test (80/20)
        split_idx = int(len(train_data) * 0.8)
        train_df = train_data.iloc[:split_idx]
        test_df = train_data.iloc[split_idx:]

        # Initialize and train model
        if model_type == 'arima':
            model = ARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        elif model_type == 'prophet':
            model = ProphetForecaster()
        elif model_type == 'lstm':
            model = LSTMForecaster(lookback=30, units=64, dropout=0.1, layers=2)
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown model type: {model_type}'
            }), 400

        # Train
        model.train(train_df, target_col='sales')

        # Evaluate
        metrics = model.evaluate(test_df, target_col='sales')

        # Cache model
        model_key = f"{product_family}_{model_type}"
        models_cache[model_key] = {
            'model': model,
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics
        }

        return jsonify({
            'success': True,
            'model_type': model_type,
            'product_family': product_family,
            'metrics': metrics,
            'trained_at': models_cache[model_key]['trained_at']
        })

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error training model: {error_msg}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/api/forecast/predict', methods=['POST'])
def predict():
    """Make forecast predictions"""
    try:
        params = request.json
        product_family = params.get('product_family', 'BREAD/BAKERY')
        model_type = params.get('model_type', 'prophet')
        steps = params.get('steps', 7)
        store_nbr = params.get('store_nbr', None)

        model_key = f"{product_family}_{model_type}"

        # Check if model is trained
        if model_key not in models_cache:
            return jsonify({
                'success': False,
                'error': 'Model not trained. Please train the model first.'
            }), 400

        model = models_cache[model_key]['model']

        # Make prediction
        if isinstance(model, LSTMForecaster):
            # LSTM needs last n days of data
            if data_loader.train_df is None:
                data_loader.load_data().preprocess_data()

            last_data = data_loader.get_product_data(product_family, store_nbr)
            last_data = last_data.sort_values('date').tail(model.lookback)

            forecast_df = model.predict(last_data, steps=steps)

            # Add dates
            last_date = last_data['date'].max()
            forecast_df['date'] = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=steps,
                freq='D'
            )
        else:
            forecast_df = model.predict(steps=steps)

        # Convert to JSON-serializable format (NaN → null via pandas JSON encoder)
        forecast_df['date'] = forecast_df['date'].astype(str)
        result = json.loads(forecast_df.to_json(orient='records'))

        return jsonify({
            'success': True,
            'forecast': result,
            'model_type': model_type,
            'product_family': product_family,
            'steps': steps
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/inventory/recommendations', methods=['POST'])
def get_inventory_recommendations():
    """Get inventory recommendations"""
    try:
        params = request.json
        product_family = params.get('product_family', 'BREAD/BAKERY')
        model_type = params.get('model_type', 'prophet')
        current_stock = params.get('current_stock', 0)
        store_nbr = params.get('store_nbr', None)

        model_key = f"{product_family}_{model_type}"

        # Check if model is trained
        if model_key not in models_cache:
            return jsonify({
                'success': False,
                'error': 'Model not trained. Please train the model first.'
            }), 400

        # Get forecast
        model = models_cache[model_key]['model']

        if isinstance(model, LSTMForecaster):
            if data_loader.train_df is None:
                data_loader.load_data().preprocess_data()

            last_data = data_loader.get_product_data(product_family, store_nbr)
            last_data = last_data.sort_values('date').tail(model.lookback)
            forecast_df = model.predict(last_data, steps=7)
        else:
            forecast_df = model.predict(steps=7)

        # Get historical data
        historical_data = data_loader.get_product_data(product_family, store_nbr)

        # Get recommendations
        recommendations = inventory_optimizer.get_recommendations(
            forecast_df,
            historical_data,
            product_family,
            current_stock=current_stock,
            lead_time=params.get('lead_time', 1),
            service_level=params.get('service_level', 0.95)
        )

        # Get alerts
        alerts = inventory_optimizer.get_alerts(recommendations)

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'alerts': alerts
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/weather/current', methods=['GET'])
def get_current_weather():
    """Get current weather"""
    try:
        city = request.args.get('city', 'Quito')
        country = request.args.get('country', 'EC')

        weather_data = weather_api.get_current_weather(city, country)

        # Convert datetime to string
        if 'timestamp' in weather_data:
            weather_data['timestamp'] = weather_data['timestamp'].isoformat()

        return jsonify({
            'success': True,
            'data': weather_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/weather/forecast', methods=['GET'])
def get_weather_forecast():
    """Get weather forecast"""
    try:
        city = request.args.get('city', 'Quito')
        country = request.args.get('country', 'EC')
        days = int(request.args.get('days', 7))

        forecast_df = weather_api.get_forecast(city, country, days)

        # Convert to JSON-serializable format (NaN → null via pandas JSON encoder)
        forecast_df['date'] = forecast_df['date'].astype(str)
        result = json.loads(forecast_df.to_json(orient='records'))

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/events/holidays', methods=['GET'])
def get_holidays():
    """Get holidays"""
    try:
        country = request.args.get('country', 'EC')
        year = request.args.get('year', None)

        if year:
            year = int(year)

        holidays_df = events_api.get_holidays(country, year)

        # Convert to JSON-serializable format (NaN → null via pandas JSON encoder)
        holidays_df['date'] = holidays_df['date'].astype(str)
        result = json.loads(holidays_df.to_json(orient='records'))

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/events/upcoming', methods=['GET'])
def get_upcoming_holidays():
    """Get upcoming holidays"""
    try:
        country = request.args.get('country', 'EC')
        days = int(request.args.get('days', 30))

        upcoming_df = events_api.get_upcoming_holidays(country, days)

        # Convert to JSON-serializable format (NaN → null via pandas JSON encoder)
        upcoming_df['date'] = upcoming_df['date'].astype(str)
        result = json.loads(upcoming_df.to_json(orient='records'))

        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/status', methods=['GET'])
def models_status():
    """Get status of all trained models."""
    return jsonify({
        'success': True,
        'total_models': len(models_cache),
        'models': [{'key': k, 'trained_at': v['trained_at']} for k, v in models_cache.items()]
    })

# ==================== Upload & Analyze Endpoints ====================

@app.route('/api/upload/parse', methods=['POST'])
def upload_parse():
    """Upload a file, parse it, detect columns, and return preview."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided.'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'success': False, 'error': 'Empty filename.'}), 400

        # Parse the file
        df, filename = parse_uploaded_file(file)

        # Store in memory
        session_id = store_upload(df, filename)

        # Detect columns
        column_info = detect_columns(df)

        # Create preview (first 20 rows)
        preview_df = df.head(20).copy()
        for col in preview_df.columns:
            if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
                preview_df[col] = preview_df[col].astype(str)

        preview = preview_df.fillna('').to_dict(orient='records')

        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'column_info': column_info,
            'preview': preview,
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/upload/train', methods=['POST'])
def upload_train():
    """Train a model on uploaded data with user-specified column mapping."""
    try:
        params = request.json
        session_id = params.get('session_id')
        date_col = params.get('date_column')
        target_col = params.get('target_column')
        model_type = params.get('model_type', 'prophet')

        if not session_id or not date_col or not target_col:
            return jsonify({
                'success': False,
                'error': 'session_id, date_column, and target_column are required.'
            }), 400

        # Retrieve uploaded data
        df, filename = get_upload(session_id)

        # Validate and prepare (returns tuple with info dict)
        prepared, prep_info = validate_and_prepare(df, date_col, target_col)

        # Check minimum rows for LSTM
        if model_type == 'lstm' and len(prepared) < 80:
            return jsonify({
                'success': False,
                'error': f'LSTM requires at least 80 rows of data. Your data has {len(prepared)} usable rows. Try Prophet or ARIMA instead.'
            }), 400

        # Split train/test (80/20)
        split_idx = int(len(prepared) * 0.8)
        train_df = prepared.iloc[:split_idx]
        test_df = prepared.iloc[split_idx:]

        # Initialize model
        if model_type == 'arima':
            model = ARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        elif model_type == 'prophet':
            model = ProphetForecaster()
        elif model_type == 'lstm':
            lookback = min(30, len(train_df) - 10)
            model = LSTMForecaster(lookback=lookback, units=64, dropout=0.1, layers=2)
        else:
            return jsonify({'success': False, 'error': f'Unknown model type: {model_type}'}), 400

        # Train
        model.train(train_df, target_col='sales')

        # Evaluate
        metrics = model.evaluate(test_df, target_col='sales')

        # Cache the model
        model_key = f"upload_{session_id}_{model_type}"
        upload_models_cache[model_key] = {
            'model': model,
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics,
            'prepared_data': prepared,
            'original_target_col': target_col,
        }

        # Serialize metrics (handle numpy types)
        safe_metrics = {}
        for k, v in metrics.items():
            try:
                safe_metrics[k] = float(v)
            except (TypeError, ValueError):
                safe_metrics[k] = str(v)

        return jsonify({
            'success': True,
            'model_type': model_type,
            'metrics': safe_metrics,
            'trained_at': upload_models_cache[model_key]['trained_at'],
            'rows_used': len(prepared),
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'aggregated': prep_info['aggregated'],
            'original_rows': prep_info['original_rows'],
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/upload/predict', methods=['POST'])
def upload_predict():
    """Make predictions using a model trained on uploaded data."""
    try:
        params = request.json
        session_id = params.get('session_id')
        model_type = params.get('model_type', 'prophet')
        steps = params.get('steps', 7)

        model_key = f"upload_{session_id}_{model_type}"

        if model_key not in upload_models_cache:
            return jsonify({
                'success': False,
                'error': 'Model not trained. Please train the model first.'
            }), 400

        cached = upload_models_cache[model_key]
        model = cached['model']
        prepared_data = cached['prepared_data']

        # Make prediction
        if isinstance(model, LSTMForecaster):
            last_data = prepared_data.sort_values('date').tail(model.lookback)
            forecast_df = model.predict(last_data, steps=steps)

            last_date = prepared_data['date'].max()
            if 'date' not in forecast_df.columns:
                forecast_df['date'] = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=steps,
                    freq='D'
                )
        else:
            forecast_df = model.predict(steps=steps)

        # Ensure date column is string (NaN → null via pandas JSON encoder)
        forecast_df['date'] = forecast_df['date'].astype(str)
        result = json.loads(forecast_df.to_json(orient='records'))

        # Also return last N rows of actual data for chart context
        tail_data = prepared_data.tail(min(30, len(prepared_data))).copy()
        tail_data['date'] = tail_data['date'].astype(str)
        historical = json.loads(tail_data.to_json(orient='records'))

        return jsonify({
            'success': True,
            'forecast': result,
            'historical': historical,
            'model_type': model_type,
            'steps': steps,
            'original_target_col': cached['original_target_col'],
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== Gemini AI Insight Endpoints ====================

@app.route('/api/insights/forecast-summary', methods=['POST'])
def insight_forecast_summary():
    """Generate a plain-English forecast summary using Gemini."""
    try:
        params = request.json or {}
        product_family = params.get('product_family', 'BREAD/BAKERY')
        model_type = params.get('model_type', 'prophet')
        metrics = params.get('metrics', {})
        predictions = params.get('predictions', [])

        text = forecast_summary(product_family, model_type, metrics, predictions)
        return jsonify({'success': True, 'insight': text})

    except RuntimeError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/insights/inventory-alert', methods=['POST'])
def insight_inventory_alert():
    """Generate a plain-English inventory action plan using Gemini."""
    try:
        params = request.json or {}
        product_family = params.get('product_family', 'BREAD/BAKERY')
        recommendations = params.get('recommendations', {})

        text = inventory_alert(product_family, recommendations)
        return jsonify({'success': True, 'insight': text})

    except RuntimeError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/insights/anomaly-explain', methods=['POST'])
def insight_anomaly_explain():
    """Generate a plain-English anomaly explanation using Gemini."""
    try:
        params = request.json or {}
        product_family = params.get('product_family', 'BREAD/BAKERY')
        date = params.get('date', '')
        actual = float(params.get('actual', 0))
        expected = float(params.get('expected', 0))
        holidays = params.get('holidays', [])
        weather = params.get('weather', {})

        text = anomaly_explain(product_family, date, actual, expected, holidays, weather)
        return jsonify({'success': True, 'insight': text})

    except RuntimeError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("Sales Forecasting API Server")
    print("="*60)
    print(f"Starting server on http://{config.HOST}:{config.PORT}")
    print("="*60)

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
