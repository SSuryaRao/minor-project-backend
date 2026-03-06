"""
Configuration settings for the Sales Forecasting System
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
    CALENDARIFIC_API_KEY = os.getenv('CALENDARIFIC_API_KEY', '')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

    # Internal shared secret (must match NEXT_PUBLIC_INTERNAL_SECRET in frontend)
    INTERNAL_API_SECRET = os.getenv('INTERNAL_API_SECRET', '')

    # Database
    DB_TYPE = os.getenv('DB_TYPE', 'sqlite')
    DB_NAME = os.getenv('DB_NAME', 'sales_forecast.db')

    # Model Settings
    MODEL_TYPE = os.getenv('MODEL_TYPE', 'prophet')
    FORECAST_DAYS = int(os.getenv('FORECAST_DAYS', 7))

    # Application
    DEBUG = os.getenv('DEBUG', 'True') == 'True'
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '0.0.0.0')

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

    # Perishable product families
    PERISHABLE_PRODUCTS = [
        'BREAD/BAKERY',
        'DAIRY',
        'DELI',
        'EGGS',
        'MEATS',
        'POULTRY',
        'SEAFOOD',
        'PRODUCE',
        'PREPARED FOODS'
    ]

    # Model hyperparameters
    ARIMA_ORDER = (1, 1, 1)
    LSTM_UNITS = 50
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32

    # Inventory settings
    SAFETY_STOCK_MULTIPLIER = 1.2
    SHELF_LIFE_DAYS = {
        'BREAD/BAKERY': 3,
        'DAIRY': 7,
        'DELI': 5,
        'EGGS': 14,
        'MEATS': 3,
        'POULTRY': 2,
        'SEAFOOD': 2,
        'PRODUCE': 5,
        'PREPARED FOODS': 2
    }
