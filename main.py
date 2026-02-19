"""
Main entry point for the Sales Forecasting Backend API

Run this file to start the Flask server:
    python main.py

Or run from the api folder:
    cd api && python app.py
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.app import app, config

if __name__ == '__main__':
    print("="*60)
    print("Sales Forecasting Backend API")
    print("="*60)
    print(f"Server: http://{config.HOST}:{config.PORT}")
    print(f"Health Check: http://{config.HOST}:{config.PORT}/api/health")
    print("="*60)
    print("\nAvailable Endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/products - List perishable products")
    print("  GET  /api/data/summary - Dataset summary")
    print("  POST /api/data/sales - Get sales data")
    print("  POST /api/forecast/train - Train ML model")
    print("  POST /api/forecast/predict - Get predictions")
    print("  POST /api/inventory/recommendations - Get inventory recommendations")
    print("  GET  /api/weather/current - Current weather")
    print("  GET  /api/events/holidays - Get holidays")
    print("="*60)
    print("\nPress CTRL+C to stop the server\n")

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
