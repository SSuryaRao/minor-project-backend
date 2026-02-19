"""
Quick Demo Script - Sales Forecasting System
Run this to see the system in action!
"""

import sys
sys.path.append('.')

from utils.data_loader import DataLoader
from models.prophet_model import ProphetForecaster
from utils.inventory_optimizer import InventoryOptimizer
import pandas as pd

def print_header(text):
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")

def main():
    print_header("SALES FORECASTING SYSTEM - QUICK DEMO")

    # Step 1: Load Data
    print_header("Step 1: Loading Data")
    loader = DataLoader()
    loader.load_data()
    loader.preprocess_data()

    stats = loader.get_summary_stats()
    print("Dataset Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Step 2: Get Product Data
    print_header("Step 2: Analyzing BREAD/BAKERY Product")
    product = 'BREAD/BAKERY'
    bread_data = loader.get_product_data(product)
    print(f"Total records: {len(bread_data)}")
    print(f"Date range: {bread_data['date'].min()} to {bread_data['date'].max()}")
    print(f"Average daily sales: {bread_data['sales'].mean():.2f} units")
    print(f"Total sales: {bread_data['sales'].sum():,.0f} units")

    # Step 3: Train Model
    print_header("Step 3: Training Prophet Model")
    # Use last 90 days for faster demo
    recent_data = bread_data.tail(365)  # Last year of data
    split = int(len(recent_data) * 0.8)
    train_data = recent_data.iloc[:split]
    test_data = recent_data.iloc[split:]

    print(f"Training data: {len(train_data)} records")
    print(f"Test data: {len(test_data)} records")
    print("\nTraining model... (this may take 10-30 seconds)")

    model = ProphetForecaster()
    model.train(train_data, target_col='sales')

    print("[OK] Model training completed!")

    # Step 4: Evaluate Model
    print_header("Step 4: Evaluating Model Performance")
    metrics = model.evaluate(test_data, target_col='sales')

    print("Performance Metrics:")
    print(f"  MAE (Mean Absolute Error): {metrics['MAE']:.2f} units")
    print(f"  RMSE (Root Mean Squared Error): {metrics['RMSE']:.2f} units")
    print(f"  MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.2f}%")

    if metrics['MAPE'] < 15:
        print("\n[OK] Excellent accuracy! (Industry standard: <25%)")
    elif metrics['MAPE'] < 25:
        print("\n[OK] Good accuracy for retail forecasting")
    else:
        print("\n[WARNING] Fair accuracy - consider more training data")

    # Step 5: Generate Forecast
    print_header("Step 5: Generating 7-Day Forecast")
    forecast = model.predict(steps=7)

    print("Forecast for next 7 days:")
    print("-" * 70)
    for idx, row in forecast.iterrows():
        date = row['date']
        pred = row['forecast']
        lower = row.get('lower_bound', pred * 0.9)
        upper = row.get('upper_bound', pred * 1.1)

        print(f"{date}: {pred:.0f} units (range: {lower:.0f} - {upper:.0f})")

    # Step 6: Inventory Recommendations
    print_header("Step 6: Inventory Recommendations")

    # Simulate current stock
    current_stock = 100

    optimizer = InventoryOptimizer()
    recommendations = optimizer.get_recommendations(
        forecast_df=forecast,
        historical_data=bread_data,
        product_family=product,
        current_stock=current_stock,
        service_level=0.95,
        lead_time=1
    )

    print(f"Product: {product}")
    print(f"Current Stock: {current_stock} units")
    print(f"\nRecommendations:")
    print(f"  Stock Status: {recommendations['stock_status']}")
    print(f"  Days of Stock: {recommendations['days_of_stock']:.1f} days")
    print(f"  Reorder Point: {recommendations['reorder_point']:.0f} units")
    print(f"  Recommended Order: {recommendations['order_quantity']:.0f} units")
    print(f"  Action: {recommendations['order_recommendation']}")
    print(f"\nRisk Assessment:")
    print(f"  Wastage Risk: {recommendations['wastage_risk']}")
    print(f"  Estimated Waste: {recommendations['estimated_waste']:.0f} units")
    print(f"  Shelf Life: {recommendations['shelf_life_days']} days")

    # Step 7: Alerts
    print_header("Step 7: Alerts and Notifications")
    alerts = optimizer.get_alerts(recommendations)

    if alerts:
        for alert in alerts:
            print(f"\n[{alert['type']}] {alert['title']}")
            print(f"  {alert['message']}")
            print(f"  Action: {alert['action']}")
    else:
        print("[OK] No alerts - inventory levels are optimal!")

    # Summary
    print_header("DEMO COMPLETED SUCCESSFULLY!")

    print("What you just saw:")
    print("  [OK] Data loading and preprocessing")
    print("  [OK] Prophet model training and evaluation")
    print("  [OK] 7-day sales forecast generation")
    print("  [OK] Inventory optimization recommendations")
    print("  [OK] Risk assessment and alerts")

    print("\nNext Steps:")
    print("  1. Run 'python api/app.py' to start the web dashboard")
    print("  2. Open http://localhost:5000 in your browser")
    print("  3. Explore different products and models")
    print("  4. Try the inventory recommendations feature")

    print("\nFiles Created:")
    print("  • All models and utilities are ready to use")
    print("  • Check notebooks/01_exploratory_data_analysis.ipynb for detailed analysis")
    print("  • Read SETUP_GUIDE.md for complete documentation")

    print("\n" + "="*70)
    print(" Thank you for using the Sales Forecasting System!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        print("\nMake sure:")
        print("  1. All dependencies are installed: pip install -r requirements.txt")
        print("  2. Dataset files are in the 'data/' folder")
        print("  3. You're running from the project root directory")
