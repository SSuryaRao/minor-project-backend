import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_sales_data(start_date='2023-01-01', num_days=365):
    """
    Generate synthetic sales data for a small bakery/kirana store
    Products: Milk, Bread, Fruits, Vegetables
    """

    dates = pd.date_range(start=start_date, periods=num_days, freq='D')

    # Define products with base sales
    products = {
        'Milk': {'base': 50, 'std': 10},
        'Bread': {'base': 40, 'std': 8},
        'Fruits': {'base': 30, 'std': 12},
        'Vegetables': {'base': 35, 'std': 10}
    }

    data = []

    for date in dates:
        day_of_week = date.dayofweek  # Monday=0, Sunday=6
        month = date.month
        is_weekend = day_of_week >= 5

        # Festivals/Events (Indian calendar approximate)
        is_festival = False
        festival_name = None

        # Major festivals (approximate dates)
        if (date.month == 1 and date.day == 26):  # Republic Day
            is_festival = True
            festival_name = "Republic Day"
        elif (date.month == 3 and 8 <= date.day <= 10):  # Holi
            is_festival = True
            festival_name = "Holi"
        elif (date.month == 8 and date.day == 15):  # Independence Day
            is_festival = True
            festival_name = "Independence Day"
        elif (date.month == 10 and 23 <= date.day <= 25):  # Diwali
            is_festival = True
            festival_name = "Diwali"
        elif (date.month == 11 and 12 <= date.day <= 14):  # Diwali period
            is_festival = True
            festival_name = "Diwali Period"
        elif day_of_week == 6:  # Sunday - local market day
            if random.random() < 0.3:
                festival_name = "Local Market Day"

        # Weather simulation (simplified)
        # Temperature: Higher in summer (Apr-Jun), moderate in winter
        if month in [4, 5, 6]:  # Summer
            temp = np.random.normal(35, 3)
            rainfall = np.random.choice([0, 0, 0, 5, 10], p=[0.7, 0.15, 0.1, 0.03, 0.02])
            weather = 'Hot'
        elif month in [7, 8, 9]:  # Monsoon
            temp = np.random.normal(28, 2)
            rainfall = np.random.choice([0, 10, 20, 50, 100], p=[0.3, 0.3, 0.2, 0.15, 0.05])
            weather = 'Rainy' if rainfall > 10 else 'Cloudy'
        elif month in [12, 1, 2]:  # Winter
            temp = np.random.normal(18, 3)
            rainfall = np.random.choice([0, 0, 5], p=[0.85, 0.1, 0.05])
            weather = 'Cold'
        else:  # Spring/Autumn
            temp = np.random.normal(25, 3)
            rainfall = np.random.choice([0, 0, 5, 10], p=[0.7, 0.2, 0.08, 0.02])
            weather = 'Pleasant'

        # Generate sales for each product
        for product, params in products.items():
            base_sales = params['base']
            std_dev = params['std']

            # Apply multipliers based on various factors
            multiplier = 1.0

            # Weekend effect (10-20% increase)
            if is_weekend:
                multiplier *= np.random.uniform(1.1, 1.2)

            # Festival effect (30-80% increase)
            if is_festival:
                multiplier *= np.random.uniform(1.3, 1.8)

            # Month-end effect (payday - 15-25% increase)
            if date.day >= 28:
                multiplier *= np.random.uniform(1.15, 1.25)

            # Weather effect
            if weather == 'Rainy' and rainfall > 20:
                # Rain reduces sales for fruits/vegetables, increases milk/bread
                if product in ['Fruits', 'Vegetables']:
                    multiplier *= np.random.uniform(0.7, 0.85)
                else:
                    multiplier *= np.random.uniform(1.05, 1.15)
            elif weather == 'Hot':
                # Hot weather increases fruit sales, decreases bread
                if product == 'Fruits':
                    multiplier *= np.random.uniform(1.1, 1.3)
                elif product == 'Bread':
                    multiplier *= np.random.uniform(0.9, 0.95)

            # Seasonal trends
            if month in [4, 5, 6] and product == 'Vegetables':
                multiplier *= 0.9  # Less fresh vegetables in summer
            if month in [10, 11, 12] and product in ['Milk', 'Vegetables']:
                multiplier *= 1.1  # Festival season demand

            # Calculate final sales with some randomness
            sales = int(base_sales * multiplier + np.random.normal(0, std_dev))
            sales = max(0, sales)  # No negative sales

            # Add row to data
            data.append({
                'date': date,
                'product': product,
                'sales_quantity': sales,
                'day_of_week': day_of_week,
                'is_weekend': int(is_weekend),
                'is_festival': int(is_festival),
                'festival_name': festival_name if festival_name else 'None',
                'temperature': round(temp, 1),
                'rainfall': rainfall,
                'weather': weather,
                'month': month,
                'day': date.day,
                'week_of_year': date.isocalendar()[1]
            })

    df = pd.DataFrame(data)
    return df

# Generate data
print("Generating synthetic sales data...")
df = generate_sales_data(start_date='2023-01-01', num_days=365)

# Save to CSV
df.to_csv('sales_data.csv', index=False)
print(f"Generated {len(df)} records")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Products: {df['product'].unique()}")
print("\nData sample:")
print(df.head(10))
print("\nData statistics:")
print(df.groupby('product')['sales_quantity'].describe())
print("\nData saved to 'sales_data.csv'")
