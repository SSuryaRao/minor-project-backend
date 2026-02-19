"""
Inventory Optimization and Recommendation System
"""
import pandas as pd
import numpy as np
from config.config import Config

class InventoryOptimizer:
    """Optimize inventory based on demand forecasts"""

    def __init__(self):
        self.config = Config()
        self.recommendations = None

    def calculate_safety_stock(self, avg_demand, std_demand, lead_time=1, service_level=0.95):
        """
        Calculate safety stock using statistical method

        Parameters:
        -----------
        avg_demand : float
            Average daily demand
        std_demand : float
            Standard deviation of demand
        lead_time : int
            Lead time in days
        service_level : float
            Desired service level (0-1)

        Returns:
        --------
        float : safety stock quantity
        """
        from scipy import stats

        # Z-score for service level
        z_score = stats.norm.ppf(service_level)

        # Safety stock formula
        safety_stock = z_score * std_demand * np.sqrt(lead_time)

        return max(0, safety_stock)

    def calculate_reorder_point(self, avg_demand, lead_time, safety_stock):
        """
        Calculate reorder point

        Parameters:
        -----------
        avg_demand : float
            Average daily demand
        lead_time : int
            Lead time in days
        safety_stock : float
            Safety stock quantity

        Returns:
        --------
        float : reorder point
        """
        reorder_point = (avg_demand * lead_time) + safety_stock
        return reorder_point

    def calculate_economic_order_quantity(self, annual_demand, ordering_cost=50, holding_cost_per_unit=2):
        """
        Calculate Economic Order Quantity (EOQ)

        Parameters:
        -----------
        annual_demand : float
            Annual demand quantity
        ordering_cost : float
            Cost per order
        holding_cost_per_unit : float
            Annual holding cost per unit

        Returns:
        --------
        float : optimal order quantity
        """
        if holding_cost_per_unit <= 0:
            return annual_demand / 12  # Monthly demand as fallback

        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        return eoq

    def get_recommendations(self, forecast_df, historical_data, product_family, **kwargs):
        """
        Generate inventory recommendations

        Parameters:
        -----------
        forecast_df : pd.DataFrame
            Forecast data with 'forecast' column
        historical_data : pd.DataFrame
            Historical sales data
        product_family : str
            Product family name
        **kwargs : additional parameters

        Returns:
        --------
        dict with inventory recommendations
        """
        # Extract parameters
        lead_time = kwargs.get('lead_time', 1)
        service_level = kwargs.get('service_level', 0.95)
        current_stock = kwargs.get('current_stock', 0)
        ordering_cost = kwargs.get('ordering_cost', 50)
        holding_cost = kwargs.get('holding_cost', 2)

        # Calculate statistics from historical data
        if 'sales' in historical_data.columns:
            avg_demand = historical_data['sales'].mean()
            std_demand = historical_data['sales'].std()
            annual_demand = historical_data['sales'].sum() * (365 / len(historical_data))
        else:
            raise ValueError("Historical data must contain 'sales' column")

        # Get forecast statistics
        forecast_avg = forecast_df['forecast'].mean()
        forecast_total = forecast_df['forecast'].sum()

        # Calculate safety stock
        safety_stock = self.calculate_safety_stock(
            avg_demand,
            std_demand,
            lead_time,
            service_level
        )

        # Calculate reorder point
        reorder_point = self.calculate_reorder_point(
            forecast_avg,
            lead_time,
            safety_stock
        )

        # Calculate EOQ
        eoq = self.calculate_economic_order_quantity(
            annual_demand,
            ordering_cost,
            holding_cost
        )

        # Get shelf life for product
        shelf_life = self.config.SHELF_LIFE_DAYS.get(product_family, 7)

        # Calculate maximum inventory (considering shelf life)
        max_inventory = forecast_avg * shelf_life * self.config.SAFETY_STOCK_MULTIPLIER

        # Calculate order quantity
        if current_stock < reorder_point:
            order_quantity = max(eoq, reorder_point - current_stock)
            # Don't order more than can be sold within shelf life
            order_quantity = min(order_quantity, max_inventory)
            order_recommendation = "ORDER NOW"
        else:
            order_quantity = 0
            order_recommendation = "NO ORDER NEEDED"

        # Calculate stock status
        days_of_stock = current_stock / forecast_avg if forecast_avg > 0 else 0

        if days_of_stock < 1:
            stock_status = "CRITICAL"
        elif days_of_stock < 3:
            stock_status = "LOW"
        elif days_of_stock < shelf_life:
            stock_status = "ADEQUATE"
        else:
            stock_status = "OVERSTOCKED"

        # Wastage risk
        if days_of_stock > shelf_life:
            wastage_risk = "HIGH"
            estimated_waste = current_stock - (forecast_avg * shelf_life)
        elif days_of_stock > shelf_life * 0.8:
            wastage_risk = "MEDIUM"
            estimated_waste = 0
        else:
            wastage_risk = "LOW"
            estimated_waste = 0

        recommendations = {
            'product_family': product_family,
            'forecast_avg_daily': round(forecast_avg, 2),
            'forecast_total_period': round(forecast_total, 2),
            'historical_avg_daily': round(avg_demand, 2),
            'historical_std': round(std_demand, 2),
            'current_stock': current_stock,
            'safety_stock': round(safety_stock, 2),
            'reorder_point': round(reorder_point, 2),
            'eoq': round(eoq, 2),
            'max_inventory': round(max_inventory, 2),
            'order_quantity': round(order_quantity, 2),
            'order_recommendation': order_recommendation,
            'days_of_stock': round(days_of_stock, 2),
            'stock_status': stock_status,
            'wastage_risk': wastage_risk,
            'estimated_waste': round(max(0, estimated_waste), 2),
            'shelf_life_days': shelf_life,
            'service_level': service_level
        }

        self.recommendations = recommendations
        return recommendations

    def print_recommendations(self, recommendations=None):
        """Print recommendations in a formatted way"""
        if recommendations is None:
            recommendations = self.recommendations

        if recommendations is None:
            print("No recommendations available")
            return

        print("\n" + "="*60)
        print(f"INVENTORY RECOMMENDATIONS: {recommendations['product_family']}")
        print("="*60)

        print("\n📊 DEMAND FORECAST:")
        print(f"  • Average Daily Forecast: {recommendations['forecast_avg_daily']:.2f} units")
        print(f"  • Total Period Forecast: {recommendations['forecast_total_period']:.2f} units")
        print(f"  • Historical Average: {recommendations['historical_avg_daily']:.2f} units")

        print("\n📦 CURRENT INVENTORY:")
        print(f"  • Current Stock: {recommendations['current_stock']:.2f} units")
        print(f"  • Days of Stock: {recommendations['days_of_stock']:.1f} days")
        print(f"  • Stock Status: {recommendations['stock_status']}")

        print("\n🎯 OPTIMAL LEVELS:")
        print(f"  • Safety Stock: {recommendations['safety_stock']:.2f} units")
        print(f"  • Reorder Point: {recommendations['reorder_point']:.2f} units")
        print(f"  • Economic Order Quantity: {recommendations['eoq']:.2f} units")
        print(f"  • Maximum Inventory: {recommendations['max_inventory']:.2f} units")

        print("\n📋 RECOMMENDATIONS:")
        print(f"  • Action: {recommendations['order_recommendation']}")
        print(f"  • Order Quantity: {recommendations['order_quantity']:.2f} units")

        print("\n⚠️  RISK ASSESSMENT:")
        print(f"  • Wastage Risk: {recommendations['wastage_risk']}")
        print(f"  • Estimated Waste: {recommendations['estimated_waste']:.2f} units")
        print(f"  • Shelf Life: {recommendations['shelf_life_days']} days")
        print(f"  • Service Level: {recommendations['service_level']*100:.0f}%")

        print("\n" + "="*60)

    def get_alerts(self, recommendations=None):
        """
        Generate alerts based on recommendations

        Parameters:
        -----------
        recommendations : dict
            Inventory recommendations

        Returns:
        --------
        list of alerts
        """
        if recommendations is None:
            recommendations = self.recommendations

        if recommendations is None:
            return []

        alerts = []

        # Critical stock alert
        if recommendations['stock_status'] == 'CRITICAL':
            alerts.append({
                'type': 'CRITICAL',
                'title': 'Critical Stock Level',
                'message': f"{recommendations['product_family']}: Only {recommendations['days_of_stock']:.1f} days of stock remaining!",
                'action': f"Order {recommendations['order_quantity']:.0f} units immediately"
            })

        # Low stock alert
        elif recommendations['stock_status'] == 'LOW':
            alerts.append({
                'type': 'WARNING',
                'title': 'Low Stock Level',
                'message': f"{recommendations['product_family']}: Stock running low ({recommendations['days_of_stock']:.1f} days remaining)",
                'action': f"Consider ordering {recommendations['order_quantity']:.0f} units"
            })

        # Overstocked alert
        elif recommendations['stock_status'] == 'OVERSTOCKED':
            alerts.append({
                'type': 'WARNING',
                'title': 'Overstock Alert',
                'message': f"{recommendations['product_family']}: Excess inventory ({recommendations['days_of_stock']:.1f} days of stock)",
                'action': "Reduce next order or promote product"
            })

        # Wastage risk alert
        if recommendations['wastage_risk'] == 'HIGH':
            alerts.append({
                'type': 'DANGER',
                'title': 'High Wastage Risk',
                'message': f"{recommendations['product_family']}: {recommendations['estimated_waste']:.0f} units may expire",
                'action': "Reduce prices or bundle with other products"
            })

        # Reorder alert
        if recommendations['order_recommendation'] == 'ORDER NOW':
            alerts.append({
                'type': 'INFO',
                'title': 'Reorder Point Reached',
                'message': f"{recommendations['product_family']}: Time to reorder",
                'action': f"Place order for {recommendations['order_quantity']:.0f} units"
            })

        return alerts

    def batch_recommendations(self, forecasts_dict, historical_data_dict, **kwargs):
        """
        Generate recommendations for multiple products

        Parameters:
        -----------
        forecasts_dict : dict
            Dictionary of {product_family: forecast_df}
        historical_data_dict : dict
            Dictionary of {product_family: historical_df}
        **kwargs : additional parameters

        Returns:
        --------
        pd.DataFrame with all recommendations
        """
        all_recommendations = []

        for product_family in forecasts_dict.keys():
            if product_family in historical_data_dict:
                try:
                    recommendations = self.get_recommendations(
                        forecasts_dict[product_family],
                        historical_data_dict[product_family],
                        product_family,
                        **kwargs
                    )
                    all_recommendations.append(recommendations)
                except Exception as e:
                    print(f"Error processing {product_family}: {str(e)}")

        recommendations_df = pd.DataFrame(all_recommendations)
        return recommendations_df
