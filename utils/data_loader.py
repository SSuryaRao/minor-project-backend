"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from config.config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()
        self.train_df = None
        self.stores_df = None
        self.holidays_df = None
        self.oil_df = None
        self.transactions_df = None
        self.preprocessed = False

    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        self.train_df = pd.read_csv(f"{self.config.DATA_DIR}/train.csv")
        self.stores_df = pd.read_csv(f"{self.config.DATA_DIR}/stores.csv")
        self.holidays_df = pd.read_csv(f"{self.config.DATA_DIR}/holidays_events.csv")
        self.oil_df = pd.read_csv(f"{self.config.DATA_DIR}/oil.csv")
        self.transactions_df = pd.read_csv(f"{self.config.DATA_DIR}/transactions.csv")

        print(f"Train data shape: {self.train_df.shape}")
        print(f"Date range: {self.train_df['date'].min()} to {self.train_df['date'].max()}")
        return self

    def preprocess_data(self):
        """Preprocess and clean data"""
        if self.preprocessed:
            print("Data already preprocessed, skipping...")
            return self

        print("\nPreprocessing data...")

        # Convert date columns to datetime
        self.train_df['date'] = pd.to_datetime(self.train_df['date'])
        self.holidays_df['date'] = pd.to_datetime(self.holidays_df['date'])
        self.oil_df['date'] = pd.to_datetime(self.oil_df['date'])
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])

        # Handle missing values in oil prices
        self.oil_df['dcoilwtico'] = self.oil_df['dcoilwtico'].ffill()

        # Merge store information
        self.train_df = self.train_df.merge(self.stores_df, on='store_nbr', how='left')

        # Merge oil prices
        self.train_df = self.train_df.merge(self.oil_df, on='date', how='left')

        # Merge transactions
        self.train_df = self.train_df.merge(
            self.transactions_df,
            on=['date', 'store_nbr'],
            how='left'
        )

        self.preprocessed = True
        print("Data preprocessing completed!")
        return self

    def get_perishable_data(self):
        """Filter only perishable products"""
        perishable_df = self.train_df[
            self.train_df['family'].isin(self.config.PERISHABLE_PRODUCTS)
        ].copy()
        print(f"\nPerishable products data shape: {perishable_df.shape}")
        return perishable_df

    def get_product_data(self, product_family, store_nbr=None):
        """Get data for specific product family and optionally specific store"""
        df = self.train_df[self.train_df['family'] == product_family].copy()

        if store_nbr is not None:
            df = df[df['store_nbr'] == store_nbr].copy()
        else:
            # Aggregate across all stores
            df = df.groupby('date').agg({
                'sales': 'sum',
                'onpromotion': 'sum',
                'dcoilwtico': 'mean',
                'transactions': 'sum'
            }).reset_index()

        return df

    def add_time_features(self, df):
        """Add time-based features"""
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        return df

    def add_holiday_features(self, df):
        """Add holiday features"""
        df = df.copy()

        # Create holiday indicator
        national_holidays = self.holidays_df[
            self.holidays_df['locale'] == 'National'
        ]['date'].unique()

        df['is_holiday'] = df['date'].isin(national_holidays).astype(int)

        # Days to/from holiday
        df = df.sort_values('date')
        df['days_to_holiday'] = 0
        df['days_from_holiday'] = 0

        for holiday in national_holidays:
            mask = df['date'] < holiday
            days_diff = (holiday - df.loc[mask, 'date']).dt.days
            df.loc[mask, 'days_to_holiday'] = days_diff.where(
                days_diff <= 7,
                df.loc[mask, 'days_to_holiday']
            )

        return df

    def create_lag_features(self, df, target_col='sales', lags=[1, 7, 14, 30]):
        """Create lag features for time series"""
        df = df.copy()
        df = df.sort_values('date')

        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()

        return df

    def get_summary_stats(self):
        """Get summary statistics"""
        if self.train_df is None:
            return "Data not loaded. Call load_data() first."

        stats = {
            'total_records': int(len(self.train_df)),
            'date_range': {
                'start': str(self.train_df['date'].min().date()),
                'end': str(self.train_df['date'].max().date())
            },
            'n_stores': int(self.train_df['store_nbr'].nunique()),
            'n_families': int(self.train_df['family'].nunique()),
            'total_sales': float(self.train_df['sales'].sum()),
            'avg_daily_sales': float(self.train_df['sales'].mean()),
            'perishable_families': len(self.config.PERISHABLE_PRODUCTS)
        }
        return stats
