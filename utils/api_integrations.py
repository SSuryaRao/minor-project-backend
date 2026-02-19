"""
External API Integrations (Weather and Events)
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from config.config import Config

class WeatherAPI:
    """OpenWeatherMap API integration"""

    def __init__(self, api_key=None):
        self.api_key = api_key or Config.OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_current_weather(self, city="Quito", country_code="EC"):
        """
        Get current weather for a city

        Parameters:
        -----------
        city : str
            City name
        country_code : str
            Country code (ISO 3166)

        Returns:
        --------
        dict with weather data
        """
        if not self.api_key or self.api_key == 'your_api_key_here':
            print("Warning: OpenWeatherMap API key not configured")
            return self._get_mock_weather()

        url = f"{self.base_url}/weather"
        params = {
            'q': f"{city},{country_code}",
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            weather_data = {
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'clouds': data['clouds']['all'],
                'timestamp': datetime.fromtimestamp(data['dt'])
            }

            return weather_data

        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return self._get_mock_weather()

    def get_forecast(self, city="Quito", country_code="EC", days=7):
        """
        Get weather forecast

        Parameters:
        -----------
        city : str
            City name
        country_code : str
            Country code
        days : int
            Number of days (max 5 for free tier)

        Returns:
        --------
        pd.DataFrame with forecast data
        """
        if not self.api_key or self.api_key == 'your_api_key_here':
            print("Warning: OpenWeatherMap API key not configured")
            return self._get_mock_forecast(days)

        url = f"{self.base_url}/forecast"
        params = {
            'q': f"{city},{country_code}",
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            forecast_list = []
            for item in data['list'][:days*8]:  # 8 forecasts per day (3-hour intervals)
                forecast_list.append({
                    'date': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'weather': item['weather'][0]['main'],
                    'description': item['weather'][0]['description'],
                    'wind_speed': item['wind']['speed'],
                    'rain_probability': item.get('pop', 0) * 100
                })

            forecast_df = pd.DataFrame(forecast_list)

            # Aggregate by day
            forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date
            daily_forecast = forecast_df.groupby('date').agg({
                'temperature': 'mean',
                'humidity': 'mean',
                'wind_speed': 'mean',
                'rain_probability': 'max'
            }).reset_index()

            return daily_forecast

        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast data: {e}")
            return self._get_mock_forecast(days)

    def _get_mock_weather(self):
        """Return mock weather data for testing"""
        return {
            'temperature': 20.0,
            'feels_like': 19.5,
            'humidity': 65,
            'pressure': 1013,
            'weather': 'Clear',
            'description': 'clear sky',
            'wind_speed': 3.5,
            'clouds': 10,
            'timestamp': datetime.now()
        }

    def _get_mock_forecast(self, days):
        """Return mock forecast data for testing"""
        dates = [(datetime.now() + timedelta(days=i)).date() for i in range(1, days+1)]
        forecast_data = {
            'date': dates,
            'temperature': [20 + i for i in range(days)],
            'humidity': [65 - i for i in range(days)],
            'wind_speed': [3.5 + i*0.2 for i in range(days)],
            'rain_probability': [10 + i*5 for i in range(days)]
        }
        return pd.DataFrame(forecast_data)


class EventsAPI:
    """Calendarific API integration for holidays and events"""

    def __init__(self, api_key=None):
        self.api_key = api_key or Config.CALENDARIFIC_API_KEY
        self.base_url = "https://calendarific.com/api/v2"

    def get_holidays(self, country="EC", year=None):
        """
        Get holidays for a country and year

        Parameters:
        -----------
        country : str
            Country code (ISO 3166)
        year : int
            Year (default: current year)

        Returns:
        --------
        pd.DataFrame with holidays
        """
        if year is None:
            year = datetime.now().year

        if not self.api_key or self.api_key == 'your_api_key_here':
            print("Warning: Calendarific API key not configured")
            return self._get_mock_holidays(year)

        url = f"{self.base_url}/holidays"
        params = {
            'api_key': self.api_key,
            'country': country,
            'year': year
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data['meta']['code'] != 200:
                print(f"API Error: {data['meta']['error_detail']}")
                return self._get_mock_holidays(year)

            holidays_list = []
            for holiday in data['response']['holidays']:
                holidays_list.append({
                    'date': holiday['date']['iso'],
                    'name': holiday['name'],
                    'description': holiday['description'],
                    'type': ', '.join(holiday['type']),
                    'country': country
                })

            holidays_df = pd.DataFrame(holidays_list)
            holidays_df['date'] = pd.to_datetime(holidays_df['date'])

            return holidays_df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching holidays data: {e}")
            return self._get_mock_holidays(year)

    def get_upcoming_holidays(self, country="EC", days=30):
        """
        Get upcoming holidays

        Parameters:
        -----------
        country : str
            Country code
        days : int
            Number of days to look ahead

        Returns:
        --------
        pd.DataFrame with upcoming holidays
        """
        current_year = datetime.now().year
        next_year = current_year + 1

        # Get holidays for current and next year
        holidays_current = self.get_holidays(country, current_year)
        holidays_next = self.get_holidays(country, next_year)

        all_holidays = pd.concat([holidays_current, holidays_next], ignore_index=True)

        # Filter upcoming holidays
        today = datetime.now().date()
        end_date = today + timedelta(days=days)

        all_holidays['date'] = pd.to_datetime(all_holidays['date'])
        upcoming = all_holidays[
            (all_holidays['date'].dt.date >= today) &
            (all_holidays['date'].dt.date <= end_date)
        ]

        return upcoming.sort_values('date').reset_index(drop=True)

    def _get_mock_holidays(self, year):
        """Return mock holidays data for testing"""
        holidays_data = {
            'date': [
                f'{year}-01-01',
                f'{year}-05-01',
                f'{year}-08-10',
                f'{year}-10-09',
                f'{year}-11-02',
                f'{year}-11-03',
                f'{year}-12-25'
            ],
            'name': [
                "New Year's Day",
                "Labour Day",
                "Independence Day",
                "Independence of Guayaquil",
                "All Souls' Day",
                "Independence of Cuenca",
                "Christmas Day"
            ],
            'description': [
                "New Year's Day",
                "Labour Day",
                "First Cry of Independence",
                "Independence of Guayaquil",
                "All Souls' Day",
                "Independence of Cuenca",
                "Christmas Day"
            ],
            'type': ['National holiday'] * 7,
            'country': ['EC'] * 7
        }

        holidays_df = pd.DataFrame(holidays_data)
        holidays_df['date'] = pd.to_datetime(holidays_df['date'])

        return holidays_df


class ExternalDataIntegrator:
    """Integrate weather and events data with sales data"""

    def __init__(self):
        self.weather_api = WeatherAPI()
        self.events_api = EventsAPI()

    def enrich_sales_data(self, sales_df, city="Quito", country="EC"):
        """
        Enrich sales data with weather and holiday information

        Parameters:
        -----------
        sales_df : pd.DataFrame
            Sales data with 'date' column
        city : str
            City name for weather data
        country : str
            Country code for holidays

        Returns:
        --------
        pd.DataFrame with enriched data
        """
        enriched_df = sales_df.copy()

        # Get year range from sales data
        enriched_df['date'] = pd.to_datetime(enriched_df['date'])
        years = enriched_df['date'].dt.year.unique()

        # Get holidays for all years
        all_holidays = pd.DataFrame()
        for year in years:
            holidays = self.events_api.get_holidays(country, year)
            all_holidays = pd.concat([all_holidays, holidays], ignore_index=True)

        # Merge holidays
        enriched_df['date_only'] = enriched_df['date'].dt.date
        all_holidays['date_only'] = all_holidays['date'].dt.date

        enriched_df = enriched_df.merge(
            all_holidays[['date_only', 'name']],
            on='date_only',
            how='left'
        )

        enriched_df['is_holiday'] = enriched_df['name'].notna().astype(int)
        enriched_df = enriched_df.drop('date_only', axis=1)
        enriched_df = enriched_df.rename(columns={'name': 'holiday_name'})

        print(f"Enriched {len(enriched_df)} records with external data")
        print(f"Found {enriched_df['is_holiday'].sum()} holiday records")

        return enriched_df

    def get_forecast_features(self, days=7, city="Quito"):
        """
        Get weather and holiday features for future dates

        Parameters:
        -----------
        days : int
            Number of days to forecast
        city : str
            City name

        Returns:
        --------
        pd.DataFrame with forecast features
        """
        # Get weather forecast
        weather_forecast = self.weather_api.get_forecast(city, days=days)

        # Get upcoming holidays
        upcoming_holidays = self.events_api.get_upcoming_holidays(days=days)

        # Merge
        forecast_features = weather_forecast.copy()
        forecast_features['date'] = pd.to_datetime(forecast_features['date'])

        if not upcoming_holidays.empty:
            upcoming_holidays['date_only'] = upcoming_holidays['date'].dt.date
            forecast_features['date_only'] = forecast_features['date'].dt.date

            forecast_features = forecast_features.merge(
                upcoming_holidays[['date_only', 'name']],
                on='date_only',
                how='left'
            )

            forecast_features['is_holiday'] = forecast_features['name'].notna().astype(int)
            forecast_features = forecast_features.drop('date_only', axis=1)
            forecast_features = forecast_features.rename(columns={'name': 'holiday_name'})
        else:
            forecast_features['is_holiday'] = 0
            forecast_features['holiday_name'] = None

        return forecast_features
