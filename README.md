---
title: salesapi
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Sales Forecasting Backend API

Flask-based REST API for sales forecasting and inventory optimization using Machine Learning.

## Features

- **Multiple ML Models**: ARIMA, Prophet, LSTM
- **Sales Forecasting**: 7-day predictions with confidence intervals
- **Inventory Optimization**: Safety stock, reorder points, EOQ calculations
- **External Data Integration**: Weather and holiday data
- **RESTful API**: JSON-based endpoints for easy integration

## Tech Stack

- **Flask** - Web framework
- **Prophet** - Time series forecasting
- **TensorFlow/Keras** - LSTM models
- **Statsmodels** - ARIMA models
- **Pandas/NumPy** - Data processing
- **Flask-CORS** - Cross-origin support

## Installation

1. Create a virtual environment:
```bash
cd backend
python -m venv .venv
```

2. Activate virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Copy .env file and update with your API keys
# (Optional - system works without external APIs using mock data)
```

## Running the Server

```bash
python main.py
```

Server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```http
GET /api/health
```

### Products
```http
GET /api/products
```

### Data Summary
```http
GET /api/data/summary
```

### Sales Data
```http
POST /api/data/sales
Content-Type: application/json

{
  "product_family": "BREAD/BAKERY",
  "store_nbr": 1,
  "start_date": "2013-01-01",
  "end_date": "2017-08-31"
}
```

### Train Model
```http
POST /api/forecast/train
Content-Type: application/json

{
  "product_family": "BREAD/BAKERY",
  "model_type": "prophet",
  "store_nbr": 1
}
```

### Get Predictions
```http
POST /api/forecast/predict
Content-Type: application/json

{
  "product_family": "BREAD/BAKERY",
  "model_type": "prophet",
  "steps": 7
}
```

### Inventory Recommendations
```http
POST /api/inventory/recommendations
Content-Type: application/json

{
  "product_family": "BREAD/BAKERY",
  "model_type": "prophet",
  "current_stock": 100,
  "lead_time": 1,
  "service_level": 0.95
}
```

### Weather
```http
GET /api/weather/current?city=Quito&country=EC
GET /api/weather/forecast?city=Quito&country=EC&days=7
```

### Events/Holidays
```http
GET /api/events/holidays?country=EC&year=2024
GET /api/events/upcoming?country=EC&days=30
```

## Project Structure

```
backend/
├── api/                    # Flask application
│   └── app.py             # Main API routes
├── models/                # ML models
│   ├── arima_model.py
│   ├── prophet_model.py
│   └── lstm_model.py
├── utils/                 # Utility modules
│   ├── data_loader.py
│   ├── api_integrations.py
│   └── inventory_optimizer.py
├── config/                # Configuration
│   └── config.py
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── .env                  # Environment variables
```

## Model Types

### Prophet (Recommended)
- Best for seasonal patterns
- Handles holidays and events
- Fast training and inference
- MAPE: ~14-18%

### ARIMA
- Classical time series model
- Good for stationary data
- Simple and interpretable
- MAPE: ~16-20%

### LSTM
- Deep learning approach
- Learns complex patterns
- Requires more data
- MAPE: ~15-19%

## Development

### Testing
```bash
python test_installation.py
```

### Quick Demo
```bash
python quick_demo.py
```

## Notes

- First API call loads dataset (may take 10-30 seconds)
- Models are cached in memory (lost on restart)
- For production, use proper database and model persistence
- CORS is enabled for all origins (configure for production)

## License

Academic project - NIST University (Project ID: 34232)
