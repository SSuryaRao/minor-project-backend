"""
Gemini AI Insights — generates plain-English explanations for forecasts,
inventory alerts, and anomaly detection using the Gemini API.
"""
import os

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def _get_model():
    if not _GENAI_AVAILABLE:
        raise RuntimeError(
            "google-generativeai is not installed. Run: pip install google-generativeai"
        )
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to backend/.env"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def forecast_summary(product_family: str, model_type: str, metrics: dict, predictions: list) -> str:
    """
    Generate a 2-3 sentence plain-English summary of a sales forecast.

    Args:
        product_family: e.g. "DAIRY"
        model_type: e.g. "prophet"
        metrics: dict with mae, rmse, mape keys
        predictions: list of dicts with date, predicted/forecast, lower, upper

    Returns:
        Plain-text insight string.
    """
    model = _get_model()

    if not predictions:
        return "No forecast data available to analyse."

    values = [p.get("predicted") or p.get("forecast") or 0 for p in predictions]
    avg = sum(values) / len(values) if values else 0

    peak = max(predictions, key=lambda p: p.get("predicted") or p.get("forecast") or 0)
    low = min(predictions, key=lambda p: p.get("predicted") or p.get("forecast") or 0)

    peak_val = peak.get("predicted") or peak.get("forecast") or 0
    low_val = low.get("predicted") or low.get("forecast") or 0
    pct_above = ((peak_val - avg) / avg * 100) if avg else 0

    mae = metrics.get("mae", "N/A")
    rmse = metrics.get("rmse", "N/A")
    mape = metrics.get("mape")
    mape_str = f"{mape * 100:.1f}%" if mape is not None else "N/A"

    prompt = (
        f"You are a retail analytics assistant. Summarise this sales forecast in 2-3 concise sentences "
        f"suitable for a store manager. Be specific with numbers.\n\n"
        f"Product: {product_family}\n"
        f"Model: {model_type.upper()}\n"
        f"Forecast period: {len(predictions)} days\n"
        f"Average predicted daily sales: {avg:.1f} units\n"
        f"Peak day: {peak.get('date', 'unknown')} with {peak_val:.0f} units ({pct_above:+.1f}% above average)\n"
        f"Lowest day: {low.get('date', 'unknown')} with {low_val:.0f} units\n"
        f"Model accuracy — MAE: {mae}, RMSE: {rmse}, MAPE: {mape_str}\n\n"
        f"Include a brief restocking recommendation based on the peak demand."
    )

    response = model.generate_content(prompt)
    return response.text.strip()


def inventory_alert(product_family: str, recommendations: dict) -> str:
    """
    Generate a plain-English inventory action plan from ML recommendations.

    Args:
        product_family: e.g. "SEAFOOD"
        recommendations: dict from inventory_optimizer (stock_status, order_quantity, etc.)

    Returns:
        Plain-text action plan string.
    """
    model = _get_model()

    stock_status = recommendations.get("stock_status", "unknown")
    order_rec = recommendations.get("order_recommendation", "unknown")
    order_qty = recommendations.get("order_quantity", "N/A")
    wastage_risk = recommendations.get("wastage_risk", "unknown")
    safety_stock = recommendations.get("safety_stock")
    days_of_stock = recommendations.get("days_of_stock")
    daily_forecast = recommendations.get("avg_daily_forecast")

    safety_str = f"{safety_stock:.0f}" if safety_stock is not None else "N/A"
    days_str = f"{days_of_stock:.1f}" if days_of_stock is not None else "N/A"
    daily_str = f"{daily_forecast:.1f}" if daily_forecast is not None else "N/A"

    prompt = (
        f"You are an inventory management assistant. Write a concise (2-3 sentences) action plan "
        f"for a store manager based on these inventory metrics. Be direct and actionable.\n\n"
        f"Product: {product_family}\n"
        f"Stock status: {stock_status}\n"
        f"Days of stock remaining: {days_str}\n"
        f"Daily forecast demand: {daily_str} units\n"
        f"Recommended action: {order_rec}\n"
        f"Suggested order quantity: {order_qty} units\n"
        f"Wastage risk: {wastage_risk}\n"
        f"Safety stock buffer: {safety_str} units\n\n"
        f"If stock is critically low, emphasise urgency. If wastage risk is high, warn against over-ordering."
    )

    response = model.generate_content(prompt)
    return response.text.strip()


def anomaly_explain(product_family: str, date: str, actual: float, expected: float,
                    holidays: list = None, weather: dict = None) -> str:
    """
    Generate a plain-English explanation for a sales anomaly.

    Args:
        product_family: e.g. "SEAFOOD"
        date: ISO date string
        actual: actual sales value
        expected: expected/baseline sales value
        holidays: list of holiday dicts near this date (optional)
        weather: weather dict for this date (optional)

    Returns:
        Plain-text explanation string.
    """
    model = _get_model()

    deviation = ((actual - expected) / expected * 100) if expected else 0
    direction = "spike" if deviation > 0 else "drop"

    holiday_info = ""
    if holidays:
        names = [h.get("name", "") for h in holidays[:3] if h.get("name")]
        if names:
            holiday_info = f"Nearby holidays: {', '.join(names)}\n"

    weather_info = ""
    if weather:
        desc = weather.get("description", "")
        temp = weather.get("temperature")
        if desc or temp is not None:
            temp_str = f", temperature: {temp}°C" if temp is not None else ""
            weather_info = f"Weather conditions: {desc}{temp_str}\n"

    prompt = (
        f"You are a retail analytics assistant. Explain this sales anomaly in 2-3 sentences "
        f"for a store manager. Suggest what may have caused it and what to do next year.\n\n"
        f"Product: {product_family}\n"
        f"Date: {date}\n"
        f"Actual sales: {actual:.0f} units\n"
        f"Expected sales: {expected:.0f} units\n"
        f"Deviation: {deviation:+.1f}% ({direction})\n"
        f"{holiday_info}"
        f"{weather_info}"
        f"\nIf a holiday or weather event seems relevant, mention it explicitly."
    )

    response = model.generate_content(prompt)
    return response.text.strip()
