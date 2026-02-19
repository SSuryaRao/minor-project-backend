"""
Upload handler for user-uploaded CSV/Excel files.
Handles parsing, validation, column auto-detection, and temporary in-memory storage.
Also supports loading server-side files directly.
"""
import pandas as pd
import numpy as np
import uuid
import time
import os


# In-memory storage for uploaded datasets (keyed by session_id)
upload_store = {}

# Cleanup threshold: remove uploads older than 30 minutes
UPLOAD_TTL_SECONDS = 1800


def cleanup_old_uploads():
    """Remove uploads older than TTL."""
    now = time.time()
    expired = [k for k, v in upload_store.items() if now - v['created_at'] > UPLOAD_TTL_SECONDS]
    for k in expired:
        del upload_store[k]


def parse_uploaded_file(file_storage):
    """
    Parse an uploaded file (CSV or Excel) into a pandas DataFrame.

    Parameters:
        file_storage: werkzeug FileStorage object from request.files

    Returns:
        tuple: (df: pd.DataFrame, filename: str)

    Raises:
        ValueError: if file type is unsupported or parsing fails
    """
    filename = file_storage.filename or ''
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    if ext == 'csv':
        df = pd.read_csv(file_storage)
    elif ext in ('xlsx', 'xls'):
        df = pd.read_excel(file_storage, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Only .csv, .xlsx, .xls are accepted.")

    if df.empty:
        raise ValueError("The uploaded file contains no data.")

    if len(df) < 10:
        raise ValueError(f"The file has only {len(df)} rows. At least 10 rows are needed for meaningful analysis.")

    return df, filename


def detect_columns(df):
    """
    Auto-detect date columns and numeric columns suitable for forecasting.

    Returns:
        dict with 'date_columns' and 'numeric_columns' lists,
        plus 'suggested_date' and 'suggested_target' best guesses.
    """
    date_columns = []
    numeric_columns = []

    for col in df.columns:
        # Check if it's already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue

        # Try to parse as date
        if df[col].dtype == object:
            try:
                sample = df[col].dropna().head(20)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
                    if parsed.notna().sum() / len(sample) > 0.8:
                        date_columns.append(col)
                        continue
            except Exception:
                pass

        # Check numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)

    # Suggest best date column (prefer columns with 'date' in name)
    suggested_date = None
    for col in date_columns:
        if 'date' in col.lower():
            suggested_date = col
            break
    if not suggested_date and date_columns:
        suggested_date = date_columns[0]

    # Suggest best target column (prefer columns with common target keywords)
    suggested_target = None
    priority_keywords = ['sales', 'revenue', 'amount', 'value', 'quantity', 'demand', 'count', 'total']
    for keyword in priority_keywords:
        for col in numeric_columns:
            if keyword in col.lower():
                suggested_target = col
                break
        if suggested_target:
            break
    if not suggested_target and numeric_columns:
        suggested_target = numeric_columns[0]

    return {
        'date_columns': date_columns,
        'numeric_columns': numeric_columns,
        'suggested_date': suggested_date,
        'suggested_target': suggested_target,
    }


def validate_and_prepare(df, date_col, target_col):
    """
    Validate the user's column mapping and prepare data for model training.
    For large datasets (>10k rows), auto-aggregates by date to reduce size.

    Parameters:
        df: raw DataFrame
        date_col: user-selected date column name
        target_col: user-selected target column name

    Returns:
        tuple: (prepared DataFrame with 'date' and 'sales' columns, info dict)

    Raises:
        ValueError: on validation errors
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in data.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    prepared = df[[date_col, target_col]].copy()
    original_rows = len(prepared)

    # Parse dates
    prepared[date_col] = pd.to_datetime(prepared[date_col], errors='coerce')
    na_dates = prepared[date_col].isna().sum()
    if na_dates > len(prepared) * 0.2:
        raise ValueError(f"Too many unparseable dates ({na_dates}/{len(prepared)}). Check your date column.")

    # Drop rows with NaT dates
    prepared = prepared.dropna(subset=[date_col])

    # Parse target as numeric
    prepared[target_col] = pd.to_numeric(prepared[target_col], errors='coerce')
    na_target = prepared[target_col].isna().sum()
    if na_target > len(prepared) * 0.3:
        raise ValueError(f"Too many non-numeric values in target column ({na_target}/{len(prepared)}).")

    # Fill small gaps in target with forward fill
    prepared[target_col] = prepared[target_col].ffill().bfill()

    # Drop any remaining NaN rows
    prepared = prepared.dropna()

    if len(prepared) < 10:
        raise ValueError(f"After cleaning, only {len(prepared)} valid rows remain. Need at least 10.")

    # Rename to standard column names expected by models
    prepared = prepared.rename(columns={date_col: 'date', target_col: 'sales'})
    prepared = prepared.sort_values('date').reset_index(drop=True)

    # Auto-aggregate large datasets by date (sum per day)
    aggregated = False
    if len(prepared) > 10000:
        prepared = prepared.groupby(prepared['date'].dt.date).agg(
            sales=('sales', 'sum')
        ).reset_index()
        prepared['date'] = pd.to_datetime(prepared['date'])
        prepared = prepared.sort_values('date').reset_index(drop=True)
        aggregated = True

    info = {
        'original_rows': original_rows,
        'final_rows': len(prepared),
        'aggregated': aggregated,
    }

    return prepared, info


def store_upload(df, filename):
    """Store parsed DataFrame in memory with a unique session_id. Returns session_id."""
    cleanup_old_uploads()
    session_id = str(uuid.uuid4())
    upload_store[session_id] = {
        'df': df,
        'filename': filename,
        'created_at': time.time(),
    }
    return session_id


def get_upload(session_id):
    """Retrieve stored DataFrame by session_id. Returns (df, filename) or raises ValueError."""
    cleanup_old_uploads()
    if session_id not in upload_store:
        raise ValueError("Upload session expired or not found. Please upload the file again.")
    entry = upload_store[session_id]
    return entry['df'], entry['filename']
