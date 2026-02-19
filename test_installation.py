"""
Test Installation Script
Run this to verify all components are working
"""
import sys

# Set UTF-8 encoding for output (Windows fix)
if sys.platform == 'win32':
    import codecs
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

print("="*70)
print(" TESTING INSTALLATION")
print("="*70)

# Test imports
tests_passed = 0
tests_failed = 0

print("\n1. Testing Core Libraries...")
try:
    import numpy as np
    print("   [OK] NumPy", np.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] NumPy - {e}")
    tests_failed += 1

try:
    import pandas as pd
    print("   [OK] Pandas", pd.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] Pandas - {e}")
    tests_failed += 1

try:
    import matplotlib
    print("   [OK] Matplotlib", matplotlib.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] Matplotlib - {e}")
    tests_failed += 1

print("\n2. Testing Web Framework...")
try:
    import flask
    print("   [OK] Flask", flask.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] Flask - {e}")
    tests_failed += 1

try:
    import flask_cors
    print("   [OK] Flask-CORS")
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] Flask-CORS - {e}")
    tests_failed += 1

print("\n3. Testing Time Series Libraries...")
try:
    import statsmodels
    print("   [OK] Statsmodels", statsmodels.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] Statsmodels - {e}")
    tests_failed += 1

try:
    from prophet import Prophet
    print("   [OK] Prophet (Facebook Forecasting)")
    tests_passed += 1
except Exception as e:
    print(f"   [WARN] Prophet - {e}")
    print("     Note: Prophet is optional. System will work with ARIMA.")

print("\n4. Testing Machine Learning...")
try:
    import sklearn
    print("   [OK] Scikit-learn", sklearn.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] Scikit-learn - {e}")
    tests_failed += 1

try:
    import scipy
    print("   [OK] SciPy", scipy.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] SciPy - {e}")
    tests_failed += 1

print("\n5. Testing Visualization...")
try:
    import seaborn
    print("   [OK] Seaborn", seaborn.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] Seaborn - {e}")
    tests_failed += 1

try:
    import plotly
    print("   [OK] Plotly", plotly.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] Plotly - {e}")
    tests_failed += 1

print("\n6. Testing Optional Packages...")
try:
    import tensorflow as tf
    print("   [OK] TensorFlow", tf.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [WARN] TensorFlow - {e}")
    print("     Note: TensorFlow is optional. Needed only for LSTM model.")

try:
    import jupyter
    print("   [OK] Jupyter")
    tests_passed += 1
except Exception as e:
    print(f"   [WARN] Jupyter - {e}")
    print("     Note: Jupyter is optional. Needed only for notebooks.")

print("\n7. Testing Utilities...")
try:
    import requests
    print("   [OK] Requests", requests.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] Requests - {e}")
    tests_failed += 1

try:
    import dotenv
    print("   [OK] Python-dotenv")
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] Python-dotenv - {e}")
    tests_failed += 1

try:
    import sqlalchemy
    print("   [OK] SQLAlchemy", sqlalchemy.__version__)
    tests_passed += 1
except Exception as e:
    print(f"   [FAIL] SQLAlchemy - {e}")
    tests_failed += 1

print("\n" + "="*70)
print(f" RESULTS: {tests_passed} passed, {tests_failed} failed")
print("="*70)

if tests_failed == 0:
    print("\n[SUCCESS] All core packages installed successfully!")
    print("\nNext steps:")
    print("  1. Run: python quick_demo.py")
    print("  2. Or start the server: python api/app.py")
    print("  3. Then open: http://localhost:5000")
elif tests_failed <= 2:
    print("\n[WARNING] Most packages installed. System should work with available models.")
    print("\nYou can proceed with:")
    print("  python api/app.py")
else:
    print("\n[ERROR] Some required packages are missing.")
    print("\nTry:")
    print("  pip install -r requirements.txt")
    print("\nOr check INSTALL.md for troubleshooting")

print("\n" + "="*70)
