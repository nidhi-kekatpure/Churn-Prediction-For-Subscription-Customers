#!/bin/bash

# Streamlit App Runner Script
echo "🚀 Starting Customer Churn Prediction Streamlit App"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Please run this script from the churn-prediction directory"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Detected Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.9" ]]; then
    echo "❌ Python 3.9+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Install required packages
echo "📦 Installing packages from requirements.txt..."
python3 -m pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "⚠️ Full installation failed. Trying core packages only..."
    python3 -m pip install streamlit pandas numpy --quiet
    
    if [ $? -ne 0 ]; then
        echo "❌ Could not install required packages. Please install manually:"
        echo "   pip install streamlit pandas numpy"
        exit 1
    fi
fi

echo "✅ Packages installed successfully!"

# Generate sample data if missing
if [ ! -f "data/train_data.csv" ]; then
    echo "📊 Generating sample data..."
    python3 -c "
import sys; sys.path.append('src')
import os
import pandas as pd
import numpy as np

# Simple data generation
np.random.seed(42)
n_samples = 1000

data = {
    'customer_id': [f'CUST_{i:06d}' for i in range(1, n_samples + 1)],
    'age': np.random.normal(35, 12, n_samples).astype(int),
    'tenure_months': np.random.exponential(24, n_samples).astype(int),
    'monthly_charges': np.random.normal(65, 25, n_samples),
    'total_charges': np.random.normal(1500, 800, n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'payment_method': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer', 'Mailed check'], n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'support_calls': np.random.poisson(2, n_samples),
    'avg_call_duration': np.random.exponential(8, n_samples),
    'data_usage_gb': np.random.lognormal(3, 1, n_samples),
    'login_frequency': np.random.poisson(15, n_samples),
    'churned': np.random.binomial(1, 0.3, n_samples)
}

df = pd.DataFrame(data)
os.makedirs('data', exist_ok=True)
df.to_csv('data/train_data.csv', index=False)
print('✅ Sample data generated!')
"
fi

# Start Streamlit app
echo "🌟 Starting Streamlit app..."
echo "📱 App will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_app.py
