#!/bin/bash

# Streamlit App Runner Script
echo "🚀 Starting Customer Churn Prediction Streamlit App"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Please run this script from the churn-prediction directory"
    exit 1
fi

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Install streamlit if not already installed
echo "📦 Checking Streamlit installation..."
pip install streamlit==1.28.1 > /dev/null 2>&1

# Check if model files exist
if [ ! -f "local_churn_model.pkl" ]; then
    echo "⚠️ Model files not found. Training a quick model..."
    python -c "
import sys; sys.path.append('src')
from train import ChurnModelTrainer
import pandas as pd

# Load or generate data
try:
    train_data = pd.read_csv('data/train_data.csv')
except:
    from utils import generate_churn_data
    import os
    os.makedirs('data', exist_ok=True)
    train_data, test_data, pred_data = generate_churn_data(5000)
    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)
    pred_data.to_csv('data/customers_to_predict.csv', index=False)

# Train model
trainer = ChurnModelTrainer()
model, encoders, metrics = trainer.train_local_model(train_data)
print('✅ Model trained and ready!')
"
fi

# Start Streamlit app
echo "🌟 Starting Streamlit app..."
echo "📱 App will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_app.py
