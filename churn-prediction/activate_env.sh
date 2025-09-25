#!/bin/bash

# Churn Prediction Project - Virtual Environment Activation Script
# Usage: source activate_env.sh

echo "🚀 Activating Churn Prediction Virtual Environment..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    return 1
fi

# Activate virtual environment
source .venv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ Virtual environment activated: $(basename $VIRTUAL_ENV)"
    echo "📍 Project directory: $(pwd)"
    echo ""
    echo "🎯 Available commands:"
    echo "  • python src/train.py          - Train churn model"
    echo "  • python src/predict.py        - Make predictions"
    echo "  • jupyter notebook             - Open Jupyter notebooks"
    echo "  • python infrastructure/aws_setup.py - Setup AWS resources"
    echo ""
    echo "📊 Quick test:"
    python -c "import pandas as pd; print(f'✅ Pandas {pd.__version__} ready')"
else
    echo "❌ Failed to activate virtual environment"
    return 1
fi