#!/bin/bash

# Customer Churn Prediction - Quick Start Script
# This script sets up and runs the complete project

echo "🚀 Customer Churn Prediction - Quick Start"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the churn-prediction directory"
    exit 1
fi

# Step 1: Setup environment
echo "🔧 Step 1: Setting up environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✅ Environment ready"

# Step 2: Generate data if missing
echo "📊 Step 2: Checking data..."
if [ ! -f "data/train_data.csv" ]; then
    echo "Generating sample data..."
    python -c "
import sys; sys.path.append('src')
from utils import generate_churn_data
import os; os.makedirs('data', exist_ok=True)
train, test, pred = generate_churn_data(5000)
train.to_csv('data/train_data.csv', index=False)
test.to_csv('data/test_data.csv', index=False)
pred.to_csv('data/customers_to_predict.csv', index=False)
print('✅ Data generated')
"
else
    echo "✅ Data already exists"
fi

# Step 3: Test functionality
echo "🧪 Step 3: Testing functionality..."
python -c "
import sys; sys.path.append('src')
from utils import generate_churn_data
from visualize import create_eda_visualizations
import pandas as pd
df = pd.read_csv('data/train_data.csv')
print(f'✅ Loaded {len(df)} training samples')
print(f'📈 Churn rate: {df[\"churned\"].mean():.2%}')
"

# Step 4: Show options
echo ""
echo "🎉 Setup Complete! Choose what to do next:"
echo ""
echo "📓 Interactive Analysis (Recommended):"
echo "   jupyter notebook notebooks/churn_analysis.ipynb"
echo ""
echo "🤖 Train Model:"
echo "   python src/train.py"
echo ""
echo "🔮 Make Predictions:"
echo "   python src/predict.py"
echo ""
echo "📊 Generate Charts:"
echo "   python -c \"import sys; sys.path.append('src'); from visualize import create_eda_visualizations; import pandas as pd; df = pd.read_csv('data/train_data.csv'); create_eda_visualizations(df); print('Charts created!')\""
echo ""
echo "☁️ Setup AWS (requires AWS CLI configured):"
echo "   python infrastructure/aws_setup.py"
echo ""
echo "📚 View Documentation:"
echo "   open README.md"
echo "   open PROJECT_SUMMARY.md"
echo ""

# Ask user what they want to do
read -p "🤔 What would you like to do? (j=jupyter, t=train, p=predict, c=charts, a=aws, q=quit): " choice

case $choice in
    j|J)
        echo "🚀 Starting Jupyter notebook..."
        jupyter notebook notebooks/churn_analysis.ipynb
        ;;
    t|T)
        echo "🤖 Training model..."
        python src/train.py
        ;;
    p|P)
        echo "🔮 Making predictions..."
        python src/predict.py
        ;;
    c|C)
        echo "📊 Generating charts..."
        python -c "
import sys; sys.path.append('src')
from visualize import create_eda_visualizations
import pandas as pd
df = pd.read_csv('data/train_data.csv')
create_eda_visualizations(df)
print('✅ Charts created in visualizations/charts/')
"
        ;;
    a|A)
        echo "☁️ Setting up AWS resources..."
        python infrastructure/aws_setup.py
        ;;
    q|Q)
        echo "👋 Thanks for using the Churn Prediction project!"
        ;;
    *)
        echo "ℹ️ You can run any of the commands above manually."
        echo "💡 Start with: jupyter notebook notebooks/churn_analysis.ipynb"
        ;;
esac

echo ""
echo "🎯 Project ready! Virtual environment is activated."
echo "💡 To reactivate later: source activate_env.sh"