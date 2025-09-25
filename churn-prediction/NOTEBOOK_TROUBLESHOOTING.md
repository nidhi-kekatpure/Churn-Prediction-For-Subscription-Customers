# 📓 Jupyter Notebook Troubleshooting Guide

## 🔍 **Common Issues and Solutions**

### **Issue 1: Import Errors**
**Problem**: `ModuleNotFoundError: No module named 'utils'`

**Solutions**:
1. **Use the fixed notebook**: `notebooks/churn_analysis_fixed.ipynb`
2. **Run from project root**: Always start Jupyter from the `churn-prediction/` directory
3. **Activate virtual environment first**:
   ```bash
   cd churn-prediction
   source .venv/bin/activate
   jupyter notebook
   ```

### **Issue 2: Path Issues**
**Problem**: `FileNotFoundError: data/train_data.csv not found`

**Solutions**:
1. **Check current directory**: The notebook should find data automatically
2. **Generate data if missing**:
   ```bash
   source .venv/bin/activate
   python -c "
   import sys; sys.path.append('src')
   from utils import generate_churn_data
   import os; os.makedirs('data', exist_ok=True)
   train, test, pred = generate_churn_data(5000)
   train.to_csv('data/train_data.csv', index=False)
   test.to_csv('data/test_data.csv', index=False)
   pred.to_csv('data/customers_to_predict.csv', index=False)
   print('✅ Data generated!')
   "
   ```

### **Issue 3: Kernel Issues**
**Problem**: Kernel not starting or wrong Python version

**Solutions**:
1. **Install kernel in virtual environment**:
   ```bash
   source .venv/bin/activate
   python -m ipykernel install --user --name=churn-prediction --display-name="Churn Prediction"
   ```

2. **Select correct kernel in Jupyter**:
   - In Jupyter: Kernel → Change Kernel → churn-prediction

### **Issue 4: Package Import Errors**
**Problem**: Packages not found even with virtual environment

**Solutions**:
1. **Verify virtual environment**:
   ```bash
   source .venv/bin/activate
   which python  # Should show .venv/bin/python
   pip list | grep pandas  # Should show pandas 2.0.3
   ```

2. **Reinstall packages if needed**:
   ```bash
   source .venv/bin/activate
   pip install --force-reinstall -r requirements.txt
   ```

## 🚀 **Step-by-Step Working Solution**

### **Method 1: Use Fixed Notebook (Recommended)**

1. **Activate environment**:
   ```bash
   cd churn-prediction
   source .venv/bin/activate
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook notebooks/churn_analysis_fixed.ipynb
   ```

3. **Run cells sequentially** - the fixed notebook handles all path issues automatically

### **Method 2: Quick Test Script**

If the notebook still doesn't work, use this test script:

```bash
source .venv/bin/activate
python -c "
# Test all functionality
import sys, os
sys.path.append('src')

print('🧪 Testing notebook functionality...')

# Test imports
from utils import generate_churn_data, calculate_model_metrics
from visualize import create_eda_visualizations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print('✅ All imports successful')

# Test data loading
if os.path.exists('data/train_data.csv'):
    df = pd.read_csv('data/train_data.csv')
    print(f'✅ Loaded {len(df)} rows of training data')
else:
    print('⚠️ Generating sample data...')
    train, test, pred = generate_churn_data(1000)
    os.makedirs('data', exist_ok=True)
    train.to_csv('data/train_data.csv', index=False)
    test.to_csv('data/test_data.csv', index=False)
    pred.to_csv('data/customers_to_predict.csv', index=False)
    df = train
    print('✅ Sample data generated and saved')

# Test basic analysis
print(f'📊 Dataset shape: {df.shape}')
print(f'📈 Churn rate: {df[\"churned\"].mean():.2%}')

# Test visualization
os.makedirs('test_charts', exist_ok=True)
create_eda_visualizations(df, save_path='test_charts')
print('✅ Visualizations created in test_charts/')

print('🎉 All functionality working! Notebook should work now.')
"
```

### **Method 3: VS Code Integration**

1. **Open in VS Code**:
   ```bash
   cd churn-prediction
   code notebooks/churn_analysis_fixed.ipynb
   ```

2. **Select Python interpreter**:
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose `.venv/bin/python`

3. **Run cells** - VS Code handles the kernel automatically

## 🔧 **Environment Verification**

Run this complete verification:

```bash
cd churn-prediction
source .venv/bin/activate

echo "🔍 Environment Verification:"
echo "Python: $(python --version)"
echo "Current directory: $(pwd)"
echo "Virtual env: $VIRTUAL_ENV"

echo -e "\n📦 Key packages:"
python -c "
import pandas as pd; print(f'Pandas: {pd.__version__}')
import numpy as np; print(f'NumPy: {np.__version__}')
import sklearn; print(f'Scikit-learn: {sklearn.__version__}')
import xgboost as xgb; print(f'XGBoost: {xgb.__version__}')
import boto3; print(f'Boto3: {boto3.__version__}')
import sagemaker; print(f'SageMaker: {sagemaker.__version__}')
"

echo -e "\n📁 Data files:"
ls -la data/ 2>/dev/null || echo "No data directory found"

echo -e "\n🧪 Import test:"
python -c "
import sys; sys.path.append('src')
from utils import generate_churn_data
from visualize import create_eda_visualizations
print('✅ All project imports working')
"

echo -e "\n🎉 If all above checks pass, the notebook should work!"
```

## 🆘 **Still Having Issues?**

### **Nuclear Option: Complete Reset**

```bash
cd churn-prediction

# 1. Remove virtual environment
rm -rf .venv

# 2. Recreate environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install packages
pip install -r requirements.txt

# 4. Install Jupyter kernel
python -m ipykernel install --user --name=churn-prediction

# 5. Generate data
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

# 6. Test notebook
jupyter notebook notebooks/churn_analysis_fixed.ipynb
```

### **Alternative: Use Python Scripts**

If Jupyter continues to have issues, you can run the analysis using Python scripts:

```bash
source .venv/bin/activate

# Run full analysis
python -c "
import sys; sys.path.append('src')
from utils import generate_churn_data, calculate_model_metrics
from visualize import create_eda_visualizations, create_model_performance_charts
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

print('🎯 Running Complete Churn Analysis...')

# Load or generate data
if os.path.exists('data/train_data.csv'):
    train_data = pd.read_csv('data/train_data.csv')
else:
    train_data, _, _ = generate_churn_data(5000)

print(f'📊 Loaded {len(train_data)} training samples')
print(f'📈 Churn rate: {train_data[\"churned\"].mean():.2%}')

# Create visualizations
create_eda_visualizations(train_data, save_path='visualizations/charts')
print('✅ EDA visualizations created')

# Quick model training
data = train_data.copy()
categorical_cols = ['contract_type', 'payment_method', 'internet_service']
for col in categorical_cols:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop(['churned', 'customer_id'], axis=1)
y = data['churned']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]
metrics = calculate_model_metrics(y_val, y_pred, y_prob)

print(f'🎯 Model Performance:')
print(f'  AUC Score: {metrics[\"auc_score\"]:.4f}')
print(f'  Accuracy: {metrics[\"accuracy\"]:.4f}')
print(f'  Precision: {metrics[\"precision\"]:.4f}')
print(f'  Recall: {metrics[\"recall\"]:.4f}')

create_model_performance_charts(y_val, y_pred, y_prob, model.feature_importances_, X.columns.tolist())
print('✅ Model performance charts created')

print('🎉 Complete analysis finished! Check visualizations/charts/ for results.')
"
```

## ✅ **Success Checklist**

- [ ] Virtual environment activated
- [ ] All packages installed and importable
- [ ] Data files exist in `data/` directory
- [ ] Jupyter kernel configured correctly
- [ ] Using `churn_analysis_fixed.ipynb` notebook
- [ ] Running from project root directory
- [ ] All imports working without errors

**If all items are checked, your notebook should work perfectly!** 🎉