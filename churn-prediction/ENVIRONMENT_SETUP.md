# 🔧 Virtual Environment Setup Guide

## ✅ Virtual Environment Successfully Created!

Your Python virtual environment has been recreated with all necessary dependencies for the churn prediction project.

## 📁 Environment Structure

```
churn-prediction/
├── .venv/                    # Virtual environment (isolated Python packages)
├── activate_env.sh          # Quick activation script
├── requirements.txt         # Package dependencies
└── ENVIRONMENT_SETUP.md     # This guide
```

## 🚀 How to Use the Virtual Environment

### **Method 1: Quick Activation (Recommended)**
```bash
cd churn-prediction
source activate_env.sh
```

### **Method 2: Manual Activation**
```bash
cd churn-prediction
source .venv/bin/activate
```

### **Method 3: VS Code Integration**
1. Open VS Code in the `churn-prediction` directory
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
3. Type "Python: Select Interpreter"
4. Choose the interpreter from `.venv/bin/python`

## 📦 Installed Packages

| Package | Version | Purpose |
|---------|---------|---------|
| **boto3** | 1.34.144 | AWS SDK for Python |
| **sagemaker** | 2.220.0 | Amazon SageMaker SDK |
| **pandas** | 2.0.3 | Data manipulation and analysis |
| **numpy** | 1.24.3 | Numerical computing |
| **scikit-learn** | 1.3.0 | Machine learning library |
| **xgboost** | 1.7.6 | Gradient boosting framework |
| **matplotlib** | 3.7.2 | Plotting and visualization |
| **seaborn** | 0.12.2 | Statistical data visualization |
| **plotly** | 5.15.0 | Interactive visualizations |
| **jupyter** | 1.0.0 | Jupyter notebook environment |

## 🧪 Testing Your Environment

### **Quick Test**
```bash
source .venv/bin/activate
python -c "import pandas, numpy, sklearn, xgboost, boto3, sagemaker; print('✅ All packages working!')"
```

### **Full Project Test**
```bash
source .venv/bin/activate
python -c "
import sys
sys.path.append('src')
from utils import generate_churn_data
train, test, pred = generate_churn_data(100)
print(f'✅ Project functionality: {len(train)} samples generated')
"
```

## 🎯 Running Project Components

### **1. Train Model**
```bash
source .venv/bin/activate
python src/train.py
```

### **2. Make Predictions**
```bash
source .venv/bin/activate
python src/predict.py
```

### **3. Launch Jupyter Notebook**
```bash
source .venv/bin/activate
jupyter notebook notebooks/churn_analysis.ipynb
```

### **4. Generate Visualizations**
```bash
source .venv/bin/activate
python -c "
import sys
sys.path.append('src')
from visualize import create_eda_visualizations
import pandas as pd
data = pd.read_csv('data/train_data.csv')
create_eda_visualizations(data)
print('✅ Charts created in visualizations/charts/')
"
```

## 🔄 Environment Management

### **Deactivate Environment**
```bash
deactivate
```

### **Update Packages**
```bash
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

### **Add New Packages**
```bash
source .venv/bin/activate
pip install new-package-name
pip freeze > requirements.txt  # Update requirements
```

### **Recreate Environment (if needed)**
```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🐛 Troubleshooting

### **Issue: "Command not found: python"**
**Solution**: Use `python3` instead of `python`
```bash
python3 -m venv .venv
```

### **Issue: "Permission denied"**
**Solution**: Make activation script executable
```bash
chmod +x activate_env.sh
```

### **Issue: "Module not found"**
**Solution**: Ensure virtual environment is activated
```bash
source .venv/bin/activate
which python  # Should show .venv/bin/python
```

### **Issue: "AWS credentials not found"**
**Solution**: Configure AWS CLI
```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region
```

### **Issue: "Jupyter kernel not found"**
**Solution**: Install IPython kernel
```bash
source .venv/bin/activate
python -m ipykernel install --user --name=churn-prediction
```

## 📊 Environment Verification Checklist

- [ ] Virtual environment activates without errors
- [ ] All required packages import successfully
- [ ] AWS credentials are configured (`aws sts get-caller-identity`)
- [ ] Project data files exist in `data/` directory
- [ ] Jupyter notebook launches successfully
- [ ] Model training script runs without errors
- [ ] Prediction script generates results

## 🎉 Success Indicators

When everything is working correctly, you should see:

```bash
$ source activate_env.sh
🚀 Activating Churn Prediction Virtual Environment...
✅ Virtual environment activated: .venv
📍 Project directory: /path/to/churn-prediction
✅ Pandas 2.0.3 ready

$ python src/train.py
🎯 Starting Churn Prediction Model Training
Loaded training data: 4000 samples
📊 Creating exploratory data analysis...
🔧 Training local XGBoost model...
Local Model Performance:
AUC Score: 0.7922
✅ Training completed successfully!
```

## 💡 Pro Tips

1. **Always activate the environment** before running any project scripts
2. **Use the activation script** for convenience: `source activate_env.sh`
3. **Check your interpreter** in VS Code to ensure it's using the virtual environment
4. **Keep requirements.txt updated** when adding new packages
5. **Test regularly** to catch environment issues early

---

**Environment Status**: ✅ Ready for Development  
**Last Updated**: September 2025  
**Python Version**: 3.9.6  
**Total Packages**: 100+ dependencies installed