# Customer Churn Prediction with AWS

A complete machine learning project to predict subscription customer churn using AWS services. This project demonstrates beginner-to-intermediate AWS ML skills with a practical business use case.

## Project Overview

**Objective**: Predict which subscription customers are likely to churn and identify revenue at risk.

**Business Value**: 
- Proactive customer retention
- Revenue protection
- Targeted marketing campaigns
- Customer lifetime value optimization

---

## Live Demo

**Try the app now**: https://churn-prediction-for-customers.streamlit.app/

---

## Architecture

```
CSV Data → S3 → SageMaker Training → Model Artifacts → SageMaker Endpoint → Predictions
                     ↓
              Visualizations & Analysis
```

## AWS Services Used

1. **Amazon S3** - Data storage for CSV files, model artifacts, and results
2. **Amazon SageMaker** - ML model training, deployment, and hosting
3. **AWS IAM** - Role-based access control for SageMaker
4. **Optional**: AWS Lambda for automation, QuickSight for dashboards

## Dataset Features

The project uses a realistic synthetic dataset with 12 features:

- **Demographics**: Age, tenure months
- **Financial**: Monthly charges, total charges
- **Service**: Contract type, payment method, internet service
- **Usage**: Data usage, login frequency, support interactions
- **Target**: Churn label (0=retained, 1=churned)

## Quick Start

### Prerequisites
- AWS CLI configured (`aws configure`)
- Python 3.8+

### 1. Setup Environment
```bash
cd churn-prediction
source activate_env.sh  # Activates virtual environment and installs dependencies
```

### 2. Interactive Analysis (Recommended)
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

### 3. Or Run Individual Components
```bash
# Setup AWS resources
python infrastructure/aws_setup.py

# Train model
python src/train.py

# Make predictions
python src/predict.py
```

## Project Structure

```
churn-prediction/
├── data/
│   ├── train_data.csv              # Training dataset (4,000 samples)
│   ├── test_data.csv               # Test dataset with labels (1,000 samples)
│   └── customers_to_predict.csv    # New customers for prediction
├── src/
│   ├── utils.py                    # Data generation and metrics
│   ├── train.py                    # Model training pipeline
│   ├── predict.py                  # Prediction pipeline
│   └── visualize.py                # Visualization functions
├── notebooks/
│   └── churn_analysis.ipynb        # Interactive analysis notebook
├── infrastructure/
│   └── aws_setup.py                # AWS resource setup
├── visualizations/
│   └── charts/                     # Generated charts and plots
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Model Performance

The XGBoost model achieves:
- **AUC Score**: ~0.85-0.90
- **Accuracy**: ~80-85%
- **Precision**: ~75-80%
- **Recall**: ~70-75%

### Key Features (by importance):
1. Tenure months
2. Monthly charges
3. Contract type
4. Support calls
5. Payment method

## Visualizations

The project generates comprehensive visualizations:

- **EDA Charts**: Churn distribution, feature correlations, categorical analysis
- **Model Performance**: ROC curve, confusion matrix, feature importance
- **Business Insights**: Risk segmentation, revenue impact analysis

## Business Insights

### Risk Segmentation:
- **High Risk** (>70% churn probability): Immediate intervention needed
- **Medium Risk** (30-70%): Targeted retention campaigns
- **Low Risk** (<30%): Standard customer success programs

### Key Findings:
- Month-to-month contracts have 3x higher churn rate
- Customers with >3 support calls are 2x more likely to churn
- Electronic check payments correlate with higher churn risk
- New customers (<6 months) need enhanced onboarding

## Advanced Usage

### SageMaker Training
```python
from src.train import ChurnModelTrainer

trainer = ChurnModelTrainer()
model, metadata = trainer.train_sagemaker_model(train_data)
```

### Real-time Predictions
```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor()
results = predictor.predict_sagemaker('data/customers_to_predict.csv')
```

### Custom Visualizations
```python
from src.visualize import create_eda_visualizations

create_eda_visualizations(your_data, save_path='custom_charts/')
```

## Cost Optimization

**Estimated AWS Costs** (us-east-1):
- S3 storage: ~$0.50/month for 1GB
- SageMaker training: ~$0.10/hour (ml.m5.large)
- SageMaker endpoint: ~$0.05/hour (ml.t2.medium)

**Cost-saving tips**:
- Use Spot instances for training
- Stop endpoints when not in use
- Use S3 Intelligent Tiering
- Set up billing alerts

## Security Best Practices

- IAM roles with least privilege access
- S3 bucket encryption enabled
- VPC endpoints for private communication
- Regular access key rotation
- CloudTrail logging enabled

## Next Steps

### Immediate Enhancements:
1. **Hyperparameter Tuning**: Use SageMaker Automatic Model Tuning
2. **Model Monitoring**: Set up data drift detection
3. **A/B Testing**: Compare model versions in production
4. **Feature Store**: Use SageMaker Feature Store for feature management

### Advanced Features:
1. **Real-time Streaming**: Kinesis + Lambda for real-time scoring
2. **Multi-model Endpoints**: Deploy multiple models on single endpoint
3. **AutoML**: Try SageMaker Autopilot for automated ML
4. **MLOps Pipeline**: CI/CD with SageMaker Pipelines

### Business Integration:
1. **CRM Integration**: Connect predictions to Salesforce/HubSpot
2. **Marketing Automation**: Trigger campaigns based on churn risk
3. **Customer Success**: Alert CSMs about high-risk customers
4. **Executive Dashboard**: QuickSight dashboard for leadership

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request
