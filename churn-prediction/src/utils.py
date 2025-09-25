import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_churn_data(n_samples=5000, test_split=0.2):
    """
    Generate realistic churn dataset with meaningful features
    """
    np.random.seed(42)
    random.seed(42)
    
    # Customer demographics
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
    ages = np.random.normal(35, 12, n_samples).astype(int)
    ages = np.clip(ages, 18, 75)
    
    # Subscription details
    tenure_months = np.random.exponential(24, n_samples).astype(int)
    tenure_months = np.clip(tenure_months, 1, 72)
    
    monthly_charges = np.random.normal(65, 25, n_samples)
    monthly_charges = np.clip(monthly_charges, 15, 150)
    
    total_charges = monthly_charges * tenure_months + np.random.normal(0, 50, n_samples)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    # Service usage patterns
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                    n_samples, p=[0.5, 0.3, 0.2])
    
    payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                                     n_samples, p=[0.35, 0.15, 0.25, 0.25])
    
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                      n_samples, p=[0.4, 0.45, 0.15])
    
    # Support interactions
    support_calls = np.random.poisson(2, n_samples)
    avg_call_duration = np.random.exponential(8, n_samples)  # minutes
    
    # Usage metrics
    data_usage_gb = np.random.lognormal(3, 1, n_samples)
    login_frequency = np.random.poisson(15, n_samples)  # per month
    
    # Create churn probability based on realistic factors
    churn_prob = 0.1  # base probability
    
    # Increase churn probability based on risk factors
    churn_prob += (contract_types == 'Month-to-month') * 0.3
    churn_prob += (payment_methods == 'Electronic check') * 0.2
    churn_prob += (support_calls > 3) * 0.25
    churn_prob += (tenure_months < 6) * 0.4
    churn_prob += (monthly_charges > 80) * 0.15
    churn_prob += (login_frequency < 5) * 0.2
    
    # Decrease churn probability for loyal customers
    churn_prob -= (tenure_months > 24) * 0.2
    churn_prob -= (contract_types == 'Two year') * 0.25
    churn_prob -= (internet_service == 'Fiber optic') * 0.1
    
    churn_prob = np.clip(churn_prob, 0.05, 0.8)
    
    # Generate actual churn labels
    churned = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'tenure_months': tenure_months,
        'monthly_charges': np.round(monthly_charges, 2),
        'total_charges': np.round(total_charges, 2),
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'internet_service': internet_service,
        'support_calls': support_calls,
        'avg_call_duration': np.round(avg_call_duration, 1),
        'data_usage_gb': np.round(data_usage_gb, 2),
        'login_frequency': login_frequency,
        'churned': churned
    })
    
    # Split into train and test
    test_size = int(n_samples * test_split)
    train_data = data.iloc[:-test_size]
    test_data = data.iloc[-test_size:].copy()
    
    # Remove churn labels from test data for prediction
    test_data_for_prediction = test_data.drop('churned', axis=1)
    
    return train_data, test_data, test_data_for_prediction

def calculate_model_metrics(y_true, y_pred, y_prob):
    """
    Calculate comprehensive model performance metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc_score': roc_auc_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics