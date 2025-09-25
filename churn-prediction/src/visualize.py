import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_eda_visualizations(df, save_path='visualizations/charts'):
    """
    Create exploratory data analysis visualizations
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Churn distribution
    plt.figure(figsize=(8, 6))
    churn_counts = df['churned'].value_counts()
    plt.pie(churn_counts.values, labels=['Retained', 'Churned'], autopct='%1.1f%%', startangle=90)
    plt.title('Customer Churn Distribution')
    plt.savefig(f'{save_path}/churn_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Churn by contract type
    plt.figure(figsize=(10, 6))
    churn_by_contract = df.groupby('contract_type')['churned'].agg(['count', 'sum', 'mean']).reset_index()
    churn_by_contract['churn_rate'] = churn_by_contract['mean']
    
    sns.barplot(data=churn_by_contract, x='contract_type', y='churn_rate')
    plt.title('Churn Rate by Contract Type')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.savefig(f'{save_path}/churn_by_contract.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Monthly charges vs churn
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='churned', y='monthly_charges')
    plt.title('Monthly Charges by Churn Status')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='churned', y='tenure_months')
    plt.title('Tenure by Churn Status')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/charges_tenure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{save_path}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Support calls vs churn
    plt.figure(figsize=(10, 6))
    support_churn = df.groupby('support_calls')['churned'].mean().reset_index()
    sns.lineplot(data=support_churn, x='support_calls', y='churned', marker='o')
    plt.title('Churn Rate by Number of Support Calls')
    plt.ylabel('Churn Rate')
    plt.xlabel('Number of Support Calls')
    plt.savefig(f'{save_path}/support_calls_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"EDA visualizations saved to {save_path}/")

def create_model_performance_charts(y_true, y_pred, y_prob, feature_importance=None, 
                                  feature_names=None, save_path='visualizations/charts'):
    """
    Create model performance visualizations
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 1. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = np.trapz(tpr, fpr)
    
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance (if provided)
    if feature_importance is not None and feature_names is not None:
        plt.figure(figsize=(10, 8))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        sns.barplot(data=importance_df.tail(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Prediction Distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='Retained', density=True)
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='Churned', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    thresholds = np.arange(0.1, 1.0, 0.1)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        pred_thresh = (y_prob >= threshold).astype(int)
        tp = np.sum((pred_thresh == 1) & (y_true == 1))
        fp = np.sum((pred_thresh == 1) & (y_true == 0))
        fn = np.sum((pred_thresh == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    plt.plot(thresholds, precisions, 'o-', label='Precision')
    plt.plot(thresholds, recalls, 's-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model performance charts saved to {save_path}/")