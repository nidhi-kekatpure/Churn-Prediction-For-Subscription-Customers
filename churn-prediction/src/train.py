import pandas as pd
import numpy as np
import json
import boto3
import sagemaker
from sagemaker.xgboost import XGBoost
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os
from utils import calculate_model_metrics
from visualize import create_eda_visualizations, create_model_performance_charts

class ChurnModelTrainer:
    def __init__(self, config_path='aws_config.json'):
        """Initialize trainer with AWS configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create boto session with correct region
        boto_session = boto3.Session(region_name=self.config['region'])
        self.sagemaker_session = sagemaker.Session(boto_session=boto_session)
        self.role = self.config['role_arn']
        self.bucket = self.config['bucket_name']
        
        print(f"✅ SageMaker session initialized for region: {self.config['region']}")
        
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['contract_type', 'payment_method', 'internet_service']
        
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        
        # Feature engineering
        data['charges_per_month_tenure'] = data['monthly_charges'] / (data['tenure_months'] + 1)
        data['total_charges_per_tenure'] = data['total_charges'] / (data['tenure_months'] + 1)
        data['avg_monthly_usage'] = data['data_usage_gb'] / (data['tenure_months'] + 1)
        data['support_calls_per_month'] = data['support_calls'] / (data['tenure_months'] + 1)
        
        # Handle any missing values (only for numeric columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        return data, label_encoders
    
    def prepare_sagemaker_data(self, df, target_col='churned'):
        """Prepare data in SageMaker XGBoost format (target first)"""
        # Separate features and target
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop([target_col, 'customer_id'], axis=1)
        else:
            X = df.drop(['customer_id'], axis=1)
            y = None
        
        # For SageMaker XGBoost, target should be first column
        if y is not None:
            sagemaker_data = pd.concat([y, X], axis=1)
        else:
            sagemaker_data = X
            
        return sagemaker_data, X.columns.tolist()
    
    def train_local_model(self, train_data):
        """Train a local XGBoost model for quick validation"""
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        
        # Preprocess data
        processed_data, label_encoders = self.preprocess_data(train_data)
        
        # Prepare features and target
        X = processed_data.drop(['churned', 'customer_id'], axis=1)
        y = processed_data['churned']
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train XGBoost model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc'
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = calculate_model_metrics(y_val, y_pred, y_prob)
        
        print("Local Model Performance:")
        print(f"AUC Score: {metrics['auc_score']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        # Create visualizations
        create_model_performance_charts(
            y_val, y_pred, y_prob, 
            model.feature_importances_, 
            X.columns.tolist()
        )
        
        # Save model and encoders locally
        joblib.dump(model, 'local_churn_model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        
        return model, label_encoders, metrics
    
    def train_sagemaker_model(self, train_data):
        """Train model using SageMaker XGBoost"""
        
        # Preprocess data
        processed_data, label_encoders = self.preprocess_data(train_data)
        
        # Prepare data for SageMaker
        sagemaker_data, feature_names = self.prepare_sagemaker_data(processed_data)
        
        # Split train/validation
        train_df, val_df = train_test_split(sagemaker_data, test_size=0.2, random_state=42)
        
        # Save to CSV (SageMaker XGBoost expects CSV)
        train_df.to_csv('train_sagemaker.csv', index=False, header=False)
        val_df.to_csv('validation_sagemaker.csv', index=False, header=False)
        
        # Upload to S3
        train_s3_path = self.sagemaker_session.upload_data(
            path='train_sagemaker.csv',
            bucket=self.bucket,
            key_prefix='training-data'
        )
        
        val_s3_path = self.sagemaker_session.upload_data(
            path='validation_sagemaker.csv',
            bucket=self.bucket,
            key_prefix='validation-data'
        )
        
        print(f"Training data uploaded to: {train_s3_path}")
        print(f"Validation data uploaded to: {val_s3_path}")
        
        # Configure XGBoost estimator
        xgb_estimator = XGBoost(
            entry_point='xgboost_script.py',
            framework_version='1.5-1',
            instance_type='ml.m5.large',
            instance_count=1,
            role=self.role,
            output_path=self.config['model_output_path'],
            sagemaker_session=self.sagemaker_session,
            hyperparameters={
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'num_round': 100
            }
        )
        
        # Create training script
        self._create_training_script()
        
        # Define training inputs
        train_input = TrainingInput(train_s3_path, content_type='text/csv')
        val_input = TrainingInput(val_s3_path, content_type='text/csv')
        
        # Start training
        print("🚀 Starting SageMaker training job...")
        xgb_estimator.fit({
            'train': train_input,
            'validation': val_input
        })
        
        print("✅ SageMaker training completed!")
        
        # Save metadata
        training_metadata = {
            'model_s3_path': xgb_estimator.model_data,
            'feature_names': feature_names,
            'label_encoders': {k: v.classes_.tolist() for k, v in label_encoders.items()},
            'training_job_name': xgb_estimator.latest_training_job.name
        }
        
        with open('training_metadata.json', 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        return xgb_estimator, training_metadata
    
    def _create_training_script(self):
        """Create the training script for SageMaker"""
        script_content = '''
import argparse
import os
import pandas as pd
import xgboost as xgb
import joblib

def model_fn(model_dir):
    """Load model for inference"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def train():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    
    # Hyperparameters
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--eval_metric", type=str, default="auc")
    
    args = parser.parse_args()
    
    # Load data
    train_df = pd.read_csv(os.path.join(args.train, "train_sagemaker.csv"), header=None)
    val_df = pd.read_csv(os.path.join(args.validation, "validation_sagemaker.csv"), header=None)
    
    # Prepare DMatrix
    dtrain = xgb.DMatrix(train_df.iloc[:, 1:], label=train_df.iloc[:, 0])
    dval = xgb.DMatrix(val_df.iloc[:, 1:], label=val_df.iloc[:, 0])
    
    # Set parameters
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "objective": args.objective,
        "eval_metric": args.eval_metric
    }
    
    # Train model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=10,
        verbose_eval=True
    )
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train()
'''
        
        with open('xgboost_script.py', 'w') as f:
            f.write(script_content)

def main():
    """Main training pipeline"""
    print("🎯 Starting Churn Prediction Model Training")
    
    # Load training data
    train_data = pd.read_csv('data/train_data.csv')
    print(f"Loaded training data: {len(train_data)} samples")
    
    # Create EDA visualizations
    print("📊 Creating exploratory data analysis...")
    create_eda_visualizations(train_data)
    
    # Initialize trainer
    trainer = ChurnModelTrainer()
    
    # Train local model first for quick validation
    print("\n🔧 Training local XGBoost model...")
    local_model, encoders, metrics = trainer.train_local_model(train_data)
    
    # Ask user if they want to proceed with SageMaker training
    proceed = input("\n🤔 Local model trained successfully. Proceed with SageMaker training? (y/n): ")
    
    if proceed.lower() == 'y':
        print("\n☁️ Training model on SageMaker...")
        sagemaker_model, metadata = trainer.train_sagemaker_model(train_data)
        print("✅ SageMaker training completed!")
        
        return sagemaker_model, metadata
    else:
        print("✅ Training completed with local model only.")
        return local_model, encoders

if __name__ == "__main__":
    main()