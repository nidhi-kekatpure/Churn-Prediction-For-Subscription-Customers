import pandas as pd
import numpy as np
import json
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
import joblib
import os
from utils import calculate_model_metrics

class ChurnPredictor:
    def __init__(self, config_path='aws_config.json'):
        """Initialize predictor with AWS configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            # Create boto session with correct region
            boto_session = boto3.Session(region_name=self.config['region'])
            self.sagemaker_session = sagemaker.Session(boto_session=boto_session)
            print(f"✅ SageMaker session initialized for region: {self.config['region']}")
        else:
            self.config = None
            print("⚠️ AWS config not found. Using local model only.")
    
    def load_local_model(self):
        """Load locally trained model and encoders"""
        try:
            model = joblib.load('local_churn_model.pkl')
            encoders = joblib.load('label_encoders.pkl')
            print("✅ Local model loaded successfully")
            return model, encoders
        except FileNotFoundError:
            print("❌ Local model files not found. Please train the model first.")
            return None, None
    
    def preprocess_prediction_data(self, df, encoders):
        """Preprocess data for prediction using saved encoders"""
        data = df.copy()
        
        # Apply label encoders
        categorical_cols = ['contract_type', 'payment_method', 'internet_service']
        
        for col in categorical_cols:
            if col in encoders:
                # Handle unseen categories
                data[col] = data[col].map(
                    lambda x: encoders[col].transform([x])[0] 
                    if x in encoders[col].classes_ 
                    else 0
                )
        
        # Feature engineering (same as training)
        data['charges_per_month_tenure'] = data['monthly_charges'] / (data['tenure_months'] + 1)
        data['total_charges_per_tenure'] = data['total_charges'] / (data['tenure_months'] + 1)
        data['avg_monthly_usage'] = data['data_usage_gb'] / (data['tenure_months'] + 1)
        data['support_calls_per_month'] = data['support_calls'] / (data['tenure_months'] + 1)
        
        # Handle missing values (only for numeric columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        return data
    
    def predict_local(self, customer_data_path):
        """Make predictions using local model"""
        model, encoders = self.load_local_model()
        if model is None:
            return None
        
        # Load customer data
        customers = pd.read_csv(customer_data_path)
        print(f"Loaded {len(customers)} customers for prediction")
        
        # Preprocess data
        processed_data = self.preprocess_prediction_data(customers, encoders)
        
        # Prepare features (same order as training)
        feature_cols = processed_data.drop(['customer_id'], axis=1).columns
        X = processed_data[feature_cols]
        
        # Make predictions
        churn_probabilities = model.predict_proba(X)[:, 1]
        churn_predictions = model.predict(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'customer_id': customers['customer_id'],
            'churn_probability': churn_probabilities,
            'churn_prediction': churn_predictions,
            'risk_level': pd.cut(churn_probabilities, 
                               bins=[0, 0.3, 0.7, 1.0], 
                               labels=['Low', 'Medium', 'High'])
        })
        
        # Add customer info for context
        results = results.merge(
            customers[['customer_id', 'monthly_charges', 'tenure_months', 'contract_type']], 
            on='customer_id'
        )
        
        return results
    
    def deploy_sagemaker_model(self, model_name=None):
        """Deploy SageMaker model to endpoint"""
        if not self.config:
            print("❌ AWS config required for SageMaker deployment")
            return None
        
        try:
            # Load training metadata
            with open('training_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            # Create model from training job
            model = sagemaker.model.Model(
                image_uri=sagemaker.image_uris.retrieve('xgboost', self.config['region'], '1.5-1'),
                model_data=metadata['model_s3_path'],
                role=self.config['role_arn'],
                sagemaker_session=self.sagemaker_session
            )
            
            # Deploy to endpoint
            endpoint_name = model_name or f"churn-model-{int(pd.Timestamp.now().timestamp())}"
            
            print(f"🚀 Deploying model to endpoint: {endpoint_name}")
            predictor = model.deploy(
                initial_instance_count=1,
                instance_type='ml.t2.medium',
                endpoint_name=endpoint_name,
                serializer=CSVSerializer(),
                deserializer=CSVDeserializer()
            )
            
            print(f"✅ Model deployed to endpoint: {endpoint_name}")
            
            # Save endpoint info
            endpoint_config = {
                'endpoint_name': endpoint_name,
                'feature_names': metadata['feature_names'],
                'label_encoders': metadata['label_encoders']
            }
            
            with open('endpoint_config.json', 'w') as f:
                json.dump(endpoint_config, f, indent=2)
            
            return predictor, endpoint_config
            
        except Exception as e:
            print(f"❌ Error deploying model: {e}")
            return None, None
    
    def predict_sagemaker(self, customer_data_path, endpoint_name=None):
        """Make predictions using SageMaker endpoint"""
        if not self.config:
            print("❌ AWS config required for SageMaker predictions")
            return None
        
        try:
            # Load endpoint config
            if os.path.exists('endpoint_config.json'):
                with open('endpoint_config.json', 'r') as f:
                    endpoint_config = json.load(f)
                endpoint_name = endpoint_name or endpoint_config['endpoint_name']
            else:
                print("❌ Endpoint config not found. Deploy model first.")
                return None
            
            # Create predictor
            predictor = Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=self.sagemaker_session,
                serializer=CSVSerializer(),
                deserializer=CSVDeserializer()
            )
            
            # Load and preprocess data
            customers = pd.read_csv(customer_data_path)
            
            # For SageMaker, we need to reconstruct encoders from config
            encoders = {}
            for col, classes in endpoint_config['label_encoders'].items():
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.classes_ = np.array(classes)
                encoders[col] = le
            
            processed_data = self.preprocess_prediction_data(customers, encoders)
            
            # Prepare features for SageMaker (no target column, no customer_id)
            feature_data = processed_data.drop(['customer_id'], axis=1)
            
            # Make predictions
            predictions = predictor.predict(feature_data.values)
            
            # Parse predictions (SageMaker returns probabilities)
            if isinstance(predictions, list) and len(predictions) > 0:
                churn_probabilities = np.array([float(p[0]) for p in predictions])
            else:
                churn_probabilities = np.array(predictions).flatten()
            
            churn_predictions = (churn_probabilities > 0.5).astype(int)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'customer_id': customers['customer_id'],
                'churn_probability': churn_probabilities,
                'churn_prediction': churn_predictions,
                'risk_level': pd.cut(churn_probabilities, 
                                   bins=[0, 0.3, 0.7, 1.0], 
                                   labels=['Low', 'Medium', 'High'])
            })
            
            # Add customer info
            results = results.merge(
                customers[['customer_id', 'monthly_charges', 'tenure_months', 'contract_type']], 
                on='customer_id'
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error making SageMaker predictions: {e}")
            return None
    
    def analyze_predictions(self, results):
        """Analyze prediction results and provide insights"""
        if results is None:
            return
        
        print("\n📊 Prediction Analysis:")
        print(f"Total customers analyzed: {len(results)}")
        print(f"Predicted churners: {results['churn_prediction'].sum()} ({results['churn_prediction'].mean():.1%})")
        
        print("\n🎯 Risk Level Distribution:")
        risk_dist = results['risk_level'].value_counts()
        for level, count in risk_dist.items():
            print(f"{level} Risk: {count} customers ({count/len(results):.1%})")
        
        print("\n💰 Revenue at Risk:")
        high_risk = results[results['risk_level'] == 'High']
        if len(high_risk) > 0:
            monthly_revenue_at_risk = high_risk['monthly_charges'].sum()
            print(f"Monthly revenue at risk (High risk customers): ${monthly_revenue_at_risk:,.2f}")
        
        print("\n🔍 Top 10 Highest Risk Customers:")
        top_risk = results.nlargest(10, 'churn_probability')[
            ['customer_id', 'churn_probability', 'monthly_charges', 'tenure_months', 'contract_type']
        ]
        print(top_risk.to_string(index=False))
        
        return results
    
    def save_predictions(self, results, filename='predictions.csv'):
        """Save predictions to file and optionally to S3"""
        if results is None:
            return
        
        # Save locally
        results.to_csv(filename, index=False)
        print(f"✅ Predictions saved to {filename}")
        
        # Upload to S3 if configured
        if self.config:
            try:
                s3_client = boto3.client('s3')
                s3_key = f"predictions/{filename}"
                s3_client.upload_file(filename, self.config['bucket_name'], s3_key)
                s3_uri = f"s3://{self.config['bucket_name']}/{s3_key}"
                print(f"✅ Predictions uploaded to {s3_uri}")
            except Exception as e:
                print(f"⚠️ Could not upload to S3: {e}")

def main():
    """Main prediction pipeline"""
    print("🔮 Starting Churn Prediction")
    
    predictor = ChurnPredictor()
    
    # Check if we have customers to predict
    customer_file = 'data/customers_to_predict.csv'
    if not os.path.exists(customer_file):
        print(f"❌ Customer data file not found: {customer_file}")
        return
    
    # Choose prediction method
    print("\nChoose prediction method:")
    print("1. Local model (faster)")
    print("2. SageMaker endpoint (cloud-based)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        print("\n🔧 Using local model for predictions...")
        results = predictor.predict_local(customer_file)
    elif choice == '2':
        print("\n☁️ Using SageMaker endpoint for predictions...")
        results = predictor.predict_sagemaker(customer_file)
    else:
        print("❌ Invalid choice. Using local model.")
        results = predictor.predict_local(customer_file)
    
    if results is not None:
        # Analyze and save results
        predictor.analyze_predictions(results)
        predictor.save_predictions(results)
        
        print("\n✅ Prediction pipeline completed successfully!")
    else:
        print("❌ Prediction failed. Please check your setup.")

if __name__ == "__main__":
    main()