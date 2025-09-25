import boto3
import json
import time
from botocore.exceptions import ClientError

class AWSChurnProjectSetup:
    def __init__(self, region='us-east-1', project_name='churn-prediction'):
        self.region = region
        self.project_name = project_name
        self.s3_client = boto3.client('s3', region_name=region)
        self.iam_client = boto3.client('iam', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        # Generate unique bucket name
        import uuid
        self.bucket_name = f"{project_name}-{str(uuid.uuid4())[:8]}"
        self.role_name = f"{project_name}-sagemaker-role"
    
    def create_s3_bucket(self):
        """Create S3 bucket for storing data and model artifacts"""
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            # Enable versioning
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            print(f"✅ S3 bucket created: {self.bucket_name}")
            return self.bucket_name
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyExists':
                print(f"❌ Bucket name {self.bucket_name} already exists. Try again.")
                return None
            else:
                print(f"❌ Error creating bucket: {e}")
                return None
    
    def create_sagemaker_role(self):
        """Create IAM role for SageMaker with necessary permissions"""
        
        # Trust policy for SageMaker
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Permissions policy
        permissions_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{self.bucket_name}",
                        f"arn:aws:s3:::{self.bucket_name}/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        try:
            # Create role
            role_response = self.iam_client.create_role(
                RoleName=self.role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=f'SageMaker execution role for {self.project_name}'
            )
            
            # Attach AWS managed policy
            self.iam_client.attach_role_policy(
                RoleName=self.role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            # Create and attach custom policy
            policy_name = f"{self.role_name}-policy"
            self.iam_client.put_role_policy(
                RoleName=self.role_name,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(permissions_policy)
            )
            
            role_arn = role_response['Role']['Arn']
            print(f"✅ SageMaker role created: {role_arn}")
            
            # Wait for role to be available
            print("⏳ Waiting for role to be available...")
            time.sleep(10)
            
            return role_arn
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                # Get existing role ARN
                role_response = self.iam_client.get_role(RoleName=self.role_name)
                role_arn = role_response['Role']['Arn']
                print(f"✅ Using existing SageMaker role: {role_arn}")
                return role_arn
            else:
                print(f"❌ Error creating role: {e}")
                return None
    
    def upload_data_to_s3(self, local_file_path, s3_key):
        """Upload data files to S3"""
        try:
            self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key)
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            print(f"✅ Uploaded {local_file_path} to {s3_uri}")
            return s3_uri
        except Exception as e:
            print(f"❌ Error uploading {local_file_path}: {e}")
            return None
    
    def setup_project(self):
        """Complete project setup"""
        print(f"🚀 Setting up AWS resources for {self.project_name}...")
        
        # Create S3 bucket
        bucket = self.create_s3_bucket()
        if not bucket:
            return None
        
        # Create SageMaker role
        role_arn = self.create_sagemaker_role()
        if not role_arn:
            return None
        
        # Upload training data
        train_s3_uri = self.upload_data_to_s3('data/train_data.csv', 'data/train_data.csv')
        test_s3_uri = self.upload_data_to_s3('data/customers_to_predict.csv', 'data/customers_to_predict.csv')
        
        config = {
            'bucket_name': self.bucket_name,
            'role_arn': role_arn,
            'region': self.region,
            'train_data_s3_uri': train_s3_uri,
            'test_data_s3_uri': test_s3_uri,
            'model_output_path': f"s3://{self.bucket_name}/model-output/",
            'predictions_output_path': f"s3://{self.bucket_name}/predictions/"
        }
        
        # Save config
        with open('aws_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ AWS setup complete!")
        print(f"📄 Configuration saved to aws_config.json")
        
        return config
    
    def cleanup_resources(self):
        """Clean up AWS resources (use with caution!)"""
        print("🧹 Cleaning up AWS resources...")
        
        try:
            # Delete S3 bucket contents
            objects = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            if 'Contents' in objects:
                delete_keys = [{'Key': obj['Key']} for obj in objects['Contents']]
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': delete_keys}
                )
            
            # Delete bucket
            self.s3_client.delete_bucket(Bucket=self.bucket_name)
            print(f"✅ Deleted S3 bucket: {self.bucket_name}")
            
            # Detach policies and delete role
            self.iam_client.detach_role_policy(
                RoleName=self.role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            self.iam_client.delete_role_policy(
                RoleName=self.role_name,
                PolicyName=f"{self.role_name}-policy"
            )
            
            self.iam_client.delete_role(RoleName=self.role_name)
            print(f"✅ Deleted IAM role: {self.role_name}")
            
        except ClientError as e:
            print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    setup = AWSChurnProjectSetup()
    config = setup.setup_project()
    
    if config:
        print("\n🎉 Ready to start training your churn model!")
        print("Next steps:")
        print("1. Run the training script: python src/train.py")
        print("2. Deploy the model endpoint")
        print("3. Make predictions on new customers")