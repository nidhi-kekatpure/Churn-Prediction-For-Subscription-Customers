
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
