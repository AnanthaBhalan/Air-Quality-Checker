"""
Quick Training Script - Optimized for Speed
Trains models with reduced hyperparameter grid for faster completion.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
import xgboost as xgb

from feature_engineering import prepare_features


def quick_train_pipeline(data_path: str = 'historical_aqi.csv',
                        target_col: str = 'aqi',
                        model_save_path: str = 'models/best_model.pkl',
                        scaler_save_path: str = 'models/scaler.pkl') -> Dict[str, Any]:
    """
    Quick training pipeline with optimized hyperparameters.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
    target_col : str
        Name of target variable
    model_save_path : str
        Path to save the best model
    scaler_save_path : str
        Path to save the scaler
    
    Returns:
    --------
    Dict with training results
    """
    
    print("\n" + "="*60)
    print("AQI FORECASTING - QUICK TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Load and prepare features
    print("Step 1: Loading and preparing data...")
    df = pd.read_csv(data_path)
    X, y = prepare_features(df, target_col=target_col)
    
    # Step 2: Temporal split
    print("\nStep 2: Creating temporal train/test split...")
    n = len(X)
    train_size = int(n * 0.85)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    print(f"  Training: {len(X_train)} samples (85%)")
    print(f"  Test: {len(X_test)} samples (15%)")
    
    # Step 3: Scale features
    print("\nStep 3: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled")
    
    # Step 4: Train models
    print("\nStep 4: Training models...")
    
    # Model 1: Ridge Regression
    print("\n  Training Ridge Regression...")
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    ridge_r2 = r2_score(y_test, ridge_pred)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    print(f"  ✓ Ridge R²: {ridge_r2:.4f}, MAE: {ridge_mae:.2f}")
    
    # Model 2: LightGBM (optimized)
    print("\n  Training LightGBM...")
    lgbm = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_train_scaled, y_train)
    lgbm_pred = lgbm.predict(X_test_scaled)
    lgbm_r2 = r2_score(y_test, lgbm_pred)
    lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
    print(f"  ✓ LightGBM R²: {lgbm_r2:.4f}, MAE: {lgbm_mae:.2f}")
    
    # Model 3: XGBoost (optimized)
    print("\n  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    print(f"  ✓ XGBoost R²: {xgb_r2:.4f}, MAE: {xgb_mae:.2f}")
    
    # Step 5: Select best model
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    models = {
        'Ridge': (ridge, ridge_r2),
        'LightGBM': (lgbm, lgbm_r2),
        'XGBoost': (xgb_model, xgb_r2)
    }
    
    best_model_name = max(models, key=lambda k: models[k][1])
    best_model = models[best_model_name][0]
    
    print(f"  ✓ Best model: {best_model_name} (R² = {models[best_model_name][1]:.4f})")
    print("="*60)
    
    # Step 6: Save artifacts
    print("\nStep 5: Saving model and scaler...")
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"  ✓ Model saved to {model_save_path}")
    print(f"  ✓ Scaler saved to {scaler_save_path}")
    
    # Prepare results
    results = {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'models': {
            'Ridge': {
                'model': ridge,
                'metrics': {'mae': ridge_mae, 'rmse': np.sqrt(mean_squared_error(y_test, ridge_pred)), 'r2': ridge_r2}
            },
            'LightGBM': {
                'model': lgbm,
                'metrics': {'mae': lgbm_mae, 'rmse': np.sqrt(mean_squared_error(y_test, lgbm_pred)), 'r2': lgbm_r2}
            },
            'XGBoost': {
                'model': xgb_model,
                'metrics': {'mae': xgb_mae, 'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)), 'r2': xgb_r2}
            }
        },
        'test_data': {
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test
        },
        'train_val_test_split': {
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    }
    
    # Save results
    joblib.dump(results, 'models/training_results.pkl')
    print(f"  ✓ Training results saved to models/training_results.pkl")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    results = quick_train_pipeline()