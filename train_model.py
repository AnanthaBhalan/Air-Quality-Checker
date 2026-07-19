"""
Model Training and Pipeline for AQI Forecasting
Implements temporal splits, trains multiple models, and performs hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, Any
from datetime import datetime

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import lightgbm as lgb
import xgboost as xgb

from feature_engineering import prepare_features


def temporal_train_val_test_split(X: pd.DataFrame, 
                                  y: pd.Series,
                                  train_ratio: float = 0.70,
                                  val_ratio: float = 0.15) -> Tuple:
    """
    Split data temporally (no shuffling) to prevent data leakage.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    train_ratio : float
        Proportion of data for training (default: 0.70)
    val_ratio : float
        Proportion of data for validation (default: 0.15)
    
    Returns:
    --------
    Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    
    n = len(X)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Temporal split (no shuffling)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]
    
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]
    
    print("="*60)
    print("TEMPORAL DATA SPLIT")
    print("="*60)
    print(f"  Training set:   {len(X_train)} samples ({len(X_train)/n*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/n*100:.1f}%)")
    print(f"  Test set:       {len(X_test)} samples ({len(X_test)/n*100:.1f}%)")
    print(f"  Total samples:  {n}")
    print("="*60 + "\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train: pd.DataFrame, 
                   X_val: pd.DataFrame, 
                   X_test: pd.DataFrame) -> Tuple:
    """
    Scale features using StandardScaler fitted on training data only.
    
    Parameters:
    -----------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices for train, validation, and test sets
    
    Returns:
    --------
    Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Features scaled using StandardScaler")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_ridge_model(X_train: np.ndarray, 
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray) -> Tuple[Ridge, Dict]:
    """
    Train Ridge Regression model with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    
    Returns:
    --------
    Tuple of (best_model, best_params)
    """
    
    print("\n" + "="*60)
    print("TRAINING RIDGE REGRESSION")
    print("="*60)
    
    # Hyperparameter grid
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    }
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid search
    ridge = Ridge()
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"  Best alpha: {best_params['alpha']}")
    print(f"  Validation RMSE: {val_rmse:.2f}")
    print(f"  Validation MAE: {val_mae:.2f}")
    print(f"  Validation R²: {val_r2:.4f}")
    print("="*60 + "\n")
    
    return best_model, best_params


def train_lightgbm_model(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray) -> Tuple[lgb.LGBMRegressor, Dict]:
    """
    Train LightGBM model with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    
    Returns:
    --------
    Tuple of (best_model, best_params)
    """
    
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM")
    print("="*60)
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31, 63],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid search
    lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1)
    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"  Best parameters: {best_params}")
    print(f"  Validation RMSE: {val_rmse:.2f}")
    print(f"  Validation MAE: {val_mae:.2f}")
    print(f"  Validation R²: {val_r2:.4f}")
    print("="*60 + "\n")
    
    return best_model, best_params


def train_xgboost_model(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    Train XGBoost model with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    
    Returns:
    --------
    Tuple of (best_model, best_params)
    """
    
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid search
    xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"  Best parameters: {best_params}")
    print(f"  Validation RMSE: {val_rmse:.2f}")
    print(f"  Validation MAE: {val_mae:.2f}")
    print(f"  Validation R²: {val_r2:.4f}")
    print("="*60 + "\n")
    
    return best_model, best_params


def evaluate_model(model: Any,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   model_name: str) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Parameters:
    -----------
    model : trained model
        Trained model object
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
    model_name : str
        Name of the model for reporting
    
    Returns:
    --------
    Dict with evaluation metrics
    """
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'model': model_name,
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    print(f"\n{model_name} Test Performance:")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    return metrics


def train_pipeline(data_path: str = 'historical_aqi.csv',
                   target_col: str = 'aqi',
                   model_save_path: str = 'models/best_model.pkl',
                   scaler_save_path: str = 'models/scaler.pkl') -> Dict[str, Any]:
    """
    Complete training pipeline.
    
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
    Dict with training results and artifacts
    """
    
    print("\n" + "="*60)
    print("AQI FORECASTING - MODEL TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Load and prepare features
    print("Step 1: Loading and preparing data...")
    df = pd.read_csv(data_path)
    X, y = prepare_features(df, target_col=target_col)
    
    # Step 2: Temporal train/val/test split
    print("\nStep 2: Creating temporal train/validation/test split...")
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(X, y)
    
    # Step 3: Scale features
    print("\nStep 3: Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Step 4: Train models
    print("\nStep 4: Training models...")
    
    # Train Ridge Regression
    ridge_model, ridge_params = train_ridge_model(X_train_scaled, y_train, X_val_scaled, y_val)
    ridge_metrics = evaluate_model(ridge_model, X_test_scaled, y_test, "Ridge Regression")
    
    # Train LightGBM
    lgbm_model, lgbm_params = train_lightgbm_model(X_train_scaled, y_train, X_val_scaled, y_val)
    lgbm_metrics = evaluate_model(lgbm_model, X_test_scaled, y_test, "LightGBM")
    
    # Train XGBoost
    xgb_model, xgb_params = train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val)
    xgb_metrics = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
    
    # Step 5: Select best model
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    models = {
        'Ridge': (ridge_model, ridge_metrics['r2']),
        'LightGBM': (lgbm_model, lgbm_metrics['r2']),
        'XGBoost': (xgb_model, xgb_metrics['r2'])
    }
    
    best_model_name = max(models, key=lambda k: models[k][1])
    best_model = models[best_model_name][0]
    
    print(f"  Best model: {best_model_name} (R² = {models[best_model_name][1]:.4f})")
    print("="*60 + "\n")
    
    # Step 6: Save artifacts
    print("Step 5: Saving model and scaler...")
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
            'Ridge': {'model': ridge_model, 'params': ridge_params, 'metrics': ridge_metrics},
            'LightGBM': {'model': lgbm_model, 'params': lgbm_params, 'metrics': lgbm_metrics},
            'XGBoost': {'model': xgb_model, 'params': xgb_params, 'metrics': xgb_metrics}
        },
        'test_data': {
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test
        },
        'train_val_test_split': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
    }
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    # Run training pipeline
    results = train_pipeline()
    
    # Save results for evaluation
    joblib.dump(results, 'models/training_results.pkl')
    print("✓ Training results saved to models/training_results.pkl")