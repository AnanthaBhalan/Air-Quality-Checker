"""
Model Evaluation and Interpretability for AQI Forecasting
Generates evaluation metrics, feature importance, and visualization plots.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    
    Returns:
    --------
    Dict with MAE, RMSE, and R² scores
    """
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Avoid division by zero
    bias = np.mean(y_pred - y_true)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2_Score': r2,
        'MAPE': mape,
        'Bias': bias
    }
    
    return metrics


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    Parameters:
    -----------
    model : trained model
        Trained model object (LightGBM, XGBoost, or Ridge)
    feature_names : list
        List of feature names
    
    Returns:
    --------
    pd.DataFrame with feature importance rankings
    """
    
    # Check if model has feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Check if model has coef_ attribute (linear models)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model type not supported for feature importance extraction")
    
    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Add rank
    feature_importance_df['rank'] = range(1, len(feature_importance_df) + 1)
    
    return feature_importance_df


def plot_forecast_vs_actual(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            timestamps: pd.Series = None,
                            save_path: str = 'outputs/forecast_vs_actual.png',
                            title: str = 'AQI Forecast vs Actual',
                            max_points: int = 500):
    """
    Plot forecast vs actual values.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    timestamps : pd.Series
        Timestamp values for x-axis (optional)
    save_path : str
        Path to save the plot
    title : str
        Plot title
    max_points : int
        Maximum number of points to plot (for performance)
    """
    
    # Sample data if too large
    if len(y_true) > max_points:
        indices = np.linspace(0, len(y_true) - 1, max_points, dtype=int)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
        if timestamps is not None:
            timestamps_plot = timestamps.iloc[indices]
        else:
            timestamps_plot = None
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        timestamps_plot = timestamps
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Time series comparison
    ax1 = axes[0, 0]
    if timestamps_plot is not None:
        ax1.plot(timestamps_plot, y_true_plot, label='Actual', alpha=0.7, linewidth=1.5)
        ax1.plot(timestamps_plot, y_pred_plot, label='Predicted', alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Time')
    else:
        ax1.plot(y_true_plot, label='Actual', alpha=0.7, linewidth=1.5)
        ax1.plot(y_pred_plot, label='Predicted', alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('AQI')
    ax1.set_title('Time Series: Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(y_true_plot, y_pred_plot, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual AQI')
    ax2.set_ylabel('Predicted AQI')
    ax2.set_title('Scatter Plot: Actual vs Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    ax3 = axes[1, 0]
    residuals = y_true_plot - y_pred_plot
    ax3.scatter(y_pred_plot, residuals, alpha=0.5, s=20)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted AQI')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Plot')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax4.set_xlabel('Prediction Error')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Prediction Errors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Forecast vs Actual plot saved to {save_path}")
    
    plt.close()


def plot_feature_importance(feature_importance_df: pd.DataFrame,
                           save_path: str = 'outputs/feature_importance.png',
                           top_n: int = 20):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        DataFrame with feature importance
    save_path : str
        Path to save the plot
    top_n : int
        Number of top features to display
    """
    
    # Select top N features
    df_plot = feature_importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Horizontal bar plot
    bars = ax.barh(range(len(df_plot)), df_plot['importance'], color='steelblue')
    
    # Customize
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        ax.text(row['importance'], i, f" {row['importance']:.4f}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to {save_path}")
    
    plt.close()


def generate_evaluation_report(model_results: Dict[str, Any],
                              feature_importance_df: pd.DataFrame,
                              save_path: str = 'outputs/evaluation_report.txt') -> str:
    """
    Generate a comprehensive evaluation report.
    
    Parameters:
    -----------
    model_results : Dict
        Dictionary containing model results and metrics
    feature_importance_df : pd.DataFrame
        DataFrame with feature importance
    save_path : str
        Path to save the report
    
    Returns:
    --------
    str: Report content
    """
    
    report = []
    report.append("="*80)
    report.append("AQI FORECASTING MODEL - EVALUATION REPORT")
    report.append("="*80)
    report.append("")
    
    # Model comparison
    report.append("MODEL PERFORMANCE COMPARISON")
    report.append("-"*80)
    
    models = model_results.get('models', {})
    for model_name, model_data in models.items():
        metrics = model_data.get('metrics', {})
        report.append(f"\n{model_name}:")
        report.append(f"  MAE:  {metrics.get('mae', 'N/A'):.2f}")
        report.append(f"  RMSE: {metrics.get('rmse', 'N/A'):.2f}")
        report.append(f"  R²:   {metrics.get('r2', 'N/A'):.4f}")
    
    report.append("")
    report.append(f"Best Model: {model_results.get('best_model_name', 'N/A')}")
    report.append("")
    
    # Data split information
    report.append("DATA SPLIT INFORMATION")
    report.append("-"*80)
    split_info = model_results.get('train_val_test_split', {})
    report.append(f"  Training samples:   {split_info.get('train_size', 'N/A')}")
    report.append(f"  Validation samples: {split_info.get('val_size', 'N/A')}")
    report.append(f"  Test samples:       {split_info.get('test_size', 'N/A')}")
    report.append("")
    
    # Feature importance
    report.append("TOP 20 FEATURE IMPORTANCE")
    report.append("-"*80)
    for idx, row in feature_importance_df.head(20).iterrows():
        report.append(f"  {row['rank']:2d}. {row['feature']:40s} {row['importance']:.6f}")
    report.append("")
    
    # Model parameters
    report.append("BEST MODEL HYPERPARAMETERS")
    report.append("-"*80)
    best_model_name = model_results.get('best_model_name', '')
    if best_model_name in models:
        params = models[best_model_name].get('params', {})
        for param, value in params.items():
            report.append(f"  {param}: {value}")
    report.append("")
    
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Save report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Evaluation report saved to {save_path}")
    
    return report_text


def run_evaluation(model_path: str = 'models/best_model.pkl',
                  scaler_path: str = 'models/scaler.pkl',
                  results_path: str = 'models/training_results.pkl',
                  data_path: str = 'historical_aqi.csv',
                  output_dir: str = 'outputs') -> Dict[str, Any]:
    """
    Run complete evaluation pipeline.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
    scaler_path : str
        Path to saved scaler
    results_path : str
        Path to saved training results
    data_path : str
        Path to data CSV
    output_dir : str
        Directory to save outputs
    
    Returns:
    --------
    Dict with evaluation results
    """
    
    print("\n" + "="*80)
    print("AQI FORECASTING - MODEL EVALUATION")
    print("="*80 + "\n")
    
    # Load artifacts
    print("Loading model artifacts...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    training_results = joblib.load(results_path)
    
    feature_names = training_results['feature_names']
    test_data = training_results['test_data']
    scaler = training_results['scaler']
    
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    print(f"✓ Model loaded: {training_results['best_model_name']}")
    print(f"✓ Test data shape: {X_test.shape}")
    
    # Scale test data using the saved scaler
    print("\nScaling test data...")
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics = calculate_metrics(y_test, y_pred)
    
    print("\n" + "="*80)
    print("TEST SET PERFORMANCE")
    print("="*80)
    print(f"  MAE:  {metrics['MAE']:.2f} AQI units")
    print(f"  RMSE: {metrics['RMSE']:.2f} AQI units")
    print(f"  R²:   {metrics['R2_Score']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Bias: {metrics['Bias']:.2f} AQI units")
    print("="*80)
    
    # Feature importance
    print("\nExtracting feature importance...")
    feature_importance_df = get_feature_importance(model, feature_names)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Forecast vs Actual plot
    plot_forecast_vs_actual(
        y_true=y_test.values,
        y_pred=y_pred,
        timestamps=None,  # Can be added if timestamp info is available
        save_path=os.path.join(output_dir, 'forecast_vs_actual.png'),
        title=f"AQI Forecast vs Actual - {training_results['best_model_name']}"
    )
    
    # Feature importance plot
    plot_feature_importance(
        feature_importance_df=feature_importance_df,
        save_path=os.path.join(output_dir, 'feature_importance.png'),
        top_n=20
    )
    
    # Generate report
    print("\nGenerating evaluation report...")
    report_text = generate_evaluation_report(
        model_results=training_results,
        feature_importance_df=feature_importance_df,
        save_path=os.path.join(output_dir, 'evaluation_report.txt')
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to {output_dir}/:")
    print("  - forecast_vs_actual.png")
    print("  - feature_importance.png")
    print("  - evaluation_report.txt")
    print()
    
    # Return results
    evaluation_results = {
        'metrics': metrics,
        'feature_importance': feature_importance_df,
        'y_true': y_test,
        'y_pred': y_pred,
        'report': report_text
    }
    
    return evaluation_results


if __name__ == "__main__":
    # Run evaluation
    results = run_evaluation()
    
    # Print report
    print("\n" + results['report'])