"""
Feature Engineering for AQI Forecasting
Handles missing values, creates temporal, lag, and rolling window features.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def handle_missing_values(df: pd.DataFrame, 
                         interpolation_threshold: int = 6) -> pd.DataFrame:
    """
    Handle missing values using linear interpolation for short gaps 
    and forward-fill for longer gaps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with missing values
    interpolation_threshold : int
        Maximum consecutive missing values to interpolate (default: 6 hours)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("Handling missing values...")
    print(f"  Missing values before: {df[numeric_cols].isnull().sum().sum()}")
    
    for col in numeric_cols:
        if df[col].isnull().any():
            # Find consecutive missing value groups
            missing_mask = df[col].isnull()
            missing_groups = (missing_mask != missing_mask.shift()).cumsum()
            group_sizes = missing_mask.groupby(missing_groups).sum()
            
            # Interpolate short gaps (<= threshold)
            short_gap_groups = group_sizes[group_sizes <= interpolation_threshold].index
            for group in short_gap_groups:
                group_mask = missing_groups == group
                if group_mask.any():
                    df.loc[group_mask, col] = df.loc[group_mask, col].interpolate(method='linear', limit_direction='both')
            
            # Forward-fill longer gaps
            long_gap_groups = group_sizes[group_sizes > interpolation_threshold].index
            for group in long_gap_groups:
                group_mask = missing_groups == group
                if group_mask.any():
                    df.loc[group_mask, col] = df.loc[group_mask, col].ffill().bfill()
    
    print(f"  Missing values after: {df[numeric_cols].isnull().sum().sum()}")
    
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical temporal features using sine/cosine encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'timestamp' column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with temporal features added
    """
    
    df = df.copy()
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract temporal components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Cyclical encoding for hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week (7-day cycle)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Cyclical encoding for month (12-month cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop original temporal columns (keep only cyclical)
    df = df.drop(['hour', 'day_of_week', 'month'], axis=1)
    
    print("✓ Created temporal features (hour, day_of_week, month with sine/cosine encoding)")
    
    return df


def create_lag_features(df: pd.DataFrame, 
                       target_col: str = 'aqi',
                       lag_cols: List[str] = None,
                       lag_hours: List[int] = None) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name (default: 'aqi')
    lag_cols : List[str]
        List of columns to create lags for (default: ['pm25', 'aqi'])
    lag_hours : List[int]
        List of lag hours (default: [1, 2, 3, 6, 12, 24])
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with lag features added
    """
    
    if lag_cols is None:
        lag_cols = ['pm25', target_col]
    
    if lag_hours is None:
        lag_hours = [1, 2, 3, 6, 12, 24]
    
    df = df.copy()
    
    print(f"Creating lag features for {lag_cols} at hours {lag_hours}...")
    
    for col in lag_cols:
        if col in df.columns:
            for lag in lag_hours:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    print(f"✓ Created {len(lag_cols) * len(lag_hours)} lag features")
    
    return df


def create_rolling_features(df: pd.DataFrame,
                           feature_cols: List[str] = None,
                           windows: List[int] = None) -> pd.DataFrame:
    """
    Create rolling window statistics (mean and std) for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : List[str]
        List of columns to create rolling features for (default: pollutant columns)
    windows : List[int]
        List of window sizes in hours (default: [6, 24])
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with rolling features added
    """
    
    if feature_cols is None:
        feature_cols = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']
    
    if windows is None:
        windows = [6, 24]
    
    df = df.copy()
    
    print(f"Creating rolling features for {feature_cols} with windows {windows}...")
    
    for col in feature_cols:
        if col in df.columns:
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
                # Rolling standard deviation
                df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
    
    num_features = len(feature_cols) * len(windows) * 2
    print(f"✓ Created {num_features} rolling features")
    
    return df


def prepare_features(df: pd.DataFrame, 
                    target_col: str = 'aqi',
                    drop_timestamp: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input dataframe
    target_col : str
        Name of target variable
    drop_timestamp : bool
        Whether to drop timestamp column from features
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y
    """
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Step 1: Handle missing values
    df = handle_missing_values(df)
    
    # Step 2: Create temporal features
    df = create_temporal_features(df)
    
    # Step 3: Create lag features
    df = create_lag_features(df, target_col=target_col)
    
    # Step 4: Create rolling features
    df = create_rolling_features(df)
    
    # Step 5: Drop rows with NaN (from lag features)
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(df)
    
    print(f"\n✓ Dropped {dropped_rows} rows with NaN values")
    print(f"  Final dataset shape: {df.shape}")
    
    # Separate features and target
    if drop_timestamp and 'timestamp' in df.columns:
        df = df.drop('timestamp', axis=1)
    
    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Target vector shape: {y.shape}")
    print(f"  Features: {list(X.columns)}")
    print("="*60 + "\n")
    
    return X, y


def get_feature_names() -> List[str]:
    """
    Return list of all feature names in the final dataset.
    
    Returns:
    --------
    List[str]
        List of feature names
    """
    
    # Base features
    base_features = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3', 
                     'temperature', 'humidity', 'wind_speed']
    
    # Temporal features
    temporal_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 
                        'day_of_week_cos', 'month_sin', 'month_cos']
    
    # Lag features
    lag_cols = ['pm25', 'aqi']
    lag_hours = [1, 2, 3, 6, 12, 24]
    lag_features = [f'{col}_lag_{lag}h' for col in lag_cols for lag in lag_hours]
    
    # Rolling features
    rolling_cols = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']
    windows = [6, 24]
    rolling_features = [f'{col}_rolling_{stat}_{window}h' 
                       for col in rolling_cols 
                       for stat in ['mean', 'std'] 
                       for window in windows]
    
    all_features = base_features + temporal_features + lag_features + rolling_features
    
    return all_features


if __name__ == "__main__":
    # Test the feature engineering pipeline
    from data_generator import generate_synthetic_data
    
    print("Testing feature engineering pipeline...")
    df = generate_synthetic_data('test_data.csv', years=1)
    X, y = prepare_features(df)
    
    print(f"\nFeature engineering complete!")
    print(f"X shape: {X.shape}, y shape: {y.shape}")