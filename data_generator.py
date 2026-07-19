"""
Synthetic AQI and Weather Data Generator
Generates 2 years of realistic hourly data with seasonal and diurnal patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_data(output_path='historical_aqi.csv', years=2):
    """
    Generate synthetic hourly AQI and meteorological data.
    
    Parameters:
    -----------
    output_path : str
        Path to save the generated CSV file
    years : int
        Number of years of data to generate (default: 2)
    
    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    
    print(f"Generating {years} years of synthetic hourly data...")
    
    # Calculate total hours
    total_hours = years * 365 * 24
    start_date = datetime(2022, 1, 1, 0, 0, 0)
    
    # Create datetime index
    dates = [start_date + timedelta(hours=i) for i in range(total_hours)]
    df = pd.DataFrame({'timestamp': dates})
    
    # Extract temporal components
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # ========== TEMPERATURE ==========
    # Base temperature with seasonal variation (warmer in summer, colder in winter)
    seasonal_temp = 20 + 15 * np.sin(2 * np.pi * (df['day_of_year'] - 80) / 365)
    
    # Diurnal variation (cooler at night, warmer during day)
    diurnal_temp = 5 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)
    
    # Random noise
    temp_noise = np.random.normal(0, 2, total_hours)
    
    df['temperature'] = seasonal_temp + diurnal_temp + temp_noise
    df['temperature'] = np.clip(df['temperature'], -5, 45)
    
    # ========== HUMIDITY ==========
    # Humidity inversely related to temperature (roughly)
    base_humidity = 60 - 0.5 * df['temperature']
    
    # Diurnal variation (higher at night, lower during day)
    diurnal_humidity = 10 * np.cos(2 * np.pi * (df['hour'] - 14) / 24)
    
    # Random noise
    humidity_noise = np.random.normal(0, 5, total_hours)
    
    df['humidity'] = base_humidity + diurnal_humidity + humidity_noise
    df['humidity'] = np.clip(df['humidity'], 20, 100)
    
    # ========== WIND SPEED ==========
    # Base wind speed with random variation
    base_wind = 10 + 5 * np.sin(2 * np.pi * df['day_of_year'] / 365)
    wind_noise = np.random.exponential(3, total_hours)
    
    df['wind_speed'] = base_wind + wind_noise
    df['wind_speed'] = np.clip(df['wind_speed'], 0, 50)
    
    # ========== PM2.5 ==========
    # Base PM2.5 with seasonal variation (higher in winter)
    seasonal_pm25 = 30 + 20 * np.sin(2 * np.pi * (df['day_of_year'] + 90) / 365)
    
    # Diurnal variation (higher during rush hours: morning 8-10, evening 18-20)
    diurnal_pm25 = 15 * (
        np.exp(-((df['hour'] - 8) ** 2) / 4) +
        np.exp(-((df['hour'] - 18) ** 2) / 4)
    )
    
    # Weekend effect (lower on weekends)
    weekend_effect = np.where(df['day_of_week'] >= 5, -10, 0)
    
    # Random noise and occasional spikes
    pm25_noise = np.random.normal(0, 8, total_hours)
    pm25_spikes = np.random.choice([0, 1], size=total_hours, p=[0.98, 0.02])
    pm25_spike_values = np.where(pm25_spikes == 1, np.random.uniform(30, 60, total_hours), 0)
    
    df['pm25'] = seasonal_pm25 + diurnal_pm25 + weekend_effect + pm25_noise + pm25_spike_values
    df['pm25'] = np.clip(df['pm25'], 5, 200)
    
    # ========== PM10 ==========
    # PM10 correlated with PM2.5 but higher values
    df['pm10'] = df['pm25'] * 1.5 + np.random.normal(10, 15, total_hours)
    df['pm10'] = np.clip(df['pm10'], 10, 300)
    
    # ========== NO2 ==========
    # NO2 follows traffic patterns (rush hours)
    seasonal_no2 = 20 + 10 * np.sin(2 * np.pi * (df['day_of_year'] + 90) / 365)
    diurnal_no2 = 15 * (
        np.exp(-((df['hour'] - 8) ** 2) / 3) +
        np.exp(-((df['hour'] - 18) ** 2) / 3)
    )
    weekend_no2 = np.where(df['day_of_week'] >= 5, -8, 0)
    no2_noise = np.random.normal(0, 5, total_hours)
    
    df['no2'] = seasonal_no2 + diurnal_no2 + weekend_no2 + no2_noise
    df['no2'] = np.clip(df['no2'], 5, 100)
    
    # ========== CO ==========
    # CO correlated with NO2
    df['co'] = 0.5 + 0.3 * (df['no2'] / 20) + np.random.normal(0, 0.2, total_hours)
    df['co'] = np.clip(df['co'], 0.1, 3.0)
    
    # ========== SO2 ==========
    # SO2 higher in winter, industrial areas
    seasonal_so2 = 15 + 10 * np.sin(2 * np.pi * (df['day_of_year'] + 90) / 365)
    so2_noise = np.random.normal(0, 4, total_hours)
    so2_spikes = np.random.choice([0, 1], size=total_hours, p=[0.97, 0.03])
    so2_spike_values = np.where(so2_spikes == 1, np.random.uniform(15, 30, total_hours), 0)
    
    df['so2'] = seasonal_so2 + so2_noise + so2_spike_values
    df['so2'] = np.clip(df['so2'], 3, 80)
    
    # ========== O3 ==========
    # O3 higher during sunny afternoons
    seasonal_o3 = 40 + 15 * np.sin(2 * np.pi * (df['day_of_year'] - 150) / 365)
    diurnal_o3 = 20 * np.exp(-((df['hour'] - 14) ** 2) / 8)
    o3_noise = np.random.normal(0, 8, total_hours)
    
    df['o3'] = seasonal_o3 + diurnal_o3 + o3_noise
    df['o3'] = np.clip(df['o3'], 10, 120)
    
    # ========== AQI CALCULATION ==========
    # Simplified AQI calculation based on PM2.5 (primary pollutant)
    # Using US EPA AQI breakpoints
    def calculate_aqi(pm25):
        """Calculate AQI from PM2.5 concentration"""
        if pm25 <= 12.0:
            aqi = (50 / 12.0) * pm25
        elif pm25 <= 35.4:
            aqi = 50 + ((100 - 50) / (35.4 - 12.0)) * (pm25 - 12.0)
        elif pm25 <= 55.4:
            aqi = 100 + ((150 - 100) / (55.4 - 35.4)) * (pm25 - 35.4)
        elif pm25 <= 150.4:
            aqi = 150 + ((200 - 150) / (150.4 - 55.4)) * (pm25 - 55.4)
        elif pm25 <= 250.4:
            aqi = 200 + ((300 - 200) / (250.4 - 150.4)) * (pm25 - 150.4)
        else:
            aqi = 300 + ((500 - 300) / (500.4 - 250.4)) * (pm25 - 250.4)
        return np.clip(aqi, 0, 500)
    
    df['aqi'] = df['pm25'].apply(calculate_aqi)
    
    # Add some correlation with other pollutants
    df['aqi'] = df['aqi'] + 0.1 * df['no2'] + 0.05 * df['so2'] + np.random.normal(0, 5, total_hours)
    df['aqi'] = np.clip(df['aqi'], 0, 500)
    
    # ========== ADD MISSING VALUES ==========
    # Introduce realistic missing value patterns (2-3% missing)
    missing_mask = np.random.choice([True, False], size=total_hours, p=[0.025, 0.975])
    
    for col in ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3', 'temperature', 'humidity', 'wind_speed']:
        col_missing = missing_mask & np.random.choice([True, False], size=total_hours, p=[0.3, 0.7])
        df.loc[col_missing, col] = np.nan
    
    # Add some consecutive missing values (simulating sensor failures)
    for _ in range(20):
        start_idx = np.random.randint(0, total_hours - 10)
        length = np.random.randint(3, 10)
        col = np.random.choice(['pm25', 'pm10', 'no2', 'co'])
        df.loc[start_idx:start_idx+length, col] = np.nan
    
    # Drop temporary columns
    df = df.drop(['hour', 'day', 'month', 'day_of_week', 'day_of_year'], axis=1)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Data generated and saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Missing values per column:")
    print(df.isnull().sum())
    
    return df


if __name__ == "__main__":
    generate_synthetic_data()