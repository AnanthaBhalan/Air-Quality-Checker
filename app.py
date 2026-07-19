"""
Interactive Streamlit Dashboard for AQI Forecasting
Allows users to view forecasts and perform what-if scenario analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="AQI Forecasting Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_scaler(model_path='models/best_model.pkl', 
                         scaler_path='models/scaler.pkl'):
    """
    Load model and scaler (cached for performance).
    
    Parameters:
    -----------
    model_path : str
        Path to model file
    scaler_path : str
        Path to scaler file
    
    Returns:
    --------
    Tuple of (model, scaler)
    """
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run train_model.py first.")
        return None, None
    
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found at {scaler_path}. Please run train_model.py first.")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler


@st.cache_data
def load_test_data(data_path='historical_aqi.csv'):
    """
    Load and prepare test data.
    
    Parameters:
    -----------
    data_path : str
        Path to data CSV
    
    Returns:
    --------
    pd.DataFrame with test data
    """
    
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}. Please run data_generator.py first.")
        return None
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def prepare_forecast_input(current_data: dict, 
                          feature_names: list,
                          scaler) -> np.ndarray:
    """
    Prepare input features for forecasting.
    
    Parameters:
    -----------
    current_data : dict
        Dictionary with current weather and pollutant values
    feature_names : list
        List of feature names in correct order
    scaler : fitted scaler
        StandardScaler object
    
    Returns:
    --------
    np.ndarray with scaled features
    """
    
    # Create feature vector
    features = np.zeros(len(feature_names))
    
    # Map current data to features
    feature_dict = {name: 0.0 for name in feature_names}
    
    # Update with current values
    for key, value in current_data.items():
        if key in feature_dict:
            feature_dict[key] = value
    
    # Create array in correct order
    for i, name in enumerate(feature_names):
        features[i] = feature_dict[name]
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    return features_scaled


def create_forecast_plot(timestamps, actual_values, predicted_values, 
                        forecast_timestamps=None, forecast_values=None,
                        title="AQI Forecast"):
    """
    Create interactive Plotly chart for AQI forecast.
    
    Parameters:
    -----------
    timestamps : array-like
        Historical timestamps
    actual_values : array-like
        Historical actual AQI values
    predicted_values : array-like
        Historical predicted AQI values
    forecast_timestamps : array-like, optional
        Future timestamps for forecast
    forecast_values : array-like, optional
        Forecasted AQI values
    title : str
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    
    fig = go.Figure()
    
    # Plot actual values
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=actual_values,
        mode='lines',
        name='Actual AQI',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Actual</b><br>Time: %{x}<br>AQI: %{y:.1f}<extra></extra>'
    ))
    
    # Plot predicted values
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=predicted_values,
        mode='lines',
        name='Predicted AQI',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        hovertemplate='<b>Predicted</b><br>Time: %{x}<br>AQI: %{y:.1f}<extra></extra>'
    ))
    
    # Plot forecast if provided
    if forecast_timestamps is not None and forecast_values is not None:
        fig.add_trace(go.Scatter(
            x=forecast_timestamps,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#d62728', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Forecast</b><br>Time: %{x}<br>AQI: %{y:.1f}<extra></extra>'
        ))
        
        # Add vertical line to separate historical and forecast
        if len(timestamps) > 0 and len(forecast_timestamps) > 0:
            fig.add_vline(
                x=timestamps.iloc[-1] if hasattr(timestamps, 'iloc') else timestamps[-1],
                line_dash="dash",
                line_color="gray",
                annotation_text="Forecast Start"
            )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f77b4')),
        xaxis_title='Time',
        yaxis_title='AQI',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add AQI category background colors
    fig.add_hrect(y0=0, y1=50, line_width=0, fillcolor="green", opacity=0.1, 
                  annotation_text="Good", annotation_position="left")
    fig.add_hrect(y0=50, y1=100, line_width=0, fillcolor="yellow", opacity=0.1,
                  annotation_text="Moderate", annotation_position="left")
    fig.add_hrect(y0=100, y1=150, line_width=0, fillcolor="orange", opacity=0.1,
                  annotation_text="Unhealthy for Sensitive Groups", annotation_position="left")
    fig.add_hrect(y0=150, y1=200, line_width=0, fillcolor="red", opacity=0.1,
                  annotation_text="Unhealthy", annotation_position="left")
    fig.add_hrect(y0=200, y1=300, line_width=0, fillcolor="purple", opacity=0.1,
                  annotation_text="Very Unhealthy", annotation_position="left")
    fig.add_hrect(y0=300, y1=500, line_width=0, fillcolor="maroon", opacity=0.1,
                  annotation_text="Hazardous", annotation_position="left")
    
    return fig


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<p class="main-header">🌍 AQI Forecasting Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time Air Quality Index Prediction & What-If Analysis</p>', 
                unsafe_allow_html=True)
    
    # Load model and data
    model, scaler = load_model_and_scaler()
    df = load_test_data()
    
    if model is None or df is None:
        st.stop()
    
    # Load training results to get feature names
    training_results = joblib.load('models/training_results.pkl')
    feature_names = training_results['feature_names']
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Forecast horizon
        forecast_horizon = st.slider(
            "Forecast Horizon (hours)",
            min_value=24,
            max_value=72,
            value=48,
            step=12,
            help="Select how many hours ahead to forecast"
        )
        
        st.divider()
        
        # What-If Scenario
        st.header("🔧 What-If Scenario")
        st.write("Adjust current conditions to see impact on forecast:")
        
        wind_speed = st.slider(
            "Wind Speed (km/h)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.5
        )
        
        temperature = st.slider(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=45.0,
            value=20.0,
            step=0.5
        )
        
        humidity = st.slider(
            "Humidity (%)",
            min_value=20,
            max_value=100,
            value=60,
            step=1
        )
        
        st.divider()
        st.info("💡 Adjust the sliders to see how weather conditions affect AQI predictions!")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Visualization", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.header("AQI Forecast vs Actual")
        
        # Get test data from training results
        test_data = training_results['test_data']
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Create timestamps for test set (last portion of data)
        test_size = len(y_test)
        test_timestamps = df['timestamp'].iloc[-test_size:]
        
        # Sample for visualization (max 500 points)
        max_points = 500
        if test_size > max_points:
            indices = np.linspace(0, test_size - 1, max_points, dtype=int)
            test_timestamps_plot = test_timestamps.iloc[indices]
            y_test_plot = y_test.iloc[indices]
            y_pred_plot = y_pred[indices]
        else:
            test_timestamps_plot = test_timestamps
            y_test_plot = y_test
            y_pred_plot = y_pred
        
        # Create forecast for what-if scenario
        # Use last available data point as base
        last_idx = -1
        current_data = {
            'pm25': df['pm25'].iloc[last_idx],
            'pm10': df['pm10'].iloc[last_idx],
            'no2': df['no2'].iloc[last_idx],
            'co': df['co'].iloc[last_idx],
            'so2': df['so2'].iloc[last_idx],
            'o3': df['o3'].iloc[last_idx],
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'hour_sin': np.sin(2 * np.pi * df['timestamp'].iloc[last_idx].hour / 24),
            'hour_cos': np.cos(2 * np.pi * df['timestamp'].iloc[last_idx].hour / 24),
            'day_of_week_sin': np.sin(2 * np.pi * df['timestamp'].iloc[last_idx].dayofweek / 7),
            'day_of_week_cos': np.cos(2 * np.pi * df['timestamp'].iloc[last_idx].dayofweek / 7),
            'month_sin': np.sin(2 * np.pi * df['timestamp'].iloc[last_idx].month / 12),
            'month_cos': np.cos(2 * np.pi * df['timestamp'].iloc[last_idx].month / 12),
        }
        
        # Add lag features (using last known values)
        for lag in [1, 2, 3, 6, 12, 24]:
            if lag <= len(df):
                current_data[f'pm25_lag_{lag}h'] = df['pm25'].iloc[-lag]
                current_data[f'aqi_lag_{lag}h'] = df['aqi'].iloc[-lag]
            else:
                current_data[f'pm25_lag_{lag}h'] = df['pm25'].iloc[0]
                current_data[f'aqi_lag_{lag}h'] = df['aqi'].iloc[0]
        
        # Add rolling features (using last known values)
        for col in ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']:
            for window in [6, 24]:
                if window <= len(df):
                    rolling_mean = df[col].iloc[-window:].mean()
                    rolling_std = df[col].iloc[-window:].std()
                else:
                    rolling_mean = df[col].mean()
                    rolling_std = df[col].std()
                
                current_data[f'{col}_rolling_mean_{window}h'] = rolling_mean
                current_data[f'{col}_rolling_std_{window}h'] = rolling_std
        
        # Prepare input for forecasting
        features = prepare_forecast_input(current_data, feature_names, scaler)
        
        # Generate forecast
        forecast_values = []
        forecast_timestamps = []
        
        current_timestamp = df['timestamp'].iloc[-1]
        
        for i in range(forecast_horizon):
            # Predict
            pred = model.predict(features)[0]
            forecast_values.append(pred)
            
            # Update timestamp
            current_timestamp = current_timestamp + pd.Timedelta(hours=1)
            forecast_timestamps.append(current_timestamp)
            
            # Update features for next prediction (simplified - in production, use actual updated values)
            # Update lag features
            for lag in [24, 12, 6, 3, 2, 1]:
                if lag == 1:
                    current_data['pm25_lag_1h'] = current_data['pm25']
                    current_data['aqi_lag_1h'] = pred
                else:
                    lag_key = f'pm25_lag_{lag}h'
                    prev_lag_key = f'pm25_lag_{lag-1}h'
                    if lag_key in current_data and prev_lag_key in current_data:
                        current_data[lag_key] = current_data[prev_lag_key]
        
        # Create plot
        fig = create_forecast_plot(
            timestamps=test_timestamps_plot,
            actual_values=y_test_plot.values,
            predicted_values=y_pred_plot,
            forecast_timestamps=forecast_timestamps,
            forecast_values=forecast_values,
            title=f"AQI Forecast - Next {forecast_horizon} Hours"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current AQI",
                value=f"{df['aqi'].iloc[-1]:.1f}",
                delta=None
            )
        
        with col2:
            avg_forecast = np.mean(forecast_values)
            st.metric(
                label=f"Avg Forecast ({forecast_horizon}h)",
                value=f"{avg_forecast:.1f}",
                delta=f"{avg_forecast - df['aqi'].iloc[-1]:.1f}"
            )
        
        with col3:
            max_forecast = np.max(forecast_values)
            st.metric(
                label="Max Forecast AQI",
                value=f"{max_forecast:.1f}",
                delta=None
            )
        
        with col4:
            min_forecast = np.min(forecast_values)
            st.metric(
                label="Min Forecast AQI",
                value=f"{min_forecast:.1f}",
                delta=None
            )
        
        # AQI Category
        def get_aqi_category(aqi):
            if aqi <= 50:
                return "Good", "🟢"
            elif aqi <= 100:
                return "Moderate", "🟡"
            elif aqi <= 150:
                return "Unhealthy for Sensitive Groups", "🟠"
            elif aqi <= 200:
                return "Unhealthy", "🔴"
            elif aqi <= 300:
                return "Very Unhealthy", "🟣"
            else:
                return "Hazardous", "🟤"
        
        category, emoji = get_aqi_category(avg_forecast)
        st.info(f"**Forecast Category:** {emoji} {category}")
    
    with tab2:
        st.header("Model Performance Metrics")
        
        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f} AQI")
        
        with col2:
            st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f} AQI")
        
        with col3:
            st.metric(label="R² Score", value=f"{r2:.4f}")
        
        with col4:
            st.metric(label="Mean Absolute Percentage Error", value=f"{mape:.2f}%")
        
        st.divider()
        
        # Feature importance
        st.subheader("Top 10 Feature Importance")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            importances = np.zeros(len(feature_names))
        
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        # Plot
        fig_imp = go.Figure(go.Bar(
            x=feature_imp_df['importance'],
            y=feature_imp_df['feature'],
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig_imp.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with tab3:
        st.header("About This Dashboard")
        
        st.markdown("""
        ### 🎯 Purpose
        This dashboard provides real-time Air Quality Index (AQI) forecasting using machine learning.
        
        ### 📊 Features
        - **24-72 Hour Forecasts:** Predict AQI up to 3 days in advance
        - **What-If Analysis:** Adjust weather conditions to see impact on predictions
        - **Model Performance:** View detailed metrics and feature importance
        
        ### 🔬 Model Information
        - **Algorithm:** Best performing model from training (Ridge, LightGBM, or XGBoost)
        - **Features:** 57 features including pollutants, weather, temporal, lag, and rolling statistics
        - **Training Data:** 2 years of hourly synthetic data with realistic patterns
        
        ### 📈 How to Use
        1. **Adjust Forecast Horizon:** Use the slider in the sidebar to select 24-72 hours
        2. **What-If Scenarios:** Modify wind speed, temperature, and humidity to see predictions
        3. **View Results:** The chart shows historical actual vs predicted values and future forecast
        
        ### ⚠️ Disclaimer
        This is a demonstration project using synthetic data. For real-world applications, 
        use actual air quality monitoring data and consult environmental experts.
        """)


if __name__ == "__main__":
    main()