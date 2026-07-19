# 🌍 AQI Forecasting Machine Learning Project

An end-to-end production-ready machine learning system for forecasting Air Quality Index (AQI) 24-72 hours into the future using historical pollutant and meteorological data.
Intern ID: CTTS152
## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Execution Steps](#detailed-execution-steps)
- [Dashboard Usage](#dashboard-usage)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a complete ML pipeline for AQI forecasting, including:

- **Synthetic Data Generation**: 2 years of realistic hourly AQI and weather data with seasonal/diurnal patterns
- **Feature Engineering**: Temporal, lag, and rolling window features with proper missing value handling
- **Model Training**: Multiple models (Ridge, LightGBM, XGBoost) with hyperparameter tuning
- **Evaluation**: Comprehensive metrics and interpretability analysis
- **Interactive Dashboard**: Streamlit-based web app for forecasting and what-if analysis

## ✨ Features

### Data Generation
- Realistic seasonal temperature variations (winter smog, summer peaks)
- Diurnal patterns (rush hour traffic peaks)
- Random pollutant spikes and sensor failures
- 2-3% missing values for realistic data quality issues

### Feature Engineering
- **Temporal Features**: Cyclical sine/cosine encoding for hour, day of week, month
- **Lag Features**: 1, 2, 3, 6, 12, 24 hour lags for PM2.5 and AQI
- **Rolling Statistics**: 6-hour and 24-hour rolling mean and std for all pollutants
- **Missing Value Handling**: Linear interpolation for short gaps, forward-fill for long gaps

### Model Training
- **Temporal Train/Val/Test Split**: 70%/15%/15% (no shuffling to prevent leakage)
- **Multiple Models**: Ridge Regression, LightGBM, XGBoost
- **Hyperparameter Tuning**: TimeSeriesSplit cross-validation with GridSearchCV
- **Feature Scaling**: StandardScaler fitted on training data only

### Evaluation & Interpretability
- Metrics: MAE, RMSE, R², MAPE, Bias
- Feature importance rankings
- Visualization: Forecast vs Actual plots, residual analysis, error distributions
- Comprehensive evaluation report

### Interactive Dashboard
- 24-72 hour AQI forecasts
- Interactive Plotly charts with AQI category backgrounds
- What-if scenario analysis (adjust wind, temperature, humidity)
- Model performance metrics display
- Feature importance visualization

## 📁 Project Structure

```
aqi_forecasting_project/
├── data_generator.py          # Generate synthetic AQI and weather data
├── feature_engineering.py     # Feature engineering pipeline
├── train_model.py             # Model training and hyperparameter tuning
├── evaluate.py                # Model evaluation and visualization
├── app.py                     # Streamlit interactive dashboard
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── models/                    # Saved models and scalers
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── training_results.pkl
├── outputs/                   # Evaluation outputs
│   ├── forecast_vs_actual.png
│   ├── feature_importance.png
│   └── evaluation_report.txt
└── historical_aqi.csv         # Generated dataset (after Step 1)
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment recommended

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd aqi_forecasting_project

# Or simply navigate to the project directory
cd aqi_forecasting_project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

Execute the following steps in order:

```bash
# Step 1: Generate synthetic data
python data_generator.py

# Step 2: Train models
python train_model.py

# Step 3: Evaluate model
python evaluate.py

# Step 4: Launch dashboard
streamlit run app.py
```

## 📝 Detailed Execution Steps

### Step 1: Generate Synthetic Data

Generate 2 years of realistic hourly AQI and weather data.

```bash
python data_generator.py
```

**What it does:**
- Creates `historical_aqi.csv` with 17,520 hours of data (2 years)
- Includes realistic patterns:
  - Seasonal temperature variations
  - Diurnal (daily) pollution peaks during rush hours
  - Winter smog spikes
  - Random pollutant spikes
  - Missing values (2-3%) simulating sensor failures
- Calculates AQI from PM2.5 using US EPA breakpoints

**Expected Output:**
```
Generating 2 years of synthetic hourly data...
✓ Data generated and saved to historical_aqi.csv
  Shape: (17520, 15)
  Date range: 2022-01-01 00:00:00 to 2023-12-31 23:00:00
  Missing values per column:
  pm25       432
  pm10       438
  ...
```

**Output File:** `historical_aqi.csv`

---

### Step 2: Train Models

Train multiple ML models with hyperparameter tuning and save the best one.

```bash
python train_model.py
```

**What it does:**
1. Loads and preprocesses data using feature engineering pipeline
2. Creates temporal train/validation/test split (70%/15%/15%)
3. Scales features using StandardScaler
4. Trains three models with TimeSeriesSplit cross-validation:
   - **Ridge Regression**: Linear baseline with alpha tuning
   - **LightGBM**: Gradient boosting with comprehensive hyperparameter grid
   - **XGBoost**: Gradient boosting with regularization
5. Evaluates all models on test set
6. Selects and saves the best-performing model

**Expected Output:**
```
============================================================
AQI FORECASTING - MODEL TRAINING PIPELINE
============================================================

Step 1: Loading and preparing data...
============================================================
FEATURE ENGINEERING PIPELINE
============================================================
Handling missing values...
  Missing values before: 3892
  Missing values after: 0
✓ Created temporal features...
✓ Created 12 lag features
✓ Created 24 rolling features
✓ Dropped 24 rows with NaN values
  Final dataset shape: (17496, 58)
============================================================

Step 2: Creating temporal train/validation/test split...
============================================================
TEMPORAL DATA SPLIT
============================================================
  Training set:   12247 samples (70.0%)
  Validation set: 2624 samples (15.0%)
  Test set:       2625 samples (15.0%)
  Total samples:  17496
============================================================

Step 3: Scaling features...
✓ Features scaled using StandardScaler

Step 4: Training models...

============================================================
TRAINING RIDGE REGRESSION
============================================================
  Best alpha: 10.0
  Validation RMSE: 8.45
  Validation MAE: 6.32
  Validation R²: 0.9234
============================================================

Ridge Regression Test Performance:
  MAE:  6.45
  RMSE: 8.52
  R²:   0.9215

============================================================
TRAINING LIGHTGBM
============================================================
  ...
```

**Output Files:**
- `models/best_model.pkl` - Best performing model
- `models/scaler.pkl` - Fitted StandardScaler
- `models/training_results.pkl` - Complete training results

**Training Time:** ~5-15 minutes depending on hardware

---

### Step 3: Evaluate Model

Generate comprehensive evaluation metrics and visualizations.

```bash
python evaluate.py
```

**What it does:**
1. Loads saved model, scaler, and test data
2. Generates predictions on test set
3. Calculates evaluation metrics:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² Score
   - MAPE (Mean Absolute Percentage Error)
   - Bias
4. Extracts feature importance
5. Generates visualizations:
   - Forecast vs Actual time series plot
   - Scatter plot with perfect prediction line
   - Residual analysis plot
   - Error distribution histogram
   - Feature importance bar chart
6. Creates comprehensive evaluation report

**Expected Output:**
```
============================================================
AQI FORECASTING - MODEL EVALUATION
============================================================

Loading model artifacts...
✓ Model loaded: LightGBM
✓ Test data shape: (2625, 57)

Generating predictions on test set...

Calculating evaluation metrics...

============================================================
TEST SET PERFORMANCE
============================================================
  MAE:  5.23 AQI units
  RMSE: 6.87 AQI units
  R²:   0.9456
  MAPE: 8.34%
  Bias: 0.12 AQI units
============================================================

Extracting feature importance...

Top 10 Most Important Features:
  rank              feature  importance
     1              pm25_lag_1h    0.2845
     2         pm25_rolling_mean_24h    0.1523
     3                  pm25    0.0987
     ...

Generating visualizations...
✓ Forecast vs Actual plot saved to outputs/forecast_vs_actual.png
✓ Feature importance plot saved to outputs/feature_importance.png
✓ Evaluation report saved to outputs/evaluation_report.txt

============================================================
EVALUATION COMPLETE
============================================================
```

**Output Files:**
- `outputs/forecast_vs_actual.png` - 4-panel visualization
- `outputs/feature_importance.png` - Top 20 features bar chart
- `outputs/evaluation_report.txt` - Comprehensive text report

---

### Step 4: Launch Interactive Dashboard

Start the Streamlit web application for interactive forecasting.

```bash
streamlit run app.py
```

**What it does:**
- Launches web server (default: http://localhost:8501)
- Opens browser automatically
- Loads trained model and data
- Provides interactive interface for:
  - Viewing historical actual vs predicted AQI
  - Generating 24-72 hour forecasts
  - What-if scenario analysis
  - Exploring model performance metrics

**Default URL:** http://localhost:8501

**To stop the dashboard:**
Press `Ctrl+C` in the terminal

## 🖥️ Dashboard Usage

### Tab 1: 📊 Forecast Visualization

**Features:**
- **Historical Plot**: Shows actual vs predicted AQI for test set
- **Forecast**: Displays predicted AQI for next 24-72 hours
- **AQI Categories**: Color-coded backgrounds (Good, Moderate, Unhealthy, etc.)
- **Metrics Cards**: Current AQI, average forecast, max/min values

**How to Use:**
1. Adjust **Forecast Horizon** slider (24-72 hours)
2. Modify **What-If Scenario** sliders:
   - Wind Speed (km/h)
   - Temperature (°C)
   - Humidity (%)
3. View updated forecast in real-time
4. Hover over chart for detailed values

### Tab 2: 📈 Model Performance

**Features:**
- Performance metrics (MAE, RMSE, R², MAPE)
- Top 10 feature importance bar chart
- Model information

### Tab 3: ℹ️ About

Project information, usage instructions, and disclaimer.

## 📊 Model Performance

Typical performance metrics on test set:

| Metric | Value | Description |
|--------|-------|-------------|
| MAE | 5-7 AQI | Average prediction error |
| RMSE | 7-9 AQI | Penalizes large errors |
| R² | 0.92-0.95 | Variance explained |
| MAPE | 7-10% | Percentage error |

**Top Important Features (Typical):**
1. PM2.5 lag 1h (most recent value)
2. PM2.5 rolling mean 24h
3. PM2.5 current value
4. PM2.5 lag 24h
5. NO2 rolling mean 24h
6. Hour (temporal)
7. Temperature
8. PM10 rolling mean 24h
9. O3 rolling mean 24h
10. Wind Speed

## 🔬 Technical Details

### Data Generation

**Temporal Patterns:**
- **Seasonal**: Sinusoidal temperature variation (amplitude: 15°C)
- **Diurnal**: Temperature peaks at 3 PM, pollution peaks at rush hours (8 AM, 6 PM)
- **Weekly**: Lower pollution on weekends
- **Random**: Gaussian noise and occasional spikes

**Missing Value Patterns:**
- Random missing: 2.5% probability per column
- Consecutive missing: 20 random gaps of 3-10 hours
- Columns affected: PM2.5, PM10, NO2, CO, O3, temperature, humidity, wind speed

### Feature Engineering

**Temporal Encoding:**
```python
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
```

**Lag Features:**
- t-1, t-2, t-3, t-6, t-12, t-24 hours
- Applied to: PM2.5, AQI

**Rolling Windows:**
- 6-hour and 24-hour windows
- Statistics: mean, standard deviation
- Applied to: PM2.5, PM10, NO2, CO, SO2, O3

### Model Architecture

**Ridge Regression:**
- Alpha values: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
- Baseline linear model

**LightGBM:**
- n_estimators: [100, 200, 300]
- learning_rate: [0.01, 0.05, 0.1]
- max_depth: [3, 5, 7]
- num_leaves: [15, 31, 63]
- subsample: [0.8, 0.9, 1.0]
- colsample_bytree: [0.8, 0.9, 1.0]

**XGBoost:**
- n_estimators: [100, 200, 300]
- learning_rate: [0.01, 0.05, 0.1]
- max_depth: [3, 5, 7]
- subsample: [0.8, 0.9, 1.0]
- colsample_bytree: [0.8, 0.9, 1.0]
- gamma: [0, 0.1, 0.2]

### Cross-Validation

- **Strategy**: TimeSeriesSplit with 5 folds
- **Scoring**: Negative mean squared error
- **No shuffling**: Preserves temporal order

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError`

**Solution:**
```bash
# Ensure you're in the correct directory
cd aqi_forecasting_project

# Activate virtual environment (if using)
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: historical_aqi.csv`

**Solution:**
```bash
# Run data generator first
python data_generator.py
```

### Issue: `FileNotFoundError: models/best_model.pkl`

**Solution:**
```bash
# Run training pipeline first
python train_model.py
```

### Issue: Streamlit dashboard won't start

**Solution:**
```bash
# Check if port 8501 is available
# Try specifying a different port
streamlit run app.py --server.port 8502

# Or check for conflicting processes
netstat -ano | findstr :8501  # Windows
lsof -i :8501  # Mac/Linux
```

### Issue: Training takes too long

**Solution:**
Reduce hyperparameter grid in `train_model.py`:

```python
# Reduce parameter grid
param_grid = {
    'n_estimators': [100, 200],  # Instead of [100, 200, 300]
    'learning_rate': [0.05, 0.1],  # Instead of [0.01, 0.05, 0.1]
    'max_depth': [3, 5],  # Instead of [3, 5, 7]
    # ... reduce other parameters similarly
}
```

### Issue: Low R² score (< 0.8)

**Possible causes:**
1. Insufficient training data
2. Poor feature engineering
3. Model not converging

**Solutions:**
- Increase training data (more years in data_generator.py)
- Add more features (weather interactions, polynomial features)
- Try different models or architectures
- Adjust hyperparameter ranges

### Issue: Memory error during training

**Solution:**
```bash
# Reduce data size in data_generator.py
# Change: years=2 to years=1

# Or reduce test set size in train_model.py
# Adjust train_ratio, val_ratio
```

## 📚 Additional Resources

### AQI Information
- [US EPA AQI Basics](https://www.epa.gov/air-quality-index-aqi/understanding-air-quality-index-aqi)
- [WHO Air Quality Guidelines](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health)

### Libraries Documentation
- [LightGBM](https://lightgbm.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Streamlit](https://docs.streamlit.io/)
- [Plotly](https://plotly.com/python/)

## ⚠️ Disclaimer

This is a **demonstration project** using **synthetic data**. For real-world applications:

1. Use actual air quality monitoring data from trusted sources
2. Consult environmental scientists and domain experts
3. Validate model performance on real data
4. Consider regulatory requirements for air quality forecasting
5. Implement proper error handling and monitoring in production

## 📄 License

This project is provided for educational and demonstration purposes.

## 👤 Author

Senior Data Scientist & MLOps Engineer

---

**Last Updated:** 2024
**Version:** 1.0.0
