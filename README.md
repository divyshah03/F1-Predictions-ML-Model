# 🏎️ Formula 1 Race Prediction System 2025

A comprehensive machine learning framework for predicting Formula 1 race outcomes using advanced feature engineering, ensemble learning algorithms, and real-time weather data integration.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture & Design](#-architecture--design)
- [Features](#-features)
- [Data Sources](#-data-sources)
- [Machine Learning Models](#-machine-learning-models)
- [Feature Engineering](#-feature-engineering)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Project Overview

📌 In this project, I built a full ML pipeline that predicts Formula 1 race outcomes using:
• FastF1 telemetry data (lap times, sector times)
• Real-time weather API integration
• Advanced feature engineering
• Gradient Boosting & XGBoost models
• Per-race prediction scripts for the entire season

### Key Capabilities

- **Multi-Grand Prix Support**: Individual prediction models for each race on the calendar
- **Real-Time Weather Integration**: OpenWeatherMap API integration for meteorological forecasting
- **Advanced Feature Engineering**: Sector times, team performance metrics, driver-specific factors
- **Ensemble Learning**: Gradient Boosting and XGBoost regression models
- **Comprehensive Evaluation**: Mean Absolute Error (MAE) metrics and feature importance analysis
- **Modular Architecture**: Professional, maintainable codebase with clear separation of concerns

## 🏗️ Architecture & Design

The project follows a **modular, object-oriented design pattern** with the following architectural principles:

### Design Patterns

- **Separation of Concerns**: Each module handles a specific responsibility (data loading, preprocessing, modeling, visualization)
- **Configuration Management**: Centralized constants for easy parameter tuning
- **Function-Based Architecture**: Reusable functions with comprehensive documentation
- **Type Hints**: Full type annotation support for better code maintainability

### Code Structure

Each Grand Prix prediction script follows a consistent structure:

1. **Configuration Constants**: All hyperparameters and settings defined at module level
2. **Data Loading Functions**: Historical race data retrieval and preprocessing
3. **Feature Engineering Functions**: Sector time aggregation, team performance scoring
4. **Model Training Functions**: Machine learning pipeline implementation
5. **Evaluation Functions**: Model performance metrics and visualization
6. **Main Execution**: Orchestrates the complete prediction workflow

## ✨ Features

### Core Functionality

- **Historical Data Analysis**: Processes 2024 race sessions to extract lap times, sector times, and driver performance metrics
- **Qualifying Time Integration**: Incorporates 2025 qualifying session results as primary predictive features
- **Sector Time Analysis**: Aggregates and analyzes sector-specific performance data
- **Team Performance Scoring**: Normalized team championship points as performance indicators

### Advanced Features

- **Weather Condition Integration**
  - Real-time weather forecast retrieval via OpenWeatherMap API
  - Rain probability assessment for wet weather performance adjustments
  - Temperature-based performance modifiers
  - Location-specific meteorological data (latitude/longitude coordinates)

- **Driver-Specific Metrics**
  - Clean air race pace calculations
  - Wet weather performance factors
  - Average position change statistics (track-specific)
  - Season-long performance trends

- **Feature Engineering Techniques**
  - Time series normalization (lap times to seconds)
  - Missing value imputation using median strategy
  - Feature scaling and transformation
  - Polynomial feature engineering (qualifying time squared)

- **Model Variants**
  - **Gradient Boosting Regressor**: Primary ensemble method with configurable hyperparameters
  - **XGBoost Regressor**: Advanced gradient boosting with monotonic constraints
  - Custom hyperparameter tuning per Grand Prix

### Visualization & Analysis

- **Feature Importance Plots**: Horizontal bar charts showing model feature contributions
- **Race Pace Analysis**: Scatter plots correlating clean air pace with predicted race times
- **Team Performance Visualization**: Multi-dimensional scatter plots with color-coded qualifying times
- **Residual Analysis**: Driver-specific prediction error calculations
- **Correlation Matrices**: Feature interrelationship analysis

## 📊 Data Sources

### Primary Data Sources

1. **FastF1 API**
   - Historical race session data (2024 season)
   - Lap time telemetry
   - Sector time breakdowns
   - Driver performance metrics
   - Session caching for improved performance

2. **OpenWeatherMap API**
   - Real-time weather forecasts
   - Precipitation probability data
   - Temperature measurements
   - Location-specific meteorological conditions

3. **Manual Data Entry**
   - 2025 qualifying session results
   - Driver code mappings
   - Team performance data
   - Track-specific historical statistics

### Data Processing Pipeline

1. **Data Acquisition**: FastF1 session loading with automatic caching
2. **Data Cleaning**: Missing value handling, data type conversion
3. **Feature Extraction**: Sector time aggregation, team performance normalization
4. **Feature Engineering**: Weather adjustments, performance factor application
5. **Data Validation**: Driver matching, data integrity checks

## 🤖 Machine Learning Models

### Model Selection

The system employs **ensemble learning algorithms** specifically designed for regression tasks:

#### Gradient Boosting Regressor
- **Algorithm**: Ensemble of decision trees with gradient descent optimization
- **Hyperparameters**:
  - `n_estimators`: Number of boosting stages (100-300)
  - `learning_rate`: Shrinkage factor (0.05-0.9)
  - `max_depth`: Maximum tree depth (3-5)
  - `random_state`: Reproducibility seed

#### XGBoost Regressor
- **Algorithm**: Optimized gradient boosting framework
- **Advanced Features**:
  - Monotonic constraints for feature relationships
  - Regularization techniques
  - Parallel processing capabilities

### Training Methodology

- **Train-Test Split**: Configurable test size (10-30% of data)
- **Cross-Validation**: Random state seeding for reproducibility
- **Feature Imputation**: Median imputation for missing values
- **Target Variable**: Average lap time per driver (seconds)

### Model Evaluation

- **Primary Metric**: Mean Absolute Error (MAE) in seconds
- **Feature Importance**: Tree-based importance scores
- **Residual Analysis**: Driver-specific prediction errors
- **Correlation Analysis**: Feature-target relationships

## 🔧 Feature Engineering

### Temporal Features

- **Qualifying Time**: Primary predictive feature (adjusted for weather conditions)
- **Lap Time**: Historical average lap times per driver
- **Sector Times**: Individual sector performance (Sector 1, 2, 3)
- **Total Sector Time**: Aggregated sector performance metric

### Performance Features

- **Clean Air Race Pace**: Driver-specific race pace in optimal conditions
- **Team Performance Score**: Normalized championship points (0-1 scale)
- **Season Points**: Cumulative driver championship points
- **Average Position Change**: Track-specific position change statistics

### Environmental Features

- **Rain Probability**: Precipitation likelihood (0-1 scale)
- **Temperature**: Ambient temperature in Celsius
- **Weather-Adjusted Qualifying Time**: Qualifying time modified by wet performance factors

### Driver-Specific Features

- **Wet Performance Factor**: Driver-specific wet weather performance multiplier
- **Last Year Winner**: Binary indicator for previous year's race winner
- **Average 2025 Performance**: Season-long average lap time trends

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Dependencies

```bash
pip install fastf1 pandas numpy scikit-learn matplotlib xgboost requests
```

### Package Versions

- `fastf1`: FastF1 API client for Formula 1 data
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `matplotlib`: Data visualization
- `xgboost`: Advanced gradient boosting
- `requests`: HTTP library for API calls

### API Keys

To enable weather data integration, configure your OpenWeatherMap API key:

1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Obtain your API key
3. Update the `API_KEY` constant in each Grand Prix script

## 🎯 Usage

### Basic Execution

Run any Grand Prix prediction script:

```bash
python "Australian GP.py"
python "Monaco GP.py"
python "Abu Dhabi GP.py"
```

### Expected Output

```
🏁 Predicted 2025 Australian GP Winner 🏁

   Driver  PredictedRaceTime (s)
0    NOR                 82.345
1    VER                 82.456
2    PIA                 82.567
...

🔍 Model Error (MAE): 2.34 seconds

🏆 Predicted in the Top 3 🏆
🥇 P1: NOR
🥈 P2: VER
🥉 P3: PIA
```

### Customization

Modify configuration constants at the top of each script:

```python
# Configuration Constants
RANDOM_STATE = 39        # Reproducibility seed
TEST_SIZE = 0.2         # Test set proportion
N_ESTIMATORS = 100      # Number of trees
LEARNING_RATE = 0.1     # Learning rate
```

## 📁 Project Structure

```
2025_f1_predictions/
│
├── Races/
│   ├── Australian GP.py      # Australian Grand Prix predictions
│   ├── Bahrain GP.py          # Bahrain Grand Prix predictions
│   ├── Chinese GP.py          # Chinese Grand Prix predictions
│   ├── Emilia GP.py           # Emilia Romagna Grand Prix predictions
│   ├── Japanese GP.py         # Japanese Grand Prix predictions
│   ├── Jeddah GP.py           # Saudi Arabian Grand Prix predictions
│   ├── Miami GP.py            # Miami Grand Prix predictions
│   ├── Monaco GP.py           # Monaco Grand Prix predictions
│   └── Abu Dhabi GP.py        # Abu Dhabi Grand Prix predictions
│
└── README.md              # Project documentation
```

### File Naming Convention

Each file corresponds to a specific Grand Prix on the 2025 Formula 1 calendar, named after the race location.

## 📈 Model Performance

### Evaluation Metrics

- **Mean Absolute Error (MAE)**: Primary performance metric measured in seconds
- **Feature Importance**: Relative contribution of each feature to predictions
- **Residual Analysis**: Driver-specific prediction accuracy

### Performance Optimization

Models are tuned per Grand Prix with:
- Track-specific hyperparameters
- Feature selection based on track characteristics
- Historical data relevance weighting

### Typical Performance

- **MAE Range**: 1.5 - 4.0 seconds (varies by track complexity)
- **Feature Importance**: Qualifying time typically contributes 30-50% to predictions
- **Weather Impact**: Rain probability significantly affects predictions when >75%

## 🔬 Technical Details

### Data Preprocessing

1. **Time Conversion**: Timedelta objects converted to total seconds
2. **Missing Value Handling**: Median imputation for numerical features
3. **Driver Code Mapping**: Full names mapped to 3-letter F1 codes
4. **Data Merging**: Left joins preserving qualifying data integrity

### Model Training Pipeline

1. **Feature Selection**: Relevant features extracted from merged datasets
2. **Data Splitting**: Stratified train-test split with random state
3. **Imputation**: Missing values filled using median strategy
4. **Model Fitting**: Gradient boosting/XGBoost training
5. **Prediction Generation**: Race time predictions for all drivers
6. **Evaluation**: MAE calculation and feature importance extraction

### Visualization Pipeline

1. **Feature Importance**: Horizontal bar charts
2. **Race Pace Analysis**: Scatter plots with driver annotations
3. **Team Performance**: Multi-dimensional visualizations
4. **Correlation Analysis**: Heatmap matrices

## 🚧 Future Enhancements

### Planned Features

- **Pit Stop Strategy Integration**: Incorporate pit stop timing and tire strategy
- **Deep Learning Models**: Neural network architectures for complex pattern recognition
- **Real-Time Data Streaming**: Live telemetry integration during race weekends
- **Multi-Race Ensemble**: Aggregate predictions across multiple models
- **Driver Form Analysis**: Recent performance trends and momentum factors
- **Tire Degradation Modeling**: Compound-specific performance degradation
- **Overtaking Probability**: Position change likelihood modeling

### Model Improvements

- Hyperparameter optimization via grid search
- Cross-validation for robust performance estimation
- Feature selection algorithms
- Model stacking and blending techniques

---

🏎️ *Predicting the future of Formula 1, one race at a time* 🚀