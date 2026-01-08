"""
F1 Race Prediction Model - Japanese GP Analysis
Predicts race performance using qualifying times, sector times, weather data, and wet performance factors.
"""

import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Configuration Constants
CACHE_DIR = "f1_cache"
YEAR = 2024
GRAND_PRIX = "Japan"
SESSION_TYPE = "R"
RANDOM_STATE = 38
TEST_SIZE = 0.2
N_ESTIMATORS = 200
LEARNING_RATE = 0.1
API_KEY = "YOURAPIKEY"
LATITUDE = 34.8823
LONGITUDE = 136.5845
FORECAST_TIME = "2025-04-05 14:00:00"


def initialize_cache():
    """Enable FastF1 caching for faster data retrieval."""
    fastf1.Cache.enable_cache(CACHE_DIR)


def load_historical_race_data(year: int, gp_name: str, session: str) -> pd.DataFrame:
    """
    Load and process historical race session data with lap and sector times.
    
    Args:
        year: Race year
        gp_name: Grand Prix name
        session: Session type (e.g., "R" for race)
        
    Returns:
        DataFrame with processed lap and sector time data
    """
    race_session = fastf1.get_session(year, gp_name, session)
    race_session.load()
    
    lap_data = race_session.laps[
        ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    ].copy()
    lap_data.dropna(inplace=True)
    
    time_columns = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    for col in time_columns:
        lap_data[f"{col} (s)"] = lap_data[col].dt.total_seconds()
    
    return lap_data


def compute_average_sector_times(lap_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average sector times per driver.
    
    Args:
        lap_data: DataFrame with lap and sector time data
        
    Returns:
        DataFrame with average sector times grouped by driver
    """
    return lap_data.groupby("Driver")[
        ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
    ].mean().reset_index()


def get_wet_performance_factors() -> dict:
    """
    Get wet performance factors for drivers.
    
    Returns:
        Dictionary mapping driver codes to wet performance factors
    """
    return {
        "VER": 0.975196, "HAM": 0.976464, "LEC": 0.975862, "NOR": 0.978179,
        "ALO": 0.972655, "RUS": 0.968678, "SAI": 0.978754, "TSU": 0.996338,
        "OCO": 0.981810, "GAS": 0.978832, "STR": 0.979857
    }


def create_qualifying_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with 2025 Japanese GP qualifying times.
    
    Returns:
        DataFrame containing driver codes and qualifying times
    """
    drivers = [
        "VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO",
        "TSU", "SAI", "HUL", "OCO", "STR"
    ]
    
    qualifying_times = [
        86.983, 86.995, 87.027, 87.299, 87.318, 87.610, 87.822, 87.897,
        88.000, 87.836, 88.570, 88.696, 89.271
    ]
    
    return pd.DataFrame({
        "Driver": drivers,
        "QualifyingTime (s)": qualifying_times
    })


def fetch_weather_data(lat: float, lon: float, forecast_time: str, api_key: str) -> tuple:
    """
    Fetch weather forecast data from OpenWeatherMap API.
    
    Args:
        lat: Latitude
        lon: Longitude
        forecast_time: Forecast time string
        api_key: API key for OpenWeatherMap
        
    Returns:
        Tuple of (rain_probability, temperature)
    """
    weather_url = (
        f"http://api.openweathermap.org/data/2.5/forecast?"
        f"lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    
    try:
        response = requests.get(weather_url)
        weather_data = response.json()
        
        forecast_data = None
        for forecast in weather_data["list"]:
            if forecast["dt_txt"] == forecast_time:
                forecast_data = forecast
                break
        
        if forecast_data:
            return forecast_data["pop"], forecast_data["main"]["temp"]
        else:
            return 0, 20
    except Exception:
        return 0, 20


def prepare_training_data(
    qualifying_df: pd.DataFrame,
    sector_times: pd.DataFrame,
    historical_laps: pd.DataFrame,
    rain_probability: float,
    temperature: float
) -> tuple:
    """
    Merge all data sources and prepare features for training.
    
    Args:
        qualifying_df: DataFrame with qualifying times
        sector_times: DataFrame with sector times
        historical_laps: DataFrame with historical lap data
        rain_probability: Rain probability
        temperature: Temperature
        
    Returns:
        Tuple of (features, targets, merged_data)
    """
    wet_performance = get_wet_performance_factors()
    
    qualifying_df = qualifying_df.copy()
    qualifying_df["WetPerformanceFactor"] = qualifying_df["Driver"].map(wet_performance)
    
    merged_data = qualifying_df.merge(
        sector_times,
        left_on="Driver",
        right_on="Driver",
        how="left"
    )
    
    merged_data["RainProbability"] = rain_probability
    merged_data["Temperature"] = temperature
    
    feature_columns = [
        "QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
        "WetPerformanceFactor", "RainProbability", "Temperature"
    ]
    features = merged_data[feature_columns].fillna(0)
    
    targets = merged_data.merge(
        historical_laps.groupby("Driver")["LapTime (s)"].mean(),
        left_on="Driver",
        right_index=True
    )["LapTime (s)"]
    
    return features, targets, merged_data


def train_prediction_model(features: pd.DataFrame, targets: pd.Series) -> tuple:
    """
    Train a gradient boosting model for race time prediction.
    
    Args:
        features: Training features
        targets: Training targets
        
    Returns:
        Tuple of (trained_model, test_features, test_targets)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE
    )
    
    model.fit(X_train, y_train)
    
    return model, X_test, y_test


def generate_predictions(model: GradientBoostingRegressor, features: pd.DataFrame) -> np.ndarray:
    """
    Generate race time predictions using the trained model.
    
    Args:
        model: Trained prediction model
        features: Feature data for prediction
        
    Returns:
        Array with predicted race times
    """
    return model.predict(features)


def evaluate_model(model: GradientBoostingRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluate model performance using mean absolute error.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Mean absolute error value
    """
    predictions = model.predict(X_test)
    return mean_absolute_error(y_test, predictions)


def display_results(predictions_df: pd.DataFrame, mae: float):
    """
    Display prediction results and model evaluation.
    
    Args:
        predictions_df: DataFrame with predictions
        mae: Mean absolute error value
    """
    print("\n🏁 Predicted 2025 Japanese GP Winner🏁\n")
    print(predictions_df[["Driver", "PredictedRaceTime (s)"]])
    print(f"\n🔍 Model Error (MAE): {mae:.2f} seconds")


def main():
    """Main execution function."""
    initialize_cache()
    
    historical_data = load_historical_race_data(YEAR, GRAND_PRIX, SESSION_TYPE)
    sector_times = compute_average_sector_times(historical_data)
    qualifying_data = create_qualifying_dataframe()
    
    rain_prob, temperature = fetch_weather_data(
        LATITUDE, LONGITUDE, FORECAST_TIME, API_KEY
    )
    
    X, y, merged_data = prepare_training_data(
        qualifying_data, sector_times, historical_data, rain_prob, temperature
    )
    
    model, X_test, y_test = train_prediction_model(X, y)
    
    predictions = generate_predictions(model, X)
    qualifying_data["PredictedRaceTime (s)"] = predictions
    qualifying_data = qualifying_data.sort_values(by="PredictedRaceTime (s)")
    
    mae = evaluate_model(model, X_test, y_test)
    display_results(qualifying_data, mae)


if __name__ == "__main__":
    main()
