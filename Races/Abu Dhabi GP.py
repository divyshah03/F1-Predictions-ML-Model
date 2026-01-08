"""
F1 Race Prediction Model - Abu Dhabi GP Analysis
Predicts race performance using qualifying times, sector times, weather data, and team performance.
"""

import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


# Configuration Constants
CACHE_DIR = "f1_cache"
YEAR = 2024
ROUND = 24
SESSION_TYPE = "R"
RANDOM_STATE = 39
TEST_SIZE = 0.1
N_ESTIMATORS = 300
LEARNING_RATE = 0.9
MAX_DEPTH = 3
MONOTONE_CONSTRAINTS = '(1, 0, 0, -1, -1)'
API_KEY = ""
LATITUDE = 24.4672
LONGITUDE = 54.6031
FORECAST_TIME = "2025-12-07 13:00:00"
RAIN_THRESHOLD = 0.75


def initialize_cache():
    """Enable FastF1 caching for faster data retrieval."""
    fastf1.Cache.enable_cache(CACHE_DIR)


def load_historical_race_data(year: int, round_num: int, session: str) -> pd.DataFrame:
    """
    Load and process historical race session data with lap and sector times.
    
    Args:
        year: Race year
        round_num: Race round number
        session: Session type (e.g., "R" for race)
        
    Returns:
        DataFrame with processed lap and sector time data
    """
    race_session = fastf1.get_session(year, round_num, session)
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
    Calculate average sector times per driver and total sector time.
    
    Args:
        lap_data: DataFrame with lap and sector time data
        
    Returns:
        DataFrame with average sector times grouped by driver
    """
    sector_times = lap_data.groupby("Driver").agg({
        "Sector1Time (s)": "mean",
        "Sector2Time (s)": "mean",
        "Sector3Time (s)": "mean"
    }).reset_index()
    
    sector_times["TotalSectorTime (s)"] = (
        sector_times["Sector1Time (s)"] +
        sector_times["Sector2Time (s)"] +
        sector_times["Sector3Time (s)"]
    )
    
    return sector_times


def get_clean_air_race_pace() -> dict:
    """
    Get clean air race pace data for drivers.
    
    Returns:
        Dictionary mapping driver codes to race pace times
    """
    return {
        "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600,
        "ALO": 94.784333, "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444,
        "STR": 95.318250, "HUL": 95.345455, "OCO": 95.682128
    }


def create_qualifying_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with 2025 Abu Dhabi GP qualifying times.
    
    Returns:
        DataFrame containing driver codes and qualifying times
    """
    drivers = [
        "RUS", "VER", "PIA", "NOR", "HAM", "LEC", "ALO", "HUL",
        "ALB", "SAI", "STR", "OCO", "GAS"
    ]
    
    qualifying_times = [
        82.645, 82.207, 82.437, 82.408, 83.394, 82.730, 82.902, 83.450,
        83.416, 83.042, 83.097, 82.913, 83.468
    ]
    
    return pd.DataFrame({
        "Driver": drivers,
        "QualifyingTime (s)": qualifying_times
    })


def get_team_data() -> tuple:
    """
    Get team points and performance scores.
    
    Returns:
        Tuple of (team_points dict, driver_to_team dict)
    """
    team_points = {
        "McLaren": 800, "Mercedes": 459, "Red Bull": 426, "Williams": 137,
        "Ferrari": 382, "Haas": 73, "Aston Martin": 80, "Kick Sauber": 68,
        "Racing Bulls": 92, "Alpine": 22
    }
    
    driver_to_team = {
        "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari",
        "RUS": "Mercedes", "HAM": "Ferrari", "GAS": "Alpine", "ALO": "Aston Martin",
        "TSU": "Racing Bulls", "SAI": "Williams", "HUL": "Kick Sauber",
        "OCO": "Alpine", "STR": "Aston Martin"
    }
    
    return team_points, driver_to_team


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
        forecast_data = next(
            (f for f in weather_data["list"] if f["dt_txt"] == forecast_time),
            None
        )
        
        if forecast_data:
            return forecast_data["pop"], forecast_data["main"]["temp"]
        else:
            return 0, 20
    except Exception:
        return 0, 20


def adjust_qualifying_time_for_weather(
    qualifying_df: pd.DataFrame,
    rain_probability: float,
    threshold: float
) -> pd.DataFrame:
    """
    Adjust qualifying times based on weather conditions.
    
    Args:
        qualifying_df: DataFrame with qualifying data
        rain_probability: Probability of rain
        threshold: Rain probability threshold for adjustment
        
    Returns:
        DataFrame with adjusted qualifying times
    """
    qualifying_df = qualifying_df.copy()
    
    if rain_probability >= threshold:
        qualifying_df["QualifyingTime"] = (
            qualifying_df["QualifyingTime (s)"] * qualifying_df.get("WetPerformanceFactor", 1.0)
        )
    else:
        qualifying_df["QualifyingTime"] = qualifying_df["QualifyingTime (s)"]
    
    return qualifying_df


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
    clean_air_pace = get_clean_air_race_pace()
    team_points, driver_to_team = get_team_data()
    
    qualifying_df = qualifying_df.copy()
    qualifying_df["CleanAirRacePace (s)"] = qualifying_df["Driver"].map(clean_air_pace)
    qualifying_df["Team"] = qualifying_df["Driver"].map(driver_to_team)
    
    max_points = max(team_points.values())
    team_performance_score = {team: points / max_points for team, points in team_points.items()}
    qualifying_df["TeamPerformanceScore"] = qualifying_df["Team"].map(team_performance_score)
    
    qualifying_df = adjust_qualifying_time_for_weather(qualifying_df, rain_probability, RAIN_THRESHOLD)
    
    merged_data = qualifying_df.merge(
        sector_times[["Driver", "TotalSectorTime (s)"]],
        on="Driver",
        how="left"
    )
    
    merged_data["RainProbability"] = rain_probability
    merged_data["Temperature"] = temperature
    merged_data["QualifyingTime"] = merged_data["QualifyingTime"]
    
    valid_drivers = merged_data["Driver"].isin(historical_laps["Driver"].unique())
    merged_data = merged_data[valid_drivers]
    
    feature_columns = [
        "QualifyingTime", "RainProbability", "Temperature",
        "TeamPerformanceScore", "CleanAirRacePace (s)"
    ]
    features = merged_data[feature_columns]
    targets = historical_laps.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])
    
    return features, targets, merged_data


def train_prediction_model(features: pd.DataFrame, targets: pd.Series) -> tuple:
    """
    Train an XGBoost model for race time prediction.
    
    Args:
        features: Training features
        targets: Training targets
        
    Returns:
        Tuple of (trained_model, test_features, test_targets, imputer)
    """
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    model = XGBRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        monotone_constraints=MONOTONE_CONSTRAINTS
    )
    
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, imputer


def generate_predictions(
    model: XGBRegressor,
    features: pd.DataFrame,
    imputer: SimpleImputer
) -> np.ndarray:
    """
    Generate race time predictions using the trained model.
    
    Args:
        model: Trained prediction model
        features: Feature data for prediction
        imputer: Fitted imputer
        
    Returns:
        Array with predicted race times
    """
    X_imputed = imputer.transform(features)
    return model.predict(X_imputed)


def evaluate_model(model: XGBRegressor, X_test: np.ndarray, y_test: pd.Series) -> float:
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


def plot_feature_importance(model: XGBRegressor, feature_names: pd.Index):
    """
    Plot feature importance from the trained model.
    
    Args:
        model: Trained model
        feature_names: Names of features
    """
    feature_importance = model.feature_importances_
    
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, feature_importance, color='skyblue')
    plt.xlabel("Importance")
    plt.title("Feature Importance in Race Time Prediction")
    plt.tight_layout()
    plt.show()


def display_results(predictions_df: pd.DataFrame, mae: float):
    """
    Display prediction results and model evaluation.
    
    Args:
        predictions_df: DataFrame with predictions
        mae: Mean absolute error value
    """
    print(predictions_df[["Driver", "PredictedRaceTime (s)"]])
    
    podium = predictions_df.loc[:7, ["Driver", "PredictedRaceTime (s)"]]
    print("\n🏆 Predicted in the Top 3 🏆")
    print(f"🥇 P1: {podium.iloc[0]['Driver']}")
    print(f"🥈 P2: {podium.iloc[1]['Driver']}")
    print(f"🥉 P3: {podium.iloc[2]['Driver']}")
    print(f"\nModel Error (MAE): {mae:.2f} seconds")


def main():
    """Main execution function."""
    initialize_cache()
    
    historical_data = load_historical_race_data(YEAR, ROUND, SESSION_TYPE)
    sector_times = compute_average_sector_times(historical_data)
    qualifying_data = create_qualifying_dataframe()
    
    rain_prob, temperature = fetch_weather_data(
        LATITUDE, LONGITUDE, FORECAST_TIME, API_KEY
    )
    
    X, y, merged_data = prepare_training_data(
        qualifying_data, sector_times, historical_data, rain_prob, temperature
    )
    
    model, X_test, y_test, imputer = train_prediction_model(X, y)
    
    predictions = generate_predictions(model, X, imputer)
    merged_data["PredictedRaceTime (s)"] = predictions
    
    final_results = merged_data.sort_values(
        by=["PredictedRaceTime (s)", "QualifyingTime"]
    ).reset_index(drop=True)
    
    mae = evaluate_model(model, X_test, y_test)
    display_results(final_results, mae)
    
    plot_feature_importance(model, X.columns)


if __name__ == "__main__":
    main()
