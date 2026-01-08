"""
F1 Race Prediction Model - Miami GP Analysis
Predicts race performance using qualifying times, sector times, weather data, and last year's winner.
"""

import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


# Configuration Constants
CACHE_DIR = "f1_cache"
YEAR = 2024
GRAND_PRIX = "Miami"
SESSION_TYPE = "R"
RANDOM_STATE = 38
TEST_SIZE = 0.2
N_ESTIMATORS = 300
LEARNING_RATE = 0.05
MAX_DEPTH = 5
API_KEY = "YOURAPIKEY"
LATITUDE = 25.7617
LONGITUDE = -80.1918
FORECAST_TIME = "2025-05-04 13:00:00"
RAIN_THRESHOLD = 0.75
LAST_YEAR_WINNER = "VER"


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
    Create DataFrame with 2025 Miami GP qualifying times.
    
    Returns:
        DataFrame containing driver codes and qualifying times
    """
    drivers = [
        "VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
        "TSU", "HAM", "STR", "GAS", "ALO", "HUL"
    ]
    
    qualifying_times = [
        86.204, 86.269, 86.375, 86.385, 86.569, 86.682, 86.754, 86.824,
        86.943, 87.006, 87.830, 87.710, 87.604, 87.473
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
        "McLaren": 203, "Mercedes": 118, "Red Bull": 92, "Williams": 25,
        "Ferrari": 84, "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6,
        "Racing Bulls": 8, "Alpine": 7
    }
    
    driver_to_team = {
        "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari",
        "RUS": "Mercedes", "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin",
        "TSU": "Racing Bulls", "SAI": "Ferrari", "HUL": "Kick Sauber",
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
    temperature: float,
    last_year_winner: str
) -> tuple:
    """
    Merge all data sources and prepare features for training.
    
    Args:
        qualifying_df: DataFrame with qualifying times
        sector_times: DataFrame with sector times
        historical_laps: DataFrame with historical lap data
        rain_probability: Rain probability
        temperature: Temperature
        last_year_winner: Driver code of last year's winner
        
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
    merged_data["LastYearWinner"] = (merged_data["Driver"] == last_year_winner).astype(int)
    merged_data["QualifyingTime"] = merged_data["QualifyingTime"] ** 2
    
    feature_columns = [
        "QualifyingTime", "RainProbability", "Temperature",
        "TeamPerformanceScore", "CleanAirRacePace (s)"
    ]
    features = merged_data[feature_columns]
    targets = historical_laps.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])
    
    return features, targets, merged_data


def train_prediction_model(features: pd.DataFrame, targets: pd.Series) -> tuple:
    """
    Train a gradient boosting model for race time prediction.
    
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
    
    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, imputer


def generate_predictions(
    model: GradientBoostingRegressor,
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


def evaluate_model(model: GradientBoostingRegressor, X_test: np.ndarray, y_test: pd.Series) -> float:
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


def plot_race_pace_effect(results_df: pd.DataFrame):
    """
    Plot the effect of clean air race pace on predicted race results.
    
    Args:
        results_df: DataFrame with predictions and race pace data
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(
        results_df["CleanAirRacePace (s)"],
        results_df["PredictedRaceTime (s)"]
    )
    
    for i, driver in enumerate(results_df["Driver"]):
        plt.annotate(
            driver,
            (results_df["CleanAirRacePace (s)"].iloc[i], results_df["PredictedRaceTime (s)"].iloc[i]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.xlabel("clean air race pace (s)")
    plt.ylabel("predicted race time (s)")
    plt.title("effect of clean air race pace on predicted race results")
    plt.tight_layout()
    plt.show()


def display_results(predictions_df: pd.DataFrame, mae: float):
    """
    Display prediction results and model evaluation.
    
    Args:
        predictions_df: DataFrame with predictions
        mae: Mean absolute error value
    """
    print("\n🏁 Predicted 2025 Miami GP Winner 🏁\n")
    print(predictions_df[["Driver", "PredictedRaceTime (s)"]])
    print(f"\nModel Error (MAE ): {mae:.2f} seconds")


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
        qualifying_data, sector_times, historical_data, rain_prob, temperature, LAST_YEAR_WINNER
    )
    
    model, X_test, y_test, imputer = train_prediction_model(X, y)
    
    predictions = generate_predictions(model, X, imputer)
    merged_data["PredictedRaceTime (s)"] = predictions
    
    final_results = merged_data.sort_values("PredictedRaceTime (s)")
    
    mae = evaluate_model(model, X_test, y_test)
    display_results(final_results, mae)
    
    plot_race_pace_effect(final_results)


if __name__ == "__main__":
    main()
