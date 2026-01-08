"""
F1 Race Prediction Model - Jeddah GP Analysis
Predicts race performance using qualifying times, sector times, weather data, and average 2025 performance.
"""

import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Configuration Constants
CACHE_DIR = "f1_cache"
YEAR = 2024
GRAND_PRIX = "Saudi Arabia"
SESSION_TYPE = "R"
RANDOM_STATE = 39
TEST_SIZE = 0.2
N_ESTIMATORS = 300
LEARNING_RATE = 0.05
MAX_DEPTH = 5
API_KEY = "yourkey"
LATITUDE = 21.4225
LONGITUDE = 39.1818
FORECAST_TIME = "2025-04-20 18:00:00"
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


def get_average_2025_performance() -> dict:
    """
    Get average 2025 season performance for drivers.
    
    Returns:
        Dictionary mapping driver codes to average lap times
    """
    return {
        "VER": 88.0, "PIA": 89.1, "LEC": 89.2, "RUS": 89.3, "HAM": 89.4,
        "GAS": 89.5, "ALO": 89.6, "TSU": 89.7, "SAI": 89.8, "HUL": 89.9,
        "OCO": 90.0, "STR": 90.1, "NOR": 90.2
    }


def create_qualifying_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with 2025 Jeddah GP qualifying times.
    
    Returns:
        DataFrame containing driver codes and qualifying times
    """
    drivers = [
        "VER", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU",
        "SAI", "HUL", "OCO", "STR", "NOR"
    ]
    
    qualifying_times = [
        87.294, 87.304, 87.670, 87.407, 88.201, 88.367, 88.303, 88.204,
        88.164, 88.782, 89.092, 88.645, 87.489
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
        "McLaren": 78, "Mercedes": 53, "Red Bull": 36, "Williams": 17,
        "Ferrari": 17, "Haas": 14, "Aston Martin": 10, "Kick Sauber": 6,
        "Racing Bulls": 3, "Alpine": 0
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
            qualifying_df["QualifyingTime (s)"] * qualifying_df["WetPerformanceFactor"]
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
        Tuple of (features, targets, clean_data)
    """
    wet_performance = get_wet_performance_factors()
    average_2025 = get_average_2025_performance()
    team_points, driver_to_team = get_team_data()
    
    qualifying_df = qualifying_df.copy()
    qualifying_df["WetPerformanceFactor"] = qualifying_df["Driver"].map(wet_performance)
    qualifying_df["Average2025Performance"] = qualifying_df["Driver"].map(average_2025)
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
    
    targets = historical_laps.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])
    
    clean_data = merged_data.copy()
    clean_data["LapTime (s)"] = targets.values
    clean_data = clean_data.dropna(subset=["LapTime (s)"])
    
    feature_columns = [
        "QualifyingTime", "TeamPerformanceScore", "RainProbability",
        "Temperature", "TotalSectorTime (s)", "Average2025Performance"
    ]
    features = clean_data[feature_columns].fillna(0)
    targets_clean = clean_data["LapTime (s)"]
    
    return features, targets_clean, clean_data


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
        max_depth=MAX_DEPTH,
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


def compute_residuals(clean_data: pd.DataFrame) -> pd.Series:
    """
    Compute residuals for each driver.
    
    Args:
        clean_data: DataFrame with actual and predicted times
        
    Returns:
        Series with mean residuals per driver
    """
    clean_data["Residual"] = clean_data["LapTime (s)"] - clean_data["PredictedRaceTime (s)"]
    return clean_data.groupby("Driver")["Residual"].mean().sort_values()


def plot_team_performance_effect(results_df: pd.DataFrame):
    """
    Plot the effect of team performance on predicted race results.
    
    Args:
        results_df: DataFrame with predictions and team performance data
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        results_df["TeamPerformanceScore"],
        results_df["PredictedRaceTime (s)"],
        c=results_df["QualifyingTime"],
        cmap='viridis'
    )
    
    for i, driver in enumerate(results_df["Driver"]):
        plt.annotate(
            driver,
            (results_df["TeamPerformanceScore"].iloc[i], results_df["PredictedRaceTime (s)"].iloc[i]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.colorbar(scatter, label="Qualifying Time")
    plt.xlabel("Team Performance Score")
    plt.ylabel("Predicted Race Time (s)")
    plt.title("Effect of Team Performance on Predicted Race Results")
    plt.tight_layout()
    plt.savefig('team_performance_effect.png')
    plt.show()


def plot_feature_importance(model: GradientBoostingRegressor, feature_names: pd.Index):
    """
    Plot feature importance from the trained model.
    
    Args:
        model: Trained model
        feature_names: Names of features
    """
    feature_importance = model.feature_importances_
    
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, feature_importance, color='skyblue')
    plt.xlabel("Importance")
    plt.title("Feature Importance in Race Time Prediction")
    plt.tight_layout()
    plt.show()


def compute_correlation_matrix(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix for features and target.
    
    Args:
        merged_data: DataFrame with all features and target
        
    Returns:
        Correlation matrix
    """
    correlation_columns = [
        "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore",
        "LastYearWinner", "Average2025Performance", "TotalSectorTime (s)", "LapTime (s)"
    ]
    
    available_columns = [col for col in correlation_columns if col in merged_data.columns]
    return merged_data[available_columns].corr()


def display_results(predictions_df: pd.DataFrame, mae: float, driver_errors: pd.Series):
    """
    Display prediction results and model evaluation.
    
    Args:
        predictions_df: DataFrame with predictions
        mae: Mean absolute error value
        driver_errors: Series with driver residuals
    """
    print("Predicted 2025 Saudi Arabian GP Winner:")
    print(predictions_df[["Driver", "PredictedRaceTime (s)"]])
    print(f"\nModel Error (MAE): {mae:.2f} seconds")
    print(f"\nDriver Residuals:\n{driver_errors}")


def main():
    """Main execution function."""
    initialize_cache()
    
    historical_data = load_historical_race_data(YEAR, GRAND_PRIX, SESSION_TYPE)
    sector_times = compute_average_sector_times(historical_data)
    qualifying_data = create_qualifying_dataframe()
    
    rain_prob, temperature = fetch_weather_data(
        LATITUDE, LONGITUDE, FORECAST_TIME, API_KEY
    )
    
    X, y, clean_data = prepare_training_data(
        qualifying_data, sector_times, historical_data, rain_prob, temperature, LAST_YEAR_WINNER
    )
    
    model, X_test, y_test = train_prediction_model(X, y)
    
    predictions = generate_predictions(model, X)
    clean_data["PredictedRaceTime (s)"] = predictions
    
    final_results = clean_data.sort_values("PredictedRaceTime (s)")
    
    mae = evaluate_model(model, X_test, y_test)
    driver_errors = compute_residuals(clean_data)
    
    display_results(final_results, mae, driver_errors)
    
    plot_team_performance_effect(final_results)
    plot_feature_importance(model, X.columns)
    
    merged_data = qualifying_data.merge(
        sector_times[["Driver", "TotalSectorTime (s)"]],
        on="Driver",
        how="left"
    )
    merged_data["LapTime (s)"] = historical_data.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])
    corr_matrix = compute_correlation_matrix(merged_data)
    print(f"\nCorrelation Matrix:\n{corr_matrix}")


if __name__ == "__main__":
    main()
