"""
F1 Race Prediction Model - Australian GP Analysis
Predicts race performance based on qualifying times using historical race data.
"""

import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Configuration Constants
CACHE_DIR = "f1_cache"
YEAR = 2024
ROUND = 3
SESSION_TYPE = "R"
RANDOM_STATE = 39
TEST_SIZE = 0.2
N_ESTIMATORS = 100
LEARNING_RATE = 0.1


def initialize_cache():
    """Enable FastF1 caching for faster data retrieval."""
    fastf1.Cache.enable_cache(CACHE_DIR)


def load_historical_race_data(year: int, round_num: int, session: str) -> pd.DataFrame:
    """
    Load and process historical race session data.
    
    Args:
        year: Race year
        round_num: Race round number
        session: Session type (e.g., "R" for race)
        
    Returns:
        DataFrame with processed lap time data
    """
    race_session = fastf1.get_session(year, round_num, session)
    race_session.load()
    
    lap_data = race_session.laps[["Driver", "LapTime"]].copy()
    lap_data.dropna(subset=["LapTime"], inplace=True)
    lap_data["LapTime (s)"] = lap_data["LapTime"].dt.total_seconds()
    
    return lap_data


def create_qualifying_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with 2025 qualifying times.
    
    Returns:
        DataFrame containing driver names and qualifying times
    """
    drivers = [
        "Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell",
        "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
        "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
    ]
    
    qualifying_times = [
        75.096, 75.180, 75.481, 75.546, 75.670, 75.737,
        75.755, 75.973, 75.980, 76.062, 76.4, 76.5
    ]
    
    return pd.DataFrame({
        "Driver": drivers,
        "QualifyingTime (s)": qualifying_times
    })


def get_driver_code_mapping() -> dict:
    """
    Map full driver names to FastF1 3-letter codes.
    
    Returns:
        Dictionary mapping driver names to codes
    """
    return {
        "Lando Norris": "NOR",
        "Oscar Piastri": "PIA",
        "Max Verstappen": "VER",
        "George Russell": "RUS",
        "Yuki Tsunoda": "TSU",
        "Alexander Albon": "ALB",
        "Charles Leclerc": "LEC",
        "Lewis Hamilton": "HAM",
        "Pierre Gasly": "GAS",
        "Carlos Sainz": "SAI",
        "Lance Stroll": "STR",
        "Fernando Alonso": "ALO"
    }


def prepare_training_data(qualifying_df: pd.DataFrame, historical_laps: pd.DataFrame) -> tuple:
    """
    Merge qualifying and historical data for model training.
    
    Args:
        qualifying_df: DataFrame with qualifying times
        historical_laps: DataFrame with historical lap times
        
    Returns:
        Tuple of (features, targets)
    """
    driver_map = get_driver_code_mapping()
    qualifying_df = qualifying_df.copy()
    qualifying_df["DriverCode"] = qualifying_df["Driver"].map(driver_map)
    
    combined = qualifying_df.merge(
        historical_laps,
        left_on="DriverCode",
        right_on="Driver"
    )
    
    features = combined[["QualifyingTime (s)"]]
    targets = combined["LapTime (s)"]
    
    if features.shape[0] == 0:
        raise ValueError("Dataset is empty after preprocessing. Check data sources!")
    
    return features, targets


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
    
    regressor = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE
    )
    
    regressor.fit(X_train, y_train)
    
    return regressor, X_test, y_test


def generate_predictions(model: GradientBoostingRegressor, qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate race time predictions for all drivers.
    
    Args:
        model: Trained prediction model
        qualifying_df: DataFrame with qualifying times
        
    Returns:
        DataFrame with predictions, sorted by predicted race time
    """
    predictions = model.predict(qualifying_df[["QualifyingTime (s)"]])
    qualifying_df = qualifying_df.copy()
    qualifying_df["PredictedRaceTime (s)"] = predictions
    
    return qualifying_df.sort_values(by="PredictedRaceTime (s)")


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
    print("\n🏁 Predicted 2025 Chinese GP Winner 🏁\n")
    print(predictions_df[["Driver", "PredictedRaceTime (s)"]])
    print(f"\n🔍 Model Error (MAE): {mae:.2f} seconds")


def main():
    """Main execution function."""
    initialize_cache()
    
    historical_data = load_historical_race_data(YEAR, ROUND, SESSION_TYPE)
    qualifying_data = create_qualifying_dataframe()
    
    X, y = prepare_training_data(qualifying_data, historical_data)
    
    trained_model, X_test, y_test = train_prediction_model(X, y)
    
    race_predictions = generate_predictions(trained_model, qualifying_data)
    
    model_error = evaluate_model(trained_model, X_test, y_test)
    
    display_results(race_predictions, model_error)


if __name__ == "__main__":
    main()
