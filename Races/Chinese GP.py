"""
F1 Race Prediction Model - Chinese GP Analysis
Predicts race performance based on qualifying times and sector times using historical race data.
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
GRAND_PRIX = "China"
SESSION_TYPE = "R"
RANDOM_STATE = 38
TEST_SIZE = 0.2
N_ESTIMATORS = 200
LEARNING_RATE = 0.1


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
    Calculate average sector times per driver from lap data.
    
    Args:
        lap_data: DataFrame with lap and sector time data
        
    Returns:
        DataFrame with average sector times grouped by driver
    """
    sector_columns = ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
    return lap_data.groupby("Driver")[sector_columns].mean().reset_index()


def compute_average_lap_times(lap_data: pd.DataFrame) -> pd.Series:
    """
    Calculate average lap times per driver.
    
    Args:
        lap_data: DataFrame with lap time data
        
    Returns:
        Series with average lap times per driver
    """
    return lap_data.groupby("Driver")["LapTime (s)"].mean().reset_index()["LapTime (s)"]


def create_qualifying_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with 2025 Chinese GP qualifying times.
    
    Returns:
        DataFrame containing driver names and qualifying times
    """
    drivers = [
        "Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen",
        "Lewis Hamilton", "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli",
        "Yuki Tsunoda", "Alexander Albon", "Esteban Ocon", "Nico Hülkenberg",
        "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.", "Pierre Gasly",
        "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"
    ]
    
    qualifying_times = [
        90.641, 90.723, 90.793, 90.817, 90.927, 91.021, 91.079, 91.103,
        91.638, 91.706, 91.625, 91.632, 91.688, 91.773, 91.840, 91.992,
        92.018, 92.092, 92.141, 92.174
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
        "Oscar Piastri": "PIA",
        "George Russell": "RUS",
        "Lando Norris": "NOR",
        "Max Verstappen": "VER",
        "Lewis Hamilton": "HAM",
        "Charles Leclerc": "LEC",
        "Isack Hadjar": "HAD",
        "Andrea Kimi Antonelli": "ANT",
        "Yuki Tsunoda": "TSU",
        "Alexander Albon": "ALB",
        "Esteban Ocon": "OCO",
        "Nico Hülkenberg": "HUL",
        "Fernando Alonso": "ALO",
        "Lance Stroll": "STR",
        "Carlos Sainz Jr.": "SAI",
        "Pierre Gasly": "GAS",
        "Oliver Bearman": "BEA",
        "Jack Doohan": "DOO",
        "Gabriel Bortoleto": "BOR",
        "Liam Lawson": "LAW"
    }


def prepare_training_data(
    qualifying_df: pd.DataFrame,
    sector_times: pd.DataFrame,
    average_lap_times: pd.Series
) -> tuple:
    """
    Merge qualifying data with sector times and prepare features for training.
    
    Args:
        qualifying_df: DataFrame with qualifying times
        sector_times: DataFrame with average sector times per driver
        average_lap_times: Series with average lap times per driver
        
    Returns:
        Tuple of (features, targets)
    """
    driver_map = get_driver_code_mapping()
    qualifying_df = qualifying_df.copy()
    qualifying_df["DriverCode"] = qualifying_df["Driver"].map(driver_map)
    
    combined = qualifying_df.merge(
        sector_times,
        left_on="DriverCode",
        right_on="Driver",
        how="left"
    )
    
    feature_columns = [
        "QualifyingTime (s)",
        "Sector1Time (s)",
        "Sector2Time (s)",
        "Sector3Time (s)"
    ]
    features = combined[feature_columns].fillna(0)
    
    return features, average_lap_times


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


def generate_predictions(model: GradientBoostingRegressor, features: pd.DataFrame) -> pd.Series:
    """
    Generate race time predictions using the trained model.
    
    Args:
        model: Trained prediction model
        features: Feature data for prediction
        
    Returns:
        Series with predicted race times
    """
    return pd.Series(model.predict(features), name="PredictedRaceTime (s)")


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
    print("\n🏁 Predicted 2025 Chinese GP Winner with New Drivers and Sector Times 🏁\n")
    print(predictions_df[["Driver", "PredictedRaceTime (s)"]])
    print(f"\n🔍 Model Error (MAE): {mae:.2f} seconds")


def main():
    """Main execution function."""
    initialize_cache()
    
    historical_data = load_historical_race_data(YEAR, GRAND_PRIX, SESSION_TYPE)
    sector_times = compute_average_sector_times(historical_data)
    average_lap_times = compute_average_lap_times(historical_data)
    
    qualifying_data = create_qualifying_dataframe()
    
    X, y = prepare_training_data(qualifying_data, sector_times, average_lap_times)
    
    trained_model, X_test, y_test = train_prediction_model(X, y)
    
    race_predictions = generate_predictions(trained_model, X)
    qualifying_data = qualifying_data.copy()
    qualifying_data["PredictedRaceTime (s)"] = race_predictions.values
    qualifying_data = qualifying_data.sort_values(by="PredictedRaceTime (s)")
    
    model_error = evaluate_model(trained_model, X_test, y_test)
    
    display_results(qualifying_data, model_error)


if __name__ == "__main__":
    main()
