
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def evaluate_model():
    print("STARTING MODEL EVALUATION")
    print("-------------------------")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load trained recommendation model
    model_data = joblib.load(
        MODELS_DIR / "recommendation_model.pkl"
    )

    user_factors = model_data["user_factors"]
    movie_factors = model_data["movie_factors"]
    user_map = model_data["user_map"]
    movie_map = model_data["movie_map"]

    # Load training and testing datasets
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    # Calculate fallback rating from training data
    global_mean = train["rating"].mean()

    # Start with fallback predictions for all test records
    predictions = np.full(len(test), global_mean)

    # Identify users and movies known to the model
    known_mask = (
        test["user_id"].isin(user_map)
        & test["movie_id"].isin(movie_map)
    )

    known_test = test.loc[known_mask]

    # Predict ratings for known users and movies
    if not known_test.empty:
        user_indices = (
            known_test["user_id"]
            .map(user_map)
            .to_numpy(dtype=int)
        )

        movie_indices = (
            known_test["movie_id"]
            .map(movie_map)
            .to_numpy(dtype=int)
        )

        known_predictions = np.sum(
            user_factors[user_indices]
            * movie_factors[movie_indices],
            axis=1
        )

        predictions[known_mask.to_numpy()] = known_predictions

    # Keep predictions between 1 and 5
    predictions = np.clip(predictions, 1, 5)

    actual = test["rating"].to_numpy()

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)

    # Save results
    results = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "Test Records"],
        "Value": [rmse, mae, len(test)]
    })

    results.to_csv(
        REPORTS_DIR / "evaluation_metrics.csv",
        index=False
    )

    print("MODEL EVALUATION SUCCESSFUL")
    print("Test records evaluated:", len(test))
    print("Known records:", int(known_mask.sum()))
    print("Fallback records:", int((~known_mask).sum()))
    print("RMSE:", round(rmse, 4))
    print("MAE:", round(mae, 4))
    print("Evaluation results saved in reports/")


if __name__ == "__main__":
    evaluate_model()
