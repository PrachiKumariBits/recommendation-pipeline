
from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def train_baseline():
    print("STARTING BASELINE MODEL TRAINING")
    print("--------------------------------")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data
    train = pd.read_csv(PROCESSED_DIR / "train.csv")

    # Calculate movie popularity from training data only
    movie_popularity = (
        train.groupby("movie_id")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    # Rank movies by number of ratings
    movie_popularity = movie_popularity.sort_values(
        by=["rating_count", "average_rating", "movie_id"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    # Save the baseline model
    joblib.dump(
        movie_popularity,
        MODELS_DIR / "popularity_model.pkl"
    )

    movie_popularity.to_csv(
        MODELS_DIR / "movie_popularity.csv",
        index=False
    )

    print("BASELINE MODEL TRAINING SUCCESSFUL")
    print("Training records:", len(train))
    print("Movies ranked:", len(movie_popularity))
    print("Model saved in models/")


if __name__ == "__main__":
    train_baseline()
