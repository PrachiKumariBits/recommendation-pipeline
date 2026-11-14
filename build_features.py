
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

def build_features():
    print("STARTING FEATURE ENGINEERING")
    print("----------------------------")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_files = sorted(PROCESSED_DIR.glob("*train*.csv"))
    if not train_files:
        raise FileNotFoundError("Training CSV not found")

    train = pd.read_csv(train_files[0])

    # Calculate overall average rating
    global_mean = train["rating"].mean()

    # Build user features
    user_features = train.groupby("user_id")["rating"].agg(
        ["mean", "count"]
    ).reset_index()

    user_features.columns = [
        "user_id", "user_avg_rating", "user_rating_count"
    ]

    user_features["user_rating_bias"] = (
        user_features["user_avg_rating"] - global_mean
    )

    # Build movie features
    movie_features = train.groupby("movie_id")["rating"].agg(
        ["mean", "count"]
    ).reset_index()

    movie_features.columns = [
        "movie_id", "movie_avg_rating", "movie_rating_count"
    ]

    movie_features["movie_rating_bias"] = (
        movie_features["movie_avg_rating"] - global_mean
    )

    # Save features
    user_features.to_csv(
        FEATURES_DIR / "user_features.csv", index=False
    )

    movie_features.to_csv(
        FEATURES_DIR / "movie_features.csv", index=False
    )

    print("FEATURE ENGINEERING SUCCESSFUL")
    print(f"Global average rating: {global_mean:.4f}")
    print(f"Total user features: {len(user_features)}")
    print(f"Total movie features: {len(movie_features)}")
    print("Files saved in data/features/")

if __name__ == "__main__":
    build_features()
