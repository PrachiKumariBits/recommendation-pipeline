
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-100k"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def prepare_data():
    print("STARTING DATA PREPARATION")
    print("-------------------------")

    # Load ratings
    ratings = pd.read_csv(
        RAW_DIR / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    # Remove duplicate records
    ratings = ratings.drop_duplicates()

    # Remove invalid ratings
    ratings = ratings[ratings["rating"].between(1, 5)]

    # Sort ratings by timestamp
    ratings = ratings.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Time-based train/test split
    # 80% training, 20% testing
    split_index = int(len(ratings) * 0.8)

    train = ratings.iloc[:split_index].copy()
    test = ratings.iloc[split_index:].copy()

    # Create output directory
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save prepared datasets
    train.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    test.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    print("DATA PREPARATION SUCCESSFUL")
    print(f"Total ratings: {len(ratings)}")
    print(f"Training records: {len(train)}")
    print(f"Testing records: {len(test)}")
    print("Files saved in data/processed/")


if __name__ == "__main__":
    prepare_data()

