
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-100k"


def validate_data():
    print("STARTING DATA VALIDATION")
    print("------------------------")

    files = {
        "users": DATA_DIR / "u.user",
        "movies": DATA_DIR / "u.item",
        "ratings": DATA_DIR / "u.data"
    }

    # Check that all required files exist
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} file: {path}")

    # Load users
    users = pd.read_csv(
        files["users"],
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1"
    )

    # Load movies
    movies = pd.read_csv(
        files["movies"],
        sep="|",
        header=None,
        encoding="latin-1"
    )

    # Load ratings
    ratings = pd.read_csv(
        files["ratings"],
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    # Dataset summary
    print(f"Total users: {len(users)}")
    print(f"Total movies: {len(movies)}")
    print(f"Total ratings: {len(ratings)}")

    # Missing values
    print("\nMissing values:")
    print("Users:", users.isnull().sum().sum())
    print("Movies:", movies.isnull().sum().sum())
    print("Ratings:", ratings.isnull().sum().sum())

    # Duplicate and invalid data checks
    print("\nDuplicate ratings:", ratings.duplicated().sum())
    print("Invalid ratings:", (~ratings["rating"].between(1, 5)).sum())

    print(
        "Invalid user IDs:",
        (~ratings["user_id"].isin(users["user_id"])).sum()
    )

    print(
        "Invalid movie IDs:",
        (~ratings["movie_id"].isin(movies[0])).sum()
    )

    # Validate movie information
    print("\nMovie information validation:")
    print("Missing Movie IDs:", movies[0].isnull().sum())
    print("Missing Movie Titles:", movies[1].isnull().sum())
    print("Duplicate Movie IDs:", movies[0].duplicated().sum())

    # Validate rating timestamps
    timestamps = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        errors="coerce",
        utc=True
    )

    print("\nTimestamp validation:")
    print("Missing timestamps:", ratings["timestamp"].isna().sum())
    print("Invalid timestamps:", timestamps.isna().sum())
    print("Earliest interaction:", timestamps.min())
    print("Latest interaction:", timestamps.max())

    print("\nDATA VALIDATION COMPLETED")


if __name__ == "__main__":
    validate_data()
