
from pathlib import Path
import pandas as pd

# Project and dataset locations
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-100k"


def load_data():
    """Load users, movies, and ratings from MovieLens 100K."""

    # Load user information
    users = pd.read_csv(
        DATA_DIR / "u.user",
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
        dtype={"zip_code": str}
    )

    # Load movie information
    movie_columns = [
        "movie_id", "title", "release_date", "video_release_date",
        "imdb_url"
    ] + [f"genre_{i}" for i in range(19)]

    movies = pd.read_csv(
        DATA_DIR / "u.item",
        sep="|",
        header=None,
        names=movie_columns,
        encoding="latin-1"
    )

    # Load user-movie ratings
    ratings = pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    return users, movies, ratings


if __name__ == "__main__":
    users, movies, ratings = load_data()

    print("DATA INGESTION SUCCESSFUL")
    print("-------------------------")
    print("Total users:", len(users))
    print("Total movies:", len(movies))
    print("Total ratings:", len(ratings))

    print("\nSample ratings:")
    print(ratings.head())

