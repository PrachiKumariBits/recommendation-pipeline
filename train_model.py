
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

def train_model():
    print("STARTING MODEL TRAINING")
    print("-----------------------")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_files = list(PROCESSED_DIR.glob("*train*.csv"))
    if not train_files:
        raise FileNotFoundError("Training CSV not found")

    train = pd.read_csv(train_files[0])

    required = {"user_id", "movie_id", "rating"}
    if not required.issubset(train.columns):
        raise ValueError("Required columns are missing")

    user_ids = sorted(train["user_id"].unique())
    movie_ids = sorted(train["movie_id"].unique())

    user_map = {uid: i for i, uid in enumerate(user_ids)}
    movie_map = {mid: i for i, mid in enumerate(movie_ids)}

    rows = train["user_id"].map(user_map).to_numpy()
    cols = train["movie_id"].map(movie_map).to_numpy()
    ratings = train["rating"].to_numpy()

    matrix = csr_matrix(
        (ratings, (rows, cols)),
        shape=(len(user_ids), len(movie_ids))
    )

    model = TruncatedSVD(n_components=50, random_state=42)
    user_factors = model.fit_transform(matrix)
    movie_factors = model.components_.T

    joblib.dump(
        {
            "model": model,
            "user_factors": user_factors,
            "movie_factors": movie_factors,
            "user_map": user_map,
            "movie_map": movie_map
        },
        MODELS_DIR / "recommendation_model.pkl"
    )

    print("MODEL TRAINING SUCCESSFUL")
    print("Training records:", len(train))
    print("Users:", len(user_ids))
    print("Movies:", len(movie_ids))
    print("Latent factors:", 50)
    print("Model saved in models/")

if __name__ == "__main__":
    train_model()
