
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

K = 5
RELEVANCE_THRESHOLD = 4


def ranking_metrics(recommendations, relevant, k):
    hits = [1 if movie in relevant else 0
            for movie in recommendations[:k]]

    precision = sum(hits) / k
    recall = sum(hits) / len(relevant) if relevant else 0.0

    dcg = sum(hit / np.log2(i + 2)
              for i, hit in enumerate(hits))
    ideal = sum(1 / np.log2(i + 2)
                for i in range(min(len(relevant), k)))
    ndcg = dcg / ideal if ideal else 0.0

    return precision, recall, ndcg


def evaluate_ranking():
    print("STARTING RANKING EVALUATION")

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    svd = joblib.load(MODELS_DIR / "recommendation_model.pkl")
    popularity = joblib.load(MODELS_DIR / "popularity_model.pkl")

    user_factors = svd["user_factors"]
    movie_factors = svd["movie_factors"]
    user_map = svd["user_map"]
    movie_map = svd["movie_map"]

    # Use the movie catalog from training data
    movie_ids = list(movie_map.keys())
    popularity_counts = train.groupby("movie_id").size()
    popular_movies = popularity_counts.sort_values(
        ascending=False
    ).index.tolist()

    # Users with at least one relevant test interaction
    relevant_test = test[test["rating"] >= RELEVANCE_THRESHOLD]
    relevant_by_user = relevant_test.groupby("user_id")["movie_id"].apply(set)

    seen_by_user = train.groupby("user_id")["movie_id"].apply(set).to_dict()

    results = []

    for model_name in ["Popularity", "SVD"]:
        precisions, recalls, ndcgs = [], [], []
        recommended_items = set()
        diversity_scores = []

        for user_id, relevant in relevant_by_user.items():
            seen = seen_by_user.get(user_id, set())
            candidates = [m for m in movie_ids if m not in seen]

            if not candidates:
                continue

            if model_name == "Popularity":
                candidate_set = set(candidates)
                recommendations = [
                    m for m in popular_movies if m in candidate_set
                ][:K]

            else:
                if user_id not in user_map:
                    candidate_set = set(candidates)
                    recommendations = [
                        m for m in popular_movies if m in candidate_set
                    ][:K]
                else:
                    u = user_factors[user_map[user_id]]
                    scored = []

                    for m in candidates:
                        if m in movie_map:
                            score = float(np.dot(
                                u, movie_factors[movie_map[m]]
                            ))
                            scored.append((m, score))

                    scored.sort(key=lambda x: x[1], reverse=True)
                    recommendations = [m for m, _ in scored[:K]]

            if not recommendations:
                continue

            p, r, n = ranking_metrics(recommendations, relevant, K)
            precisions.append(p)
            recalls.append(r)
            ndcgs.append(n)

            recommended_items.update(recommendations)

            # Intra-list diversity using item latent vectors
            vectors = [
                movie_factors[movie_map[m]]
                for m in recommendations if m in movie_map
            ]

            pairwise = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    a, b = vectors[i], vectors[j]
                    denominator = np.linalg.norm(a) * np.linalg.norm(b)
                    if denominator > 0:
                        similarity = np.dot(a, b) / denominator
                        pairwise.append(1 - similarity)

            if pairwise:
                diversity_scores.append(float(np.mean(pairwise)))

        results.append({
            "Model": model_name,
            "Precision@5": np.mean(precisions) if precisions else 0,
            "Recall@5": np.mean(recalls) if recalls else 0,
            "NDCG@5": np.mean(ndcgs) if ndcgs else 0,
            "Coverage@5": len(recommended_items) / len(movie_ids),
            "Diversity@5": np.mean(diversity_scores) if diversity_scores else 0
        })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(REPORTS_DIR / "ranking_metrics.csv", index=False)

    print(result_df.to_string(index=False))
    print("RANKING EVALUATION COMPLETED")
    print("Results saved to reports/ranking_metrics.csv")


if __name__ == "__main__":
    evaluate_ranking()
