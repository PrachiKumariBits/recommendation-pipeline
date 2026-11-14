
from pathlib import Path
import pandas as pd
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def track_experiments():
    print("STARTING MLFLOW EXPERIMENT TRACKING")
    print("-----------------------------------")

    # Use SQLite database for MLflow tracking
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment(
        "Movie_Recommendation_Comparison"
    )

    # Load existing evaluation results
    comparison_file = REPORTS_DIR / "model_comparison.csv"
    ranking_file = REPORTS_DIR / "ranking_metrics.csv"

    if not comparison_file.exists():
        raise FileNotFoundError(comparison_file)

    if not ranking_file.exists():
        raise FileNotFoundError(ranking_file)

    comparison = pd.read_csv(comparison_file)
    ranking = pd.read_csv(ranking_file)

    # Validate required columns
    required_comparison = {
        "Metric",
        "Value_Baseline",
        "Value_Recommendation"
    }

    required_ranking = {
        "Model",
        "Precision@5",
        "Recall@5",
        "NDCG@5",
        "Coverage@5",
        "Diversity@5"
    }

    if not required_comparison.issubset(comparison.columns):
        raise ValueError(
            "Missing model comparison columns"
        )

    if not required_ranking.issubset(ranking.columns):
        raise ValueError(
            "Missing ranking metrics columns"
        )

    metric_names = {
        "RMSE": "rmse",
        "MAE": "mae"
    }

    models = {
        "Popularity": "Value_Baseline",
        "SVD": "Value_Recommendation"
    }

    # Log metrics for each recommendation model
    for model_name, comparison_column in models.items():

        model_ranking = ranking[
            ranking["Model"] == model_name
        ]

        if model_ranking.empty:
            raise ValueError(
                f"Ranking results missing for {model_name}"
            )

        with mlflow.start_run(run_name=model_name):

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("top_k", 5)

            if model_name == "SVD":
                mlflow.log_param("latent_factors", 50)

            # Log RMSE and MAE
            for original_name, logged_name in metric_names.items():

                matching = comparison[
                    comparison["Metric"] == original_name
                ]

                if matching.empty:
                    raise ValueError(
                        f"Missing metric: {original_name}"
                    )

                value = float(
                    matching.iloc[0][comparison_column]
                )

                mlflow.log_metric(logged_name, value)

            # Log ranking evaluation metrics
            row = model_ranking.iloc[0]

            for metric in [
                "Precision@5",
                "Recall@5",
                "NDCG@5",
                "Coverage@5",
                "Diversity@5"
            ]:
                mlflow.log_metric(
                    metric.lower().replace("@", "_at_"),
                    float(row[metric])
                )

            # Save evaluation files as MLflow artifacts
            mlflow.log_artifact(str(comparison_file))
            mlflow.log_artifact(str(ranking_file))

            print(f"Logged experiment: {model_name}")

    print("-----------------------------------")
    print("MLFLOW EXPERIMENT TRACKING COMPLETED")


if __name__ == "__main__":
    track_experiments()

