
# Movie Recommendation Pipeline
## Data Management for Machine Learning - Assignment 2

### 1. Project Overview

This project implements an end-to-end movie recommendation pipeline using the MovieLens dataset.

The pipeline covers data ingestion, validation, preparation, transformation, feature engineering, model training, evaluation, experiment tracking, and orchestration.

Two recommendation approaches are evaluated:
- Popularity-based recommendation (baseline)
- Matrix factorization using Singular Value Decomposition (SVD)

### 2. Technologies Used

- Python 3.11
- Pandas and NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Joblib
- MLflow
- macOS Terminal

### 3. Project Structure

recommendation-pipeline/
- data/
  - raw/
  - processed/
  - features/
- models/
- reports/
- src/
  - ingestion/load_data.py
  - validation/validate_data.py
  - preparation/prepare_data.py
  - transformation/transform_data.py
  - features/build_features.py
  - training/train_baseline.py
  - training/train_model.py
  - evaluation/evaluate_baseline.py
  - evaluation/evaluate_model.py
  - evaluation/evaluate_ranking.py
  - tracking/track_experiments.py
  - orchestration/run_pipeline.py
- README.md

### 4. Pipeline Workflow

The pipeline executes the following stages:

1. Data Ingestion
2. Data Validation
3. Data Preparation
4. Data Transformation
5. Feature Engineering
6. Baseline Model Training
7. SVD Model Training
8. Baseline Evaluation
9. SVD Model Evaluation
10. Ranking Evaluation
11. MLflow Experiment Tracking

The orchestration script runs these stages sequentially and stops if a stage fails.

### 5. Running the Pipeline

Open Terminal and navigate to the project directory:

```bash
cd ~/Desktop/recommendation-pipeline
```

Activate the Python virtual environment:

```bash
source .venv/bin/activate
```

Run the complete pipeline:

```bash
python src/orchestration/run_pipeline.py
```

The pipeline generates model artifacts and evaluation reports.

### 6. Model Evaluation

Rating prediction is evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

Recommendation ranking is evaluated using:

- Precision@5
- Recall@5
- NDCG@5
- Coverage@5
- Diversity@5

### 7. Ranking Evaluation Results

| Model | Precision@5 | Recall@5 | NDCG@5 | Coverage@5 | Diversity@5 |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.258621 | 0.053145 | 0.276046 | 0.029703 | 0.833762 |
| SVD | 0.262759 | 0.050232 | 0.282701 | 0.158416 | 0.853226 |

The SVD model achieves higher Precision@5, NDCG@5, Coverage@5, and Diversity@5. The popularity model achieves slightly higher Recall@5.

### 8. Generated Reports

The following files are available in the reports directory:

- baseline_metrics.csv
- evaluation_metrics.csv
- model_comparison.csv
- model_comparison.png
- ranking_metrics.csv

### 9. Experiment Tracking

MLflow is used to record and compare recommendation model experiments.

The experiment is named:

Movie_Recommendation_Comparison

Two model runs are recorded:
- Popularity
- SVD

To launch the MLflow interface using the local SQLite tracking database:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the interface at:

http://127.0.0.1:5000

### 10. Pipeline Execution Results

The complete pipeline executed successfully.

- Total stages: 11
- Completed stages: 11
- Execution time: 7.83 seconds
- Experiment tracking: Successful

### 11. Conclusion

The project demonstrates an end-to-end movie recommendation workflow, including data processing, model training, evaluation, experiment tracking, and pipeline orchestration.

The SVD model provides improvements in several ranking metrics compared with the popularity-based baseline, particularly recommendation coverage.

The pipeline is reproducible using the orchestration script, provided the required dataset and Python dependencies are available.
