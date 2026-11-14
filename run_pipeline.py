
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PIPELINE_STEPS = [
    ("Load Data", "src/ingestion/load_data.py"),
    ("Validate Data", "src/validation/validate_data.py"),
    ("Prepare Data", "src/preparation/prepare_data.py"),
    ("Transform Data", "src/transformation/transform_data.py"),
    ("Build Features", "src/features/build_features.py"),
    ("Train Baseline", "src/training/train_baseline.py"),
    ("Train Recommendation Model", "src/training/train_model.py"),
    ("Evaluate Baseline", "src/evaluation/evaluate_baseline.py"),
    ("Evaluate Model", "src/evaluation/evaluate_model.py"),
    ("Evaluate Ranking", "src/evaluation/evaluate_ranking.py"),
    ("Track Experiments", "src/tracking/track_experiments.py"),
]


def run_pipeline():
    print("=" * 60)
    print("MOVIE RECOMMENDATION PIPELINE")
    print("=" * 60)

    start_time = time.time()

    for step_number, (step_name, script) in enumerate(
        PIPELINE_STEPS, start=1
    ):
        print(f"\nSTEP {step_number}/{len(PIPELINE_STEPS)}: {step_name}")
        print("-" * 60)

        script_path = PROJECT_ROOT / script

        if not script_path.is_file():
            print(f"ERROR: Script not found: {script}")
            sys.exit(1)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=False,
        )

        if result.returncode != 0:
            print(f"\nFAILED: {step_name}")
            print("Pipeline stopped due to an error.")
            sys.exit(result.returncode)

        print(f"COMPLETED: {step_name}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print(f"Total execution time: {elapsed:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
