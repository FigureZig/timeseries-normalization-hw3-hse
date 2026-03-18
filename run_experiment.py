import argparse
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd

from tqdm import tqdm

from config import (
    DATA_DIR, RESULTS_DIR, M4_FREQUENCY, N_SERIES, TEST_SIZE,
    CATBOOST_PARAMS, PATCHTST_PARAMS, LAGS, SEASONAL_LAGS,
    SCALING_METHODS, MODELS, METRICS, RANDOM_SEED
)
from src.data_loader import M4DataLoader
from src.evaluate import evaluate_experiment, analyze_scaling_impact
from src.train import run_experiment
from src.utils import set_seed, save_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_series", type=int, default=N_SERIES)
    parser.add_argument("--force_reload", action="store_true")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    loader = M4DataLoader(frequency=M4_FREQUENCY)
    train_series, test_series = loader.load_cached_series()

    if train_series is None or args.force_reload:
        train_series, test_series = loader.load_series(n_series=args.n_series)
        loader.cache_series(train_series, test_series)
    else:
        train_series = train_series[:args.n_series]
        test_series = test_series[:args.n_series]

    if not train_series:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RESULTS_DIR) / f"run_{timestamp}"
    data_dir = run_dir / "data"
    plots_dir = run_dir / "plots"

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_results_df = {}
    all_per_horizon_df = {}
    all_predictions = {}
    all_metadata = {}

    model_configs = {"catboost": CATBOOST_PARAMS, "patchtst": PATCHTST_PARAMS}

    for scaler in tqdm(SCALING_METHODS, desc="Total Experiment Progress", position=0):
        scaler_name = scaler if scaler else "none"

        predictions, metadata = run_experiment(
            train_series=train_series,
            test_series=test_series,
            scaler_method=scaler,
            model_configs=model_configs,
            lags=LAGS,
            prediction_length=TEST_SIZE,
            season_length=SEASONAL_LAGS[0]
        )

        res_df, horizon_df = evaluate_experiment(test_series, predictions, METRICS)
        all_results_df[scaler_name] = res_df
        all_per_horizon_df[scaler_name] = horizon_df
        all_predictions[scaler_name] = predictions
        all_metadata[scaler_name] = metadata

    impact_df = analyze_scaling_impact(
        results_all_scaling=all_results_df,
        scaling_methods=[m if m else "none" for m in SCALING_METHODS],
        baseline_models=["naive", "seasonal_naive", "auto_theta", "auto_ets"],
        global_models=["catboost", "patchtst"]
    )

    config_meta = {
        "frequency": M4_FREQUENCY,
        "n_series": args.n_series,
        "test_size": TEST_SIZE,
        "scaling_methods": SCALING_METHODS,
        "models": MODELS,
        "metrics": METRICS,
        "random_seed": RANDOM_SEED
    }

    final_results = pd.concat(all_results_df.values(), keys=all_results_df.keys()).reset_index(level=0).rename(
        columns={'level_0': 'scaling'})
    final_horizon = pd.concat(all_per_horizon_df.values(), keys=all_per_horizon_df.keys()).reset_index(level=0).rename(
        columns={'level_0': 'scaling'})

    save_results(final_results, final_horizon, impact_df, config_meta, data_dir)

    with open(data_dir / "raw_data.pkl", "wb") as f:
        pickle.dump({
            "train_series": train_series,
            "test_series": test_series,
            "predictions": all_predictions,
            "metadata": all_metadata
        }, f)


if __name__ == "__main__":
    main()