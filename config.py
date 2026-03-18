from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"
RESULTS_DIR: Path = BASE_DIR / "results"

M4_FREQUENCY: str = "Hourly"
N_SERIES: int = 50
TEST_SIZE: int = 48
RANDOM_SEED: int = 42

CATBOOST_PARAMS: Dict[str, Any] = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "RMSE",
    "verbose": False,
    "random_seed": RANDOM_SEED
}

PATCHTST_PARAMS: Dict[str, Any] = {
    "context_length": 168,
    "prediction_length": TEST_SIZE,
    "patch_length": 24,
    "stride": 8,
    "num_layers": 3,
    "hidden_size": 128,
    "num_heads": 4,
    "dropout": 0.1,
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001
}

LAGS: List[int] = [1, 2, 3, 4, 5, 6, 12, 24, 48]
SEASONAL_LAGS: List[int] = [24, 48, 168]

SCALING_METHODS: List[Optional[str]] = [None, "standard", "robust", "quantile"]
MODELS: List[str] = ["naive", "seasonal_naive", "auto_theta", "auto_ets", "catboost", "patchtst"]
METRICS: List[str] = ["smape", "mae", "rmse"]