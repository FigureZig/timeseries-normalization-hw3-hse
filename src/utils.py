import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_results(
    results_df: pd.DataFrame,
    per_horizon_df: pd.DataFrame,
    impact_df: pd.DataFrame,
    config: Dict[str, Any],
    results_dir: Path
) -> None:
    ensure_dir(results_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_df.to_csv(results_dir / f"results_{timestamp}.csv", index=False)
    per_horizon_df.to_csv(results_dir / f"per_horizon_{timestamp}.csv", index=False)
    impact_df.to_csv(results_dir / f"impact_{timestamp}.csv", index=False)

    with open(results_dir / f"config_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)