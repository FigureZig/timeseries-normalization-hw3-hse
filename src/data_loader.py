import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException

from config import DATA_DIR, M4_FREQUENCY, N_SERIES, RANDOM_SEED

logger = logging.getLogger(__name__)


class M4DataLoader:
    BASE_URL = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset"
    FREQUENCIES = {
        "Hourly": "H",
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y"
    }

    def __init__(self, frequency: str = M4_FREQUENCY) -> None:
        if frequency not in self.FREQUENCIES:
            raise ValueError(f"Invalid frequency: {frequency}")

        self.frequency = frequency
        self.freq_code = self.FREQUENCIES[frequency]
        self.data_dir = Path(DATA_DIR) / f"M4_{frequency}"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._train_file = self.data_dir / f"m4_{self.frequency.lower()}_train.csv"
        self._test_file = self.data_dir / f"m4_{self.frequency.lower()}_test.csv"
        self._cache_file = Path(DATA_DIR) / f"m4_{self.frequency.lower()}_sampled.pkl"

    def _download_file(self, url: str, dest_path: Path) -> None:
        try:
            logger.info(f"Downloading {url} to {dest_path}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
        except RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            raise

    def download_m4_data(self) -> None:
        if self._train_file.exists() and self._test_file.exists():
            logger.info(f"M4 {self.frequency} data already exists locally.")
            return

        train_url = f"{self.BASE_URL}/Train/{self.frequency}-train.csv"
        test_url = f"{self.BASE_URL}/Test/{self.frequency}-test.csv"

        self._download_file(train_url, self._train_file)
        self._download_file(test_url, self._test_file)

        logger.info("M4 data download completed successfully.")

    def load_series(self, n_series: int = N_SERIES) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        self.download_m4_data()

        train_df = pd.read_csv(self._train_file)
        test_df = pd.read_csv(self._test_file)

        rng = np.random.default_rng(RANDOM_SEED)
        n_available = len(train_df)
        sampled_indices = rng.choice(n_available, min(n_series, n_available), replace=False)

        train_series, test_series = [], []

        for idx in sampled_indices:
            train_values = train_df.iloc[idx, 1:].dropna().values.astype(float)
            test_values = test_df.iloc[idx, 1:].dropna().values.astype(float)

            if len(train_values) > 0 and len(test_values) > 0:
                train_series.append(train_values)
                test_series.append(test_values)

        logger.info(f"Loaded {len(train_series)} series from M4 {self.frequency}")
        return train_series, test_series

    def cache_series(self, train_series: List[np.ndarray], test_series: List[np.ndarray]) -> None:
        try:
            with open(self._cache_file, "wb") as f:
                pickle.dump({"train": train_series, "test": test_series}, f)
            logger.info(f"Series cached successfully at {self._cache_file}")
        except IOError as e:
            logger.error(f"Failed to cache series: {e}")

    def load_cached_series(self) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        if not self._cache_file.exists():
            logger.info("Cache file not found.")
            return None, None

        try:
            with open(self._cache_file, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Successfully loaded cached series from {self._cache_file}")
            return data.get("train"), data.get("test")
        except (IOError, pickle.UnpicklingError) as e:
            logger.error(f"Failed to load cached series: {e}")
            return None, None