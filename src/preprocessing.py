from typing import List, Tuple, Optional, Any
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer


class TimeSeriesScaler:
    def __init__(self, method: Optional[str] = None) -> None:
        self.method = method if method != "none" else None
        self.scaler: Optional[BaseEstimator] = None
        self.is_fitted: bool = False

        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        elif self.method == "quantile":
            self.scaler = QuantileTransformer(output_distribution="normal", n_quantiles=100)
        elif self.method is not None:
            raise ValueError(f"Unknown scaling method: {self.method}")

    def fit(self, series: np.ndarray) -> "TimeSeriesScaler":
        if self.scaler is not None:
            self.scaler.fit(series.reshape(-1, 1))
        self.is_fitted = True
        return self

    def transform(self, series: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform.")
        if self.scaler is None:
            return series.copy()
        return self.scaler.transform(series.reshape(-1, 1)).flatten()

    def inverse_transform(self, series: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        if self.scaler is None:
            return series.copy()
        return self.scaler.inverse_transform(series.reshape(-1, 1)).flatten()

    def fit_transform(self, series: np.ndarray) -> np.ndarray:
        return self.fit(series).transform(series)


def create_lag_features(series: np.ndarray, lags: List[int]) -> np.ndarray:
    n = len(series)
    max_lag = max(lags, default=0)

    if n <= max_lag:
        return np.full((n, len(lags)), np.nan)

    features = np.zeros((n - max_lag, len(lags)))
    for i, lag in enumerate(lags):
        features[:, i] = series[max_lag - lag: n - lag]

    return features


def create_date_features(series_length: int, frequency: str) -> np.ndarray:
    time_idx = np.arange(series_length).reshape(-1, 1)

    if frequency == "H":
        hour, day = time_idx % 24, (time_idx // 24) % 7
        return np.column_stack([
            time_idx,
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 7), np.cos(2 * np.pi * day / 7)
        ])

    if frequency == "D":
        day, month = time_idx % 30, (time_idx // 30) % 12
        return np.column_stack([
            time_idx,
            np.sin(2 * np.pi * day / 30), np.cos(2 * np.pi * day / 30),
            np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
        ])

    return time_idx


def prepare_data_for_model(
        train_series: List[np.ndarray],
        test_series: List[np.ndarray],
        scaler_method: Optional[str],
        **kwargs: Any
) -> Tuple[List[np.ndarray], List[np.ndarray], List[TimeSeriesScaler]]:
    scaled_train, scaled_test, scalers = [], [], []

    for train, test in zip(train_series, test_series):
        scaler = TimeSeriesScaler(method=scaler_method)
        scaled_train.append(scaler.fit_transform(train))
        scaled_test.append(scaler.transform(test))
        scalers.append(scaler)

    return scaled_train, scaled_test, scalers