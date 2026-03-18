from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostRegressor
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm


class NaiveModel:
    def __init__(self) -> None:
        self.last_value: float = 0.0

    def fit(self, series: np.ndarray) -> "NaiveModel":
        self.last_value = float(series[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last_value)


class SeasonalNaiveModel:
    def __init__(self, season_length: int = 24) -> None:
        self.season_length = season_length
        self.seasonal_values: Optional[np.ndarray] = None
        self.fallback_value: float = 0.0

    def fit(self, series: np.ndarray) -> "SeasonalNaiveModel":
        self.fallback_value = float(series[-1])
        if len(series) < self.season_length:
            self.seasonal_values = np.array([self.fallback_value])
        else:
            self.seasonal_values = series[-self.season_length:].copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.seasonal_values is None:
            return np.full(horizon, self.fallback_value)

        n_seasons = len(self.seasonal_values)
        indices = np.arange(horizon) % n_seasons
        return self.seasonal_values[indices]


class AutoThetaModel:
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.fallback_value: float = 0.0

    def fit(self, series: np.ndarray) -> "AutoThetaModel":
        self.fallback_value = float(series[-1])
        try:
            self.model = ThetaModel(series).fit()
        except Exception:
            self.model = None
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.model is None:
            return np.full(horizon, self.fallback_value)

        try:
            return self.model.forecast(horizon).to_numpy()
        except Exception:
            return np.full(horizon, self.fallback_value)


class AutoETSModel:
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.fallback_value: float = 0.0

    def fit(self, series: np.ndarray) -> "AutoETSModel":
        self.fallback_value = float(series[-1])
        try:
            if len(series) > 24:
                self.model = ExponentialSmoothing(
                    series, seasonal_periods=24, trend=True, seasonal="add"
                ).fit()
            else:
                self.model = ExponentialSmoothing(series, trend=True).fit()
        except Exception:
            self.model = None
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.model is None:
            return np.full(horizon, self.fallback_value)

        try:
            return self.model.forecast(horizon).to_numpy()
        except Exception:
            return np.full(horizon, self.fallback_value)


class CatBoostModel:
    def __init__(self, params: Dict[str, Any], lags: List[int], prediction_length: int) -> None:
        self.lags = lags
        self.prediction_length = prediction_length
        self.model = CatBoostRegressor(**params)

    def _create_features(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        max_lag = max(self.lags)

        if n <= max_lag:
            return np.empty((0, len(self.lags)))

        features = np.zeros((n - max_lag, len(self.lags)))
        for i, lag in enumerate(self.lags):
            features[:, i] = series[max_lag - lag: -lag]

        return features

    def fit(self, train_series: List[np.ndarray]) -> "CatBoostModel":
        X_list, y_list = [], []
        max_lag = max(self.lags)

        for series in train_series:
            if len(series) <= max_lag:
                continue
            X_list.append(self._create_features(series))
            y_list.append(series[max_lag:])

        if not X_list:
            return self

        X_all = np.vstack(X_list)
        y_all = np.hstack(y_list)

        self.model.fit(X_all, y_all, verbose=False)
        return self

    def predict(self, test_series: List[np.ndarray]) -> List[np.ndarray]:
        predictions = []
        max_lag = max(self.lags)

        for series in test_series:
            current_series = series.copy()
            pred = np.zeros(self.prediction_length)

            for i in range(self.prediction_length):
                if len(current_series) < max_lag:
                    next_val = current_series[-1] if len(current_series) > 0 else 0.0
                else:
                    features = current_series[-max_lag:]
                    X_pred = np.array([features[-lag] for lag in self.lags]).reshape(1, -1)
                    next_val = self.model.predict(X_pred)[0]

                pred[i] = next_val
                current_series = np.append(current_series, next_val)

            predictions.append(pred)

        return predictions


class PatchEmbedding(nn.Module):
    def __init__(self, patch_length: int, stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.linear = nn.Linear(patch_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        patches = []

        for i in range(0, seq_len - self.patch_length + 1, self.stride):
            patches.append(x[:, i: i + self.patch_length])

        if not patches:
            patches = [x]

        patches_tensor = torch.stack(patches, dim=1)
        return self.linear(patches_tensor)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class PatchTST(nn.Module):
    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            patch_length: int = 24,
            stride: int = 8,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 3,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_length, stride, d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.output_projection(x)


class PatchTSTModel:
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[PatchTST] = None
        self.series_stats: List[Tuple[float, float]] = []
        self.train_losses: List[float] = []

    def _normalize_series(self, series: np.ndarray) -> Tuple[np.ndarray, float, float]:
        mean = float(np.mean(series))
        std = float(np.std(series))
        if std < 1e-8:
            std = 1.0
        return (series - mean) / std, mean, std

    def _prepare_data(self, series_list: List[np.ndarray], context_length: int) -> Tuple[
        Optional[DataLoader], List[Tuple[float, float]]]:
        X, y, stats = [], [], []
        pred_len = self.params["prediction_length"]

        for series in series_list:
            if len(series) <= context_length + pred_len:
                continue

            normalized_series, mean, std = self._normalize_series(series)
            stats.append((mean, std))

            for i in range(0, len(normalized_series) - context_length - pred_len + 1, context_length // 2):
                X.append(normalized_series[i: i + context_length])
                y.append(normalized_series[i + context_length: i + context_length + pred_len])

        if not X:
            return None, []

        dataset = TensorDataset(torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)))
        dataloader = DataLoader(
            dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            drop_last=True
        )
        return dataloader, stats

    def fit(self, train_series: List[np.ndarray]) -> "PatchTSTModel":
        self.model = PatchTST(
            context_length=self.params["context_length"],
            prediction_length=self.params["prediction_length"],
            patch_length=self.params["patch_length"],
            stride=self.params["stride"],
            d_model=self.params["hidden_size"],
            n_heads=self.params["num_heads"],
            n_layers=self.params["num_layers"],
            dropout=self.params["dropout"]
        ).to(self.device)

        dataloader, self.series_stats = self._prepare_data(train_series, self.params["context_length"])

        if dataloader is None:
            return self

        optimizer = optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])
        criterion = nn.MSELoss()

        self.model.train()
        self.train_losses = []

        epoch_iterator = tqdm(range(self.params["epochs"]), desc="PatchTST Training", leave=False)
        for epoch in epoch_iterator:
            total_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            self.train_losses.append(avg_loss)

            epoch_iterator.set_postfix(loss=f"{avg_loss:.4f}")

        return self

    def predict(self, test_series: List[np.ndarray]) -> List[np.ndarray]:
        if self.model is None:
            return [np.full(self.params["prediction_length"], np.nan) for _ in test_series]

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for series in test_series:
                normalized_series, mean, std = self._normalize_series(series)
                context = normalized_series[-self.params["context_length"]:]

                if len(context) < self.params["context_length"]:
                    context = np.pad(context, (self.params["context_length"] - len(context), 0), "edge")

                X = torch.FloatTensor(context).unsqueeze(0).to(self.device)
                pred_normalized = self.model(X).cpu().numpy().flatten()

                predictions.append(pred_normalized * std + mean)

        return predictions