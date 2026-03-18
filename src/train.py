from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.models import (
    NaiveModel,
    SeasonalNaiveModel,
    AutoThetaModel,
    AutoETSModel,
    CatBoostModel,
    PatchTSTModel
)
from src.preprocessing import prepare_data_for_model


def train_global_model(
        train_series: List[np.ndarray],
        model_type: str,
        params: Dict[str, Any],
        lags: Optional[List[int]] = None,
        prediction_length: Optional[int] = None
) -> Any:
    if model_type == "catboost":
        if lags is None or prediction_length is None:
            raise ValueError()
        return CatBoostModel(
            params=params,
            lags=lags,
            prediction_length=prediction_length
        ).fit(train_series)

    if model_type == "patchtst":
        return PatchTSTModel(params=params).fit(train_series)

    raise ValueError()


def run_experiment(
        train_series: List[np.ndarray],
        test_series: List[np.ndarray],
        scaler_method: Optional[str],
        model_configs: Dict[str, Any],
        lags: List[int],
        prediction_length: int,
        season_length: int = 24
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Any]]:
    scaled_train, _, scalers = prepare_data_for_model(
        train_series=train_series,
        test_series=test_series,
        scaler_method=scaler_method
    )

    predictions: Dict[str, List[np.ndarray]] = {
        "naive": [],
        "seasonal_naive": [],
        "auto_theta": [],
        "auto_ets": [],
        "catboost": [],
        "patchtst": []
    }

    metadata: Dict[str, Any] = {}

    for train, scaler in zip(scaled_train, scalers):
        predictions["naive"].append(
            scaler.inverse_transform(NaiveModel().fit(train).predict(prediction_length))
        )
        predictions["seasonal_naive"].append(
            scaler.inverse_transform(SeasonalNaiveModel(season_length).fit(train).predict(prediction_length))
        )
        predictions["auto_theta"].append(
            scaler.inverse_transform(AutoThetaModel().fit(train).predict(prediction_length))
        )
        predictions["auto_ets"].append(
            scaler.inverse_transform(AutoETSModel().fit(train).predict(prediction_length))
        )

    if "catboost" in model_configs:
        cb_model = train_global_model(
            scaled_train, "catboost", model_configs["catboost"], lags, prediction_length
        )
        predictions["catboost"] = [
            scaler.inverse_transform(pred) for pred, scaler in zip(cb_model.predict(scaled_train), scalers)
        ]

    if "patchtst" in model_configs:
        pt_model = train_global_model(
            scaled_train, "patchtst", model_configs["patchtst"]
        )
        predictions["patchtst"] = [
            scaler.inverse_transform(pred) for pred, scaler in zip(pt_model.predict(scaled_train), scalers)
        ]
        metadata["patchtst_train_losses"] = pt_model.train_losses

    return predictions, metadata