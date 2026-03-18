from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def calculate_metrics(
        y_true: List[np.ndarray],
        y_pred: List[np.ndarray],
        metrics: List[str]
) -> Dict[str, float]:
    y_t_flat = np.concatenate(y_true)
    y_p_flat = np.concatenate(y_pred)

    results = {}
    if "smape" in metrics:
        results["smape"] = smape(y_t_flat, y_p_flat)
    if "mae" in metrics:
        results["mae"] = float(mean_absolute_error(y_t_flat, y_p_flat))
    if "rmse" in metrics:
        results["rmse"] = float(np.sqrt(mean_squared_error(y_t_flat, y_p_flat)))

    return results


def evaluate_experiment(
        test_series: List[np.ndarray],
        predictions: Dict[str, List[np.ndarray]],
        metrics: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_records = []
    horizon_records = []

    for model_name, model_preds in predictions.items():
        if not model_preds:
            continue

        metric_vals = calculate_metrics(test_series, model_preds, metrics)
        results_records.append({"model": model_name, **metric_vals})

        try:
            y_t_arr = np.array(test_series)
            y_p_arr = np.array(model_preds)

            for h in range(y_t_arr.shape[1]):
                h_true, h_pred = y_t_arr[:, h], y_p_arr[:, h]
                for m in metrics:
                    val = 0.0
                    if m == "smape":
                        val = smape(h_true, h_pred)
                    elif m == "mae":
                        val = float(mean_absolute_error(h_true, h_pred))
                    elif m == "rmse":
                        val = float(np.sqrt(mean_squared_error(h_true, h_pred)))

                    horizon_records.append({
                        "model": model_name, "horizon": h + 1, "metric": m, "value": val
                    })
        except ValueError:
            # Медленный путь: для рядов разной длины
            max_horizon = max((len(s) for s in test_series), default=0)
            for h in range(max_horizon):
                y_t_h, y_p_h = [], []
                for t_s, p_s in zip(test_series, model_preds):
                    if h < len(t_s) and h < len(p_s):
                        y_t_h.append(t_s[h])
                        y_p_h.append(p_s[h])

                if not y_t_h:
                    continue

                y_t_h_arr, y_p_h_arr = np.array(y_t_h), np.array(y_p_h)
                for m in metrics:
                    val = 0.0
                    if m == "smape":
                        val = smape(y_t_h_arr, y_p_h_arr)
                    elif m == "mae":
                        val = float(mean_absolute_error(y_t_h_arr, y_p_h_arr))
                    elif m == "rmse":
                        val = float(np.sqrt(mean_squared_error(y_t_h_arr, y_p_h_arr)))

                    horizon_records.append({
                        "model": model_name, "horizon": h + 1, "metric": m, "value": val
                    })

    return pd.DataFrame(results_records), pd.DataFrame(horizon_records)


def analyze_scaling_impact(
        results_all_scaling: Dict[Any, pd.DataFrame],
        scaling_methods: List[Any],
        baseline_models: List[str],
        global_models: List[str]
) -> pd.DataFrame:
    all_models = baseline_models + global_models
    base_key = None if None in results_all_scaling else "none"

    if base_key not in results_all_scaling:
        return pd.DataFrame()

    df_base = results_all_scaling[base_key].set_index("model")["smape"].to_dict()
    impact_records = []

    for method in scaling_methods:
        if method not in results_all_scaling:
            continue

        df_current = results_all_scaling[method].set_index("model")["smape"].to_dict()
        method_name = "none" if method is None else str(method)

        for model in all_models:
            base_val = df_base.get(model)
            curr_val = df_current.get(model)

            if base_val is not None and curr_val is not None:
                improvement = ((base_val - curr_val) / base_val) * 100.0 if base_val > 0 else 0.0
                impact_records.append({
                    "model": model,
                    "scaling": method_name,
                    "smape": curr_val,
                    "improvement": improvement
                })

    return pd.DataFrame(impact_records)