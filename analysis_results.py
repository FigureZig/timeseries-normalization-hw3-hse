import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('fontTools').setLevel(logging.WARNING)

import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def get_latest_run_dir(results_dir: Path) -> Path:
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        sys.exit(1)
    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def plot_showcase_grid(raw_data: dict, plots_dir: Path) -> None:
    scaler = "standard"
    model = "patchtst"

    if scaler not in raw_data["predictions"] or model not in raw_data["predictions"][scaler]:
        return

    train_series = raw_data["train_series"]
    test_series = raw_data["test_series"]
    preds = raw_data["predictions"][scaler][model]

    smapes = []
    for t, p in zip(test_series, preds):
        denom = (np.abs(t) + np.abs(p)) / 2.0
        denom = np.where(denom == 0, 1e-10, denom)
        smapes.append(np.mean(np.abs(t - p) / denom) * 100)

    smapes = np.array(smapes)
    sorted_indices = np.argsort(smapes)

    if len(sorted_indices) < 6:
        selected_indices = sorted_indices
    else:
        best_2 = sorted_indices[:2]
        worst_2 = sorted_indices[-2:]
        mid_idx = len(sorted_indices) // 2
        avg_2 = sorted_indices[mid_idx - 1:mid_idx + 1]
        selected_indices = np.concatenate([best_2, avg_2, worst_2])

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ax in zip(selected_indices, axes):
        train_len = len(train_series[idx])
        test_len = len(test_series[idx])
        plot_train_len = min(train_len, test_len * 4)
        train_plot = train_series[idx][-plot_train_len:]

        x_train = np.arange(plot_train_len)
        x_test = np.arange(plot_train_len, plot_train_len + test_len)

        ax.plot(x_train, train_plot, color="#3498db", label="Train", linewidth=1.5, alpha=0.8)
        ax.plot(x_test, test_series[idx], color="#2ecc71", label="Actual", linewidth=2)
        ax.plot(x_test, preds[idx], color="#e74c3c", linestyle="--", label="Predicted", linewidth=2)

        ax.set_title(f"Ряд {idx} (SMAPE: {smapes[idx]:.2f}%)", pad=10, fontweight="bold")
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(plots_dir / "forecast_showcase.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(plots_dir / "forecast_showcase.png", format="png", bbox_inches="tight")
    plt.close()


def plot_training_history(raw_data: dict, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    colors = {"none": "#95a5a6", "standard": "#3498db", "robust": "#2ecc71", "quantile": "#9b59b6"}

    for scaler, meta in raw_data["metadata"].items():
        if "patchtst_train_losses" in meta and meta["patchtst_train_losses"]:
            losses = meta["patchtst_train_losses"]
            ax.plot(losses, label=f"PatchTST ({scaler})", color=colors.get(scaler, "#333333"), linewidth=2)
            plotted = True

    if not plotted:
        return

    ax.set_title("Кривые обучения PatchTST (MSE Loss vs Epochs)", pad=15, fontweight="bold")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE Loss")
    ax.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "training_history.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(plots_dir / "training_history.png", format="png", bbox_inches="tight")
    plt.close()


def plot_horizon_degradation(horizon_df: pd.DataFrame, plots_dir: Path) -> None:
    df_smape = horizon_df[horizon_df["metric"] == "smape"]
    global_models = ["catboost", "patchtst"]
    df_global = df_smape[df_smape["model"].isin(global_models)]

    if df_global.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df_global,
        x="horizon",
        y="value",
        hue="model",
        style="scaling",
        markers=False,
        dashes=True,
        ax=ax,
        palette=["#e74c3c", "#34495e"]
    )

    ax.set_title("Деградация качества прогноза по горизонту (SMAPE)", pad=15, fontweight="bold")
    ax.set_xlabel("Шаг прогноза")
    ax.set_ylabel("SMAPE (%)")

    plt.tight_layout()
    plt.savefig(plots_dir / "horizon_degradation.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(plots_dir / "horizon_degradation.png", format="png", bbox_inches="tight")
    plt.close()


def plot_scaling_impact(df: pd.DataFrame, plots_dir: Path) -> None:
    global_models = ["catboost", "patchtst"]
    df_global = df[df["model"].isin(global_models)].copy()
    if df_global.empty: return

    base_df = df_global[df_global["scaling"] == "none"].set_index("model")["smape"]
    plot_data = []

    for model in global_models:
        if model not in base_df.index: continue
        base_val = base_df.loc[model]
        for scaler in ["standard", "robust", "quantile"]:
            val_series = df_global[(df_global["model"] == model) & (df_global["scaling"] == scaler)]["smape"]
            if not val_series.empty:
                improvement = ((base_val - val_series.values[0]) / base_val) * 100
                plot_data.append({
                    "Model": "CatBoost" if model == "catboost" else "PatchTST",
                    "Scaler": scaler.capitalize(),
                    "Improvement (%)": improvement
                })

    if not plot_data: return
    impact_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=impact_df, x="Model", y="Improvement (%)", hue="Scaler",
                palette=["#2c3e50", "#7f8c8d", "#bdc3c7"], edgecolor="black", linewidth=1, ax=ax)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_title("Влияние нормализации на качество прогноза", pad=15, fontweight="bold")
    ax.set_ylabel("Улучшение SMAPE (%) $\\rightarrow$ Выше лучше")
    ax.set_xlabel("")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, size=10)

    plt.tight_layout()
    plt.savefig(plots_dir / "scaling_impact.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(plots_dir / "scaling_impact.png", format="png", bbox_inches="tight")
    plt.close()


def plot_model_comparison(df: pd.DataFrame, plots_dir: Path) -> None:
    if "model" not in df.columns or "smape" not in df.columns: return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="model", y="smape", color="#ecf0f1", boxprops={"edgecolor": "black"},
                medianprops={"color": "#c0392b", "linewidth": 1.5}, whiskerprops={"color": "black"},
                capprops={"color": "black"}, ax=ax)
    sns.stripplot(data=df, x="model", y="smape", color="#2c3e50", alpha=0.6, size=6, ax=ax)

    ax.set_title("Сравнение распределения метрики SMAPE по моделям", pad=15, fontweight="bold")
    ax.set_ylabel("SMAPE (%) $\\rightarrow$ Ниже лучше")
    ax.set_xlabel("")

    # Исправление warning'а с тиками
    ax.set_xticks(range(len(df["model"].unique())))
    ax.set_xticklabels([m.capitalize() for m in df["model"].unique()], rotation=45)

    plt.tight_layout()
    plt.savefig(plots_dir / "models_comparison.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(plots_dir / "models_comparison.png", format="png", bbox_inches="tight")
    plt.close()


def main() -> None:
    results_dir = Path("results")
    if not results_dir.exists():
        sys.exit(1)

    latest_run = get_latest_run_dir(results_dir)
    data_dir = latest_run / "data"
    plots_dir = latest_run / "plots"

    try:
        results_df = pd.read_csv(list(data_dir.glob("results_*.csv"))[0])
        horizon_df = pd.read_csv(list(data_dir.glob("per_horizon_*.csv"))[0])
        with open(data_dir / "raw_data.pkl", "rb") as f:
            raw_data = pickle.load(f)
    except Exception:
        sys.exit(1)

    plot_showcase_grid(raw_data, plots_dir)
    plot_training_history(raw_data, plots_dir)
    plot_horizon_degradation(horizon_df, plots_dir)
    plot_scaling_impact(results_df, plots_dir)
    plot_model_comparison(results_df, plots_dir)


if __name__ == "__main__":
    main()