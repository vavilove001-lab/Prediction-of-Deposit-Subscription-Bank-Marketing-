import json
import os
from datetime import datetime
from typing import Any, Optional, List, Dict

import pandas as pd
import numpy as np
import sklearn

# Вычисляем корень проекта: поднимаемся из src на один уровень вверх
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE_DIR = os.path.join(ROOT_DIR, "experiments")


def log_experiment(
    name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    features: List[str],
    random_state: int = 42,
    base_dir: Optional[str] = None,
    artifact_path: Optional[str] = None,
    baseline_roc_auc: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Логирует эксперимент. По умолчанию сохраняет в <корень>/experiments.
    base_dir можно переопределить, если нужно писать в другое место.
    """
    output_dir = base_dir if base_dir is not None else DEFAULT_BASE_DIR
    os.makedirs(output_dir, exist_ok=True)

    log_data = {
        "experiment_name": name,
        "timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "params": params,
        "metrics": metrics,
        "features": features,
        "baseline_roc_auc": baseline_roc_auc,
        "artifact_path": artifact_path,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    if baseline_roc_auc is not None and "roc_auc" in metrics:
        log_data["warning_needed"] = metrics["roc_auc"] < baseline_roc_auc
    else:
        log_data["warning_needed"] = ("roc_auc" in metrics and metrics["roc_auc"] < 0.55)

    # JSON-лог
    ts = log_data["timestamp"].replace(":", "_").replace(".", "_").replace("-", "_")
    json_path = os.path.join(output_dir, f"{ts}_{name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)

    # summary.csv
    summary_path = os.path.join(output_dir, "summary.csv")
    row = {
        "experiment_name": name,
        "timestamp": log_data["timestamp"],
        "model_type": params.get("model_type", "unknown"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "baseline_roc_auc": baseline_roc_auc,
        "num_features": len(features),
        "warning_needed": log_data["warning_needed"],
        "artifact_path": artifact_path or "",
    }
    df_row = pd.DataFrame([row])

    if os.path.exists(summary_path):
        df_summary = pd.concat([pd.read_csv(summary_path), df_row], ignore_index=True)
    else:
        df_summary = df_row

    df_summary.to_csv(summary_path, index=False)
    return log_data


def save_plot(fig, experiment_name: str, plot_name: str, base_dir: Optional[str] = None) -> str:
    """
    Сохраняет matplotlib-фигуру в <корень>/experiments/{experiment_name}/plots/.
    По умолчанию base_dir берётся из ROOT_DIR.
    Возвращает абсолютный путь к файлу.
    """
    output_dir = base_dir if base_dir is not None else DEFAULT_BASE_DIR

    plot_dir = os.path.join(output_dir, experiment_name, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{plot_name}.png"
    filepath = os.path.join(plot_dir, filename)

    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    return filepath