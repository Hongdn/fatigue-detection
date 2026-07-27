"""维度回归模块 — XGBoost + SHAP 双任务回归

连接 step3_baseline 输出的 z-scored 特征，训练/推理 arousal 和 exertion，
通过 SHAP 输出每个 openSMILE 维度的贡献值用于告警溯源。

模块：
    model    : train_models / predict / explain_segment
    dataset  : load_parquet_metadata / load_parquet_with_audio
    pipeline : train_from_step3 / predict_from_step3
"""

from .model import train_models, predict, explain_segment, load_models
from .pipeline import train_from_step3, predict_from_step3

__all__ = [
    "train_models",
    "predict",
    "explain_segment",
    "load_models",
    "train_from_step3",
    "predict_from_step3",
]
