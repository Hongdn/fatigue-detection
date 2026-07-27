"""维度回归 pipeline — 对接 step3 输出，训练 / 推理 / 解释

使用：
    from step4_regression import train_from_step3, predict_from_step3

    # 方式一：从 step3 DataFrame 训练
    models = train_from_step3(df_z, fatigue_labels)

    # 方式二：从已训练的模型推理
    results = predict_from_step3(df_z_new, model_dir="models/")

    # 方式三：端到端（音频文件 → 疲劳判定）
    results = predict_from_step3("REC-1.WAV", model_dir="models/")
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .model import (
    train_models,
    predict,
    explain_segment,
    load_models,
    _detect_feature_cols,
)


def train_from_step3(
    df: pd.DataFrame,
    y_arousal: np.ndarray,
    y_exertion: Optional[np.ndarray] = None,
    output_dir: str = "step4_regression/models/",
    n_estimators: int = 200,
    max_depth: int = 6,
    verbose: bool = True,
) -> Dict:
    """从 step3 输出的 z-scored DataFrame 训练 XGBoost 双任务回归

    Args:
        df: step3 输出的 z-scored DataFrame
        y_arousal: arousal 标签（如 DeEAR score_arousal）∈ [0, 1]
        y_exertion: exertion 标签（如 DeEAR score_nature）；None 则用代理标签
        output_dir: 模型输出目录
        n_estimators: XGBoost 树数量
        max_depth: 最大树深度
        verbose: 打印训练信息

    Returns:
        {"arousal": XGBRegressor, "exertion": XGBRegressor, "feature_cols": [...]}
    """
    return train_models(
        df=df,
        y_arousal=y_arousal,
        y_exertion=y_exertion,
        output_dir=output_dir,
        n_estimators=n_estimators,
        max_depth=max_depth,
        verbose=verbose,
    )


def predict_from_step3(
    input_data: str | pd.DataFrame,
    model_dir: str = "step4_regression/models/",
    denoise: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """从 step3 输出或原始音频文件推理，输出 arousal + exertion + SHAP

    两种使用方式：
    1. 传入 z-scored DataFrame（已跑完 step1-3）→ 直接推理
    2. 传入音频文件路径 → 内部走 step1→step2→step3→step4 全链路

    Args:
        input_data: z-scored DataFrame 或音频文件路径
        model_dir: 训练好的模型目录
        denoise: 是否降噪（仅输入为音频文件时有效）
        verbose: 打印进度

    Returns:
        DataFrame，列：segment_id, arousal, exertion, fatigue_prob,
                       [+ SHAP 贡献列，如果模型已训练]
    """
    # 加载模型
    if verbose:
        print(f"加载模型: {model_dir}")
    models = load_models(model_dir)
    feature_cols = models["feature_cols"]

    # 获取 DataFrame
    if isinstance(input_data, str):
        # 音频文件 → 走全链路
        from step3_baseline.pipeline import align_from_pipeline

        if verbose:
            print(f"=== 全链路推理: {input_data} ===")

        df_z, _ = align_from_pipeline(
            input_data,
            denoise=denoise,
            verbose=verbose,
        )
    elif isinstance(input_data, pd.DataFrame):
        df_z = input_data
    else:
        raise TypeError(f"input_data 须为文件路径或 DataFrame，收到 {type(input_data)}")

    # 确保特征列对齐
    missing = [c for c in feature_cols if c not in df_z.columns]
    if missing:
        raise KeyError(
            f"输入 DataFrame 缺少以下特征列: {missing}\n"
            f"请确认已通过 step2 (openSMILE eGeMAPSv02) 提取特征"
        )

    # 推理
    if verbose:
        print(f"  XGBoost 推理中 ({len(df_z)} 段) ...")
    results = predict(models, df_z, add_shap=True)

    if verbose:
        n_fatigue = (results["fatigue_prob"] > 0.5).sum()
        print(f"  → {n_fatigue}/{len(results)} 段被判疲劳倾向")

    return results


def cross_validate(
    df: pd.DataFrame,
    fatigue_labels: np.ndarray,
    speaker_col: str = "speaker_label",
    n_folds: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """Leave-one-speaker-out 交叉验证

    对每位管制员留出测试，其余人训练，评估泛化性能。

    Args:
        df: step3 输出的 z-scored DataFrame
        fatigue_labels: 疲劳标签
        speaker_col: 说话人列名
        n_folds: fold 数
        verbose: 打印进度

    Returns:
        DataFrame，每行一个 fold 的评估指标
    """
    speakers = df[speaker_col].unique()
    if n_folds and len(speakers) > n_folds:
        speakers = speakers[:n_folds]

    results = []
    for held_out in speakers:
        train_mask = df[speaker_col] != held_out
        test_mask = df[speaker_col] == held_out

        df_train = df[train_mask]
        df_test = df[test_mask]
        y_train = fatigue_labels[train_mask.values]
        y_test = fatigue_labels[test_mask.values]

        if verbose:
            print(f"  Fold: 留出 {held_out}（训练 {len(df_train)}，测试 {len(df_test)}）")

        models = train_models(df_train, y_train, output_dir="/tmp/xgb_cv/", verbose=False)
        preds = predict(models, df_test, add_shap=False)

        from .model import ccc_score
        ccc = ccc_score(1.0 - y_test.astype(np.float32), preds["arousal"].values)
        acc = ((preds["fatigue_prob"] > 0.5) == (y_test > 0.5)).mean()

        results.append({
            "held_out": held_out,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "ccc_arousal": round(ccc, 4),
            "accuracy": round(float(acc), 4),
        })

    return pd.DataFrame(results)
