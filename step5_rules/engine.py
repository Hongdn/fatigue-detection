"""step5 规则引擎 — 基于 arousal + exertion 判定工作状态

输入 step4 推理结果（arousal, exertion, fatigue_prob），输出状态标签。
使用 percentile 自适应阈值，不依赖手工调参。

状态定义：
    兴奋: 高唤醒度
    稳定: 适中唤醒度 + 自然说话
    疲劳: 低唤醒度 + 费力/不自然
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class StateResult:
    """单段状态判定结果"""
    segment_id: str
    arousal: float
    exertion: float
    fatigue_prob: float
    state: str           # 兴奋 / 稳定 / 疲劳
    confidence: str      # high / medium / low
    reason: str          # 可解释的判定理由


# ━━━ 阈值计算 ━━━

def _compute_thresholds(values: np.ndarray) -> dict:
    """从数据分布自动计算阈值"""
    return {
        "low": np.percentile(values, 25),
        "high": np.percentile(values, 75),
        "mean": values.mean(),
    }


# ━━━ 状态判定 ━━━

def classify_state(
    arousal: float,
    exertion: float,
    arousal_thresholds: Optional[dict] = None,
    exertion_thresholds: Optional[dict] = None,
) -> tuple[str, str, str]:
    """单点状态判定

    Args:
        arousal: 唤醒度 ∈ [0, 1]
        exertion: 费力程度 ∈ [0, 1]
        arousal_thresholds: {"low": ..., "high": ...}
        exertion_thresholds: {"low": ..., "high": ...}

    Returns:
        (state, confidence, reason)
    """
    at = arousal_thresholds or {"low": 0.55, "high": 0.80}
    et = exertion_thresholds or {"low": 0.85, "high": 0.95}

    reasons = []

    # 兴奋: 高 arousal
    if arousal > at["high"]:
        reasons.append(f"arousal={arousal:.2f} > 阈值({at['high']:.2f}) → 高唤醒")
        return "兴奋", "high", "; ".join(reasons)

    # 疲劳: 低 arousal + 高 exertion（费力）
    if arousal < at["low"] and exertion > et["high"]:
        reasons.append(f"arousal={arousal:.2f} < 阈值({at['low']:.2f}) → 低唤醒")
        reasons.append(f"exertion={exertion:.2f} > 阈值({et['high']:.2f}) → 高费力")
        return "疲劳", "high", "; ".join(reasons)

    # 疲劳趋势: 低 arousal（即使 exertion 不高）
    if arousal < at["low"]:
        reasons.append(f"arousal={arousal:.2f} < 阈值({at['low']:.2f}) → 低唤醒")
        reasons.append(f"exertion={exertion:.2f} ≤ 阈值({et['high']:.2f})")
        return "疲劳", "medium", "; ".join(reasons)

    # 稳定
    reasons.append(f"arousal={arousal:.2f} 在 [{at['low']:.2f}, {at['high']:.2f}] 之间")
    return "稳定", "high", "; ".join(reasons)


# ━━━ 批量判定 ━━━

def classify_dataframe(
    df: pd.DataFrame,
    arousal_col: str = "arousal",
    exertion_col: str = "exertion",
    baseline: Optional["SpeakerBaseline"] = None,
    speaker_col: Optional[str] = None,
) -> pd.DataFrame:
    """对 step4 推理结果的 DataFrame 逐行判定状态

    Args:
        df: step4 predict() 输出的 DataFrame
        arousal_col: arousal 列名
        exertion_col: exertion 列名
        baseline: SpeakerBaseline 实例，传入则做个体归一化
        speaker_col: 说话人列名（baseline 传入时必填）

    Returns:
        DataFrame，新增 state / confidence / reason / arousal_z / exertion_z 列
    """
    result = df.copy()

    # 个体基线归一化（如果有）
    if baseline is not None:
        if speaker_col is None or speaker_col not in df.columns:
            raise ValueError("baseline 传入时需要 speaker_col 且列存在于 DataFrame 中")
        a_z, e_z, bl_conf, bl_using_ind = [], [], [], []
        for _, row in df.iterrows():
            sid = str(row[speaker_col])
            br = baseline.normalize(sid, row[arousal_col], row[exertion_col])
            a_z.append(br.arousal_z)
            e_z.append(br.exertion_z)
            bl_conf.append(br.confidence)
            bl_using_ind.append(br.using_individual)
            baseline.update(sid, row[arousal_col], row[exertion_col])
        result["arousal_z"] = a_z
        result["exertion_z"] = e_z
        result["baseline_conf"] = bl_conf
        result["using_individual_baseline"] = bl_using_ind

    # 自动计算阈值（基于当前数据分布）
    at = _compute_thresholds(df[arousal_col].values)
    et = _compute_thresholds(df[exertion_col].values)

    states, confidences, reasons = [], [], []
    for _, row in df.iterrows():
        s, c, r = classify_state(
            row[arousal_col], row[exertion_col],
            arousal_thresholds=at, exertion_thresholds=et,
        )
        states.append(s)
        confidences.append(c)
        reasons.append(r)

    # 如果有个体基线，用基线状态降级分类置信度
    if baseline is not None:
        for i in range(len(result)):
            bl_conf_val = str(result.at[i, "baseline_conf"])
            c = confidences[i]
            if bl_conf_val == "low":
                c = "low" if c == "low" else ("medium" if c == "high" else "low")
            elif bl_conf_val == "medium" and c == "high":
                c = "medium"
            confidences[i] = c

    result["state"] = states
    result["confidence"] = confidences
    result["reason"] = reasons

    return result


# ━━━ SHAP 增强判定 ━━━

def classify_with_shap(
    df: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """在批量判定基础上，附加 SHAP 特征解释

    要求 df 包含 step4 predict(add_shap=True) 输出的 shap_arousal_* 列

    Args:
        df: step4 推理结果（含 SHAP 列）
        top_k: 每条解释包含前几个特征

    Returns:
        DataFrame，额外包含 shap_reason 列
    """
    df = classify_dataframe(df)

    # 找到 shap 列
    shap_cols = [c for c in df.columns if c.startswith("shap_arousal_")]
    if not shap_cols:
        df["shap_reason"] = "SHAP 列未找到"
        return df

    shap_reasons = []
    for idx, row in df.iterrows():
        contributions = {}
        for col in shap_cols:
            feature_name = col.replace("shap_arousal_", "")
            contributions[feature_name] = row[col]

        # top_k 绝对值最大的
        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top = sorted_feats[:top_k]

        parts = []
        for feat, val in top:
            direction = "↑" if val > 0 else "↓"
            parts.append(f"{feat.split('_')[0]}{direction}({val:+.3f})")

        shap_reasons.append(" | ".join(parts))

    df["shap_reason"] = shap_reasons
    return df


# ━━━ 汇总 ━━━

def state_summary(df: pd.DataFrame) -> dict:
    """输出状态分布统计"""
    if "state" not in df.columns:
        df = classify_dataframe(df)

    counts = df["state"].value_counts()
    return {
        "total": len(df),
        "兴奋": int(counts.get("兴奋", 0)),
        "稳定": int(counts.get("稳定", 0)),
        "疲劳": int(counts.get("疲劳", 0)),
        "兴奋_pct": round(100 * counts.get("兴奋", 0) / len(df), 1),
        "疲劳_pct": round(100 * counts.get("疲劳", 0) / len(df), 1),
        "avg_arousal": round(df["arousal"].mean(), 4),
        "avg_exertion": round(df["exertion"].mean(), 4),
    }
