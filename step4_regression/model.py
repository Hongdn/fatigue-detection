"""XGBoost 双任务回归 + SHAP 可解释 — 训练、推理、解释

用途：
    from step4_regression import train_models, predict, explain_segment

    # 训练
    models = train_models(df_z, fatigue_labels, output_dir="models/")

    # 推理
    results = predict(models, df_z_new)

    # 解释
    explanation = explain_segment(models["arousal"], df_z_new.iloc[0])
"""

import json
import os
import warnings
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ━━━ 延迟导入（避免未安装时报错） ━━━

def _check_xgboost():
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def _check_shap():
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


# ━━━ 数据类 ━━━

@dataclass
class RegressionResult:
    """单段语音的回归结果"""
    segment_id: str
    arousal: float          # ∈ [0, 1]
    exertion: float         # ∈ [0, 1]
    fatigue_prob: float = 0.0   # XGBoost 原生输出
    shap_arousal: Optional[Dict[str, float]] = None   # 特征名 → SHAP 贡献值
    shap_exertion: Optional[Dict[str, float]] = None


# ━━━ 工具函数 ━━━

def _detect_feature_cols(df: pd.DataFrame) -> list:
    """从 step3 输出的 DataFrame 中自动识别 88 维特征列"""
    META = {"start_sec", "end_sec", "duration_sec", "speaker_label",
            "quality_flag", "low_confidence", "segment_id"}
    return [c for c in df.columns if c not in META]


def _exertion_proxy(df: pd.DataFrame) -> np.ndarray:
    """构造 exertion 代理标签（jitter/shimmer/HNR z-score 加权→Sigmoid）

    注意：输入 df 是 step3 输出的 z-scored DataFrame（特征列已是 z 值）。
    这里直接取 z 值加权，无需再做一次 z-score。

    Args:
        df: step3 输出的 z-scored DataFrame，特征名需包含 jitterLocal / shimmerLocaldB / HNRdBACF

    Returns:
        np.ndarray, shape (n,), exertion proxy ∈ [0, 1]
    """
    feature_cols = _detect_feature_cols(df)

    def _find_col(containing: str) -> str:
        matches = [c for c in feature_cols if containing in c]
        if not matches:
            raise KeyError(f"找不到包含 '{containing}' 的特征列，请确认 openSMILE 版本为 eGeMAPSv02")
        return matches[0]

    jitter_col = _find_col("jitterLocal")
    shimmer_col = _find_col("shimmerLocaldB")
    hnr_col = _find_col("HNRdBACF")

    z_jitter = df[jitter_col].values
    z_shimmer = df[shimmer_col].values
    z_hnr = -df[hnr_col].values  # HNR 越低 → exertion 越高

    raw = 0.4 * z_jitter + 0.4 * z_shimmer + 0.2 * z_hnr
    return 1.0 / (1.0 + np.exp(-raw))  # Sigmoid → [0, 1]


# ━━━ CCC 评估 ━━━

def ccc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """一致性相关系数（Concordance Correlation Coefficient）

    SER 回归标准指标，同时衡量相关性和偏差。

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        CCC ∈ [-1, 1]，越接近 1 越好
    """
    mu_t, mu_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(ddof=0), y_pred.var(ddof=0)
    cov = ((y_true - mu_t) * (y_pred - mu_p)).mean()
    denominator = var_t + var_p + (mu_t - mu_p) ** 2
    if denominator < 1e-12:
        return 0.0
    return float(2 * cov / denominator)


# ━━━ 训练 ━━━

def train_models(
    df: pd.DataFrame,
    y_arousal: np.ndarray,
    y_exertion: Optional[np.ndarray] = None,
    output_dir: str = "step4_regression/models/",
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """训练 XGBoost arousal 和 exertion 双任务回归模型

    输入 openSMILE 88 维 z-scored 特征，训练两个独立的 XGBoost 回归器：
    - arousal：用 DeEAR score_arousal 作为目标
    - exertion：用 DeEAR score_nature 作为目标（v2.2；如未提供则用嗓音质量代理标签）

    Args:
        df: step3 输出的 z-scored DataFrame（特征列已是 z 值）
        y_arousal: arousal 标签，如 DeEAR score_arousal ∈ [0, 1]
        y_exertion: exertion 标签，如 DeEAR score_nature ∈ [0, 1]；None 则用代理标签
        output_dir: 模型输出目录
        n_estimators: 树的数量
        max_depth: 最大树深度
        learning_rate: 学习率
        subsample: 每棵树的样本采样比例
        seed: 随机种子
        verbose: 打印训练信息

    Returns:
        {"arousal": XGBRegressor, "exertion": XGBRegressor, "feature_cols": [...]}

    Raises:
        ImportError: xgboost 未安装
    """
    if not _check_xgboost():
        raise ImportError("xgboost 未安装。请执行: pip install xgboost")

    from xgboost import XGBRegressor

    feature_cols = _detect_feature_cols(df)
    X = df[feature_cols].values.astype(np.float32)

    # exertion 标签：优先用传入的，否则用嗓音质量代理
    if y_exertion is None:
        y_exertion = _exertion_proxy(df).astype(np.float32)
        if verbose:
            print("  exertion 标签未提供，使用嗓音质量代理标签（jitter/shimmer/HNR）")
    else:
        y_exertion = y_exertion.astype(np.float32)

    if verbose:
        print(f"训练 XGBoost 双任务回归模型")
        print(f"  样本数: {len(X)}, 特征维度: {len(feature_cols)}")

    # 训练 arousal 模型
    if verbose:
        print("  [1/2] 训练 arousal 模型 ...")
    model_arousal = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        objective="reg:squarederror",
        random_state=seed,
        verbosity=0,
    )
    model_arousal.fit(X, y_arousal)

    # 训练 exertion 模型
    if verbose:
        print("  [2/2] 训练 exertion 模型 ...")
    model_exertion = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        objective="reg:squarederror",
        random_state=seed,
        verbosity=0,
    )
    model_exertion.fit(X, y_exertion)

    # 评估
    pred_a = model_arousal.predict(X)
    pred_e = model_exertion.predict(X)
    ccc_a = ccc_score(y_arousal, pred_a)
    ccc_e = ccc_score(y_exertion, pred_e)

    if verbose:
        print(f"  → CCC_arousal = {ccc_a:.4f}, CCC_exertion = {ccc_e:.4f}")
        print(f"  → arousal 预测范围: [{pred_a.min():.2f}, {pred_a.max():.2f}]")
        print(f"  → exertion 预测范围: [{pred_e.min():.2f}, {pred_e.max():.2f}]")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    model_arousal.save_model(os.path.join(output_dir, "model_arousal.json"))
    model_exertion.save_model(os.path.join(output_dir, "model_exertion.json"))

    # 保存特征列名（推理时需要对齐列顺序）
    with open(os.path.join(output_dir, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"  → 模型已保存至 {output_dir}")

    return {
        "arousal": model_arousal,
        "exertion": model_exertion,
        "feature_cols": feature_cols,
    }


# ━━━ 加载 ━━━

def load_models(model_dir: str = "step4_regression/models/") -> Dict:
    """加载已训练的模型

    Args:
        model_dir: 模型文件目录

    Returns:
        {"arousal": XGBRegressor, "exertion": XGBRegressor, "feature_cols": [...]}

    Raises:
        ImportError: xgboost 未安装
        FileNotFoundError: 模型文件不存在
    """
    if not _check_xgboost():
        raise ImportError("xgboost 未安装。请执行: pip install xgboost")

    from xgboost import XGBRegressor

    arousal_path = os.path.join(model_dir, "model_arousal.json")
    exertion_path = os.path.join(model_dir, "model_exertion.json")
    cols_path = os.path.join(model_dir, "feature_cols.json")

    model_arousal = XGBRegressor()
    model_arousal.load_model(arousal_path)

    model_exertion = XGBRegressor()
    model_exertion.load_model(exertion_path)

    with open(cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    return {
        "arousal": model_arousal,
        "exertion": model_exertion,
        "feature_cols": feature_cols,
    }


# ━━━ 推理 ━━━

def predict(
    models: Dict,
    df: pd.DataFrame,
    add_shap: bool = True,
) -> pd.DataFrame:
    """对 step3 输出的 z-scored DataFrame 逐段推理

    Args:
        models: train_models() 或 load_models() 返回的字典
        df: step3 输出的 z-scored DataFrame
        add_shap: 是否附加 SHAP 贡献值

    Returns:
        DataFrame，列：segment_id, arousal, exertion, fatigue_prob,
                        [+ shap_arousal_jitter, shap_arousal_shimmer, ...]
    """
    feature_cols = models["feature_cols"]
    X = df[feature_cols].values.astype(np.float32)

    pred_a = models["arousal"].predict(X)
    pred_e = models["exertion"].predict(X)

    # arousal → fatigue prob（逆映射）
    fatigue_prob = 1.0 - pred_a
    fatigue_prob = np.clip(fatigue_prob, 0.0, 1.0)

    result = pd.DataFrame({
        "segment_id": df.index,
        "arousal": np.clip(pred_a, 0.0, 1.0),
        "exertion": np.clip(pred_e, 0.0, 1.0),
        "fatigue_prob": fatigue_prob,
    })

    if "start_sec" in df.columns:
        result["start_sec"] = df["start_sec"].values
    if "end_sec" in df.columns:
        result["end_sec"] = df["end_sec"].values

    # SHAP 贡献值（可选，初始训练后验证用）
    if add_shap and _check_shap():
        try:
            import shap
            explainer_a = shap.TreeExplainer(models["arousal"])
            shap_vals_a = explainer_a.shap_values(X)
            for i, col in enumerate(feature_cols):
                result[f"shap_arousal_{col}"] = shap_vals_a[:, i]
            explainer_e = shap.TreeExplainer(models["exertion"])
            shap_vals_e = explainer_e.shap_values(X)
            for i, col in enumerate(feature_cols):
                result[f"shap_exertion_{col}"] = shap_vals_e[:, i]
        except Exception:
            pass  # SHAP 可选，失败不影响推理

    return result


# ━━━ 解释 ━━━

def explain_segment(
    model,
    feature_row: pd.Series,
    feature_cols: list,
    top_k: int = 5,
) -> RegressionResult:
    """对单个语音段的预测给出 SHAP 解释

    Args:
        model: XGBRegressor（arousal 或 exertion 模型）
        feature_row: 单段 88 维特征
        feature_cols: 特征列名列表（顺序需与训练一致）
        top_k: 返回贡献绝对值最大的前 k 个特征

    Returns:
        RegressionResult，包含预测值和 SHAP 贡献拆解

    Raises:
        ImportError: shap 未安装
    """
    if not _check_shap():
        raise ImportError("shap 未安装。请执行: pip install shap")

    import shap

    X = feature_row[feature_cols].values.astype(np.float32).reshape(1, -1)
    pred = float(model.predict(X)[0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]

    # 取 top-k 贡献绝对值最大的特征
    contributions = {
        feature_cols[i]: float(shap_values[i])
        for i in np.argsort(np.abs(shap_values))[::-1][:top_k]
    }

    return RegressionResult(
        segment_id=str(feature_row.name) if hasattr(feature_row, "name") else "seg",
        arousal=pred,
        exertion=0.0,
        fatigue_prob=1.0 - pred,
        shap_arousal=contributions,
    )


# ━━━ 全局模型缓存 ━━━

_models_cache: Optional[Dict] = None


def get_models(model_dir: str = "step4_regression/models/") -> Dict:
    """获取全局模型实例（单例，避免重复加载）"""
    global _models_cache
    if _models_cache is None:
        _models_cache = load_models(model_dir)
    return _models_cache
