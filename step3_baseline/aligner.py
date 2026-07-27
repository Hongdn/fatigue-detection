"""个体基线对齐 — 每说话人 z-score 标准化，消除个体差异

核心逻辑：
    z = (x - baseline_mean) / baseline_std

    每个说话人有自己的 baseline（均值+标准差）。
    样本不足时用群体均值兜底，标记 low_confidence=True。

使用：
    from baseline import BaselineAligner

    # 一步到位
    aligner = BaselineAligner()
    df_z = aligner.fit_transform(df, speaker_col="speaker_label")

    # 持久化 + 后续复用
    aligner.save("baselines/speaker_01.json")
    aligner = BaselineAligner.load("baselines/speaker_01.json")
    df_new_z = aligner.transform(df_new)
"""

import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class BaselineAligner:
    """个体内 z-score 基线对齐器

    Attributes:
        baselines_: {speaker_id: {"mean": Series, "std": Series, "n": int}}
        global_: {"mean": Series, "std": Series}  # 群体基线（兜底用）
        feature_cols_: 特征列名列表
        min_samples_: 个体最少样本数（低于此值用群体兜底）
    """

    def __init__(self, min_samples: int = 3, epsilon: float = 1e-8):
        """
        Args:
            min_samples: 个体最少样本数（默认3），低于此值用群体基线兜底
            epsilon: 防除零常数
        """
        self.min_samples = min_samples
        self.epsilon = epsilon
        self.baselines_: Dict[str, dict] = {}
        self.global_: Dict[str, pd.Series] = {}
        self.feature_cols_: List[str] = []
        self._fitted = False

    # ━━━━━━ 核心方法 ━━━━━━

    def fit(
        self,
        df: pd.DataFrame,
        speaker_col: str = "speaker_label",
        feature_cols: Optional[List[str]] = None,
    ) -> "BaselineAligner":
        """计算每个说话人的基线（均值+标准差）

        Args:
            df: 特征 DataFrame（含 speaker_col 列 + 88 维特征列）
            speaker_col: 说话人列名
            feature_cols: 使用哪些特征列（None=自动排除元数据列）

        Returns:
            self
        """
        if feature_cols is None:
            feature_cols = _detect_feature_cols(df)
        self.feature_cols_ = feature_cols

        # 全局基线（群体均值兜底）
        self.global_["mean"] = df[feature_cols].mean()
        self.global_["std"] = df[feature_cols].std()

        # 每个说话人计算基线
        for speaker, group in df.groupby(speaker_col):
            n = len(group)
            mean = group[feature_cols].mean()
            std = group[feature_cols].std()
            self.baselines_[speaker] = {
                "mean": mean,
                "std": std,
                "n": n,
                "use_global": n < self.min_samples,
            }

        self._fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        speaker_col: str = "speaker_label",
        add_confidence: bool = True,
    ) -> pd.DataFrame:
        """对每个样本做个体内 z-score 标准化

        Args:
            df: 特征 DataFrame
            speaker_col: 说话人列名
            add_confidence: 是否添加 low_confidence 列

        Returns:
            z-scored DataFrame（特征列被替换为 z 值）

        Raises:
            RuntimeError: 未 fit() 过
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 或 fit_transform()")

        df_z = df.copy()
        feature_cols = self.feature_cols_

        # 逐行 z-score
        z_values = []
        low_conf_flags = []

        for _, row in df.iterrows():
            speaker = row.get(speaker_col, "unknown")
            x = row[feature_cols].values.astype(np.float64)

            # 获取基线
            baseline = self.baselines_.get(speaker)
            if baseline is None or baseline["use_global"]:
                mean = self.global_["mean"].values
                std = self.global_["std"].values
                use_global = True
            else:
                mean = baseline["mean"].values
                std = baseline["std"].values
                use_global = False

            # z = (x - mean) / std
            std_safe = np.where(std < self.epsilon, 1.0, std)
            z = (x - mean) / std_safe

            # 标准差为0的特征 → z=0
            z[std < self.epsilon] = 0.0

            z_values.append(z)
            low_conf_flags.append(use_global)

        # 写入 DataFrame
        df_z[feature_cols] = np.array(z_values)
        if add_confidence:
            df_z["low_confidence"] = low_conf_flags

        return df_z

    def fit_transform(
        self,
        df: pd.DataFrame,
        speaker_col: str = "speaker_label",
    ) -> pd.DataFrame:
        """fit + transform 一步完成"""
        return self.fit(df, speaker_col).transform(df, speaker_col)

    # ━━━━━━ 持久化 ━━━━━━

    def save(self, path: str) -> None:
        """保存基线到 JSON 文件"""
        data = {
            "min_samples": self.min_samples,
            "epsilon": self.epsilon,
            "feature_cols": self.feature_cols_,
            "global": {
                "mean": self.global_["mean"].to_dict(),
                "std": self.global_["std"].to_dict(),
            },
            "baselines": {},
        }
        for speaker, info in self.baselines_.items():
            data["baselines"][speaker] = {
                "mean": info["mean"].to_dict(),
                "std": info["std"].to_dict(),
                "n": info["n"],
                "use_global": info["use_global"],
            }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BaselineAligner":
        """从 JSON 文件加载基线"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        aligner = cls(
            min_samples=data["min_samples"],
            epsilon=data["epsilon"],
        )
        aligner.feature_cols_ = data["feature_cols"]
        aligner.global_["mean"] = pd.Series(data["global"]["mean"])
        aligner.global_["std"] = pd.Series(data["global"]["std"])

        for speaker, info in data["baselines"].items():
            aligner.baselines_[speaker] = {
                "mean": pd.Series(info["mean"]),
                "std": pd.Series(info["std"]),
                "n": info["n"],
                "use_global": info["use_global"],
            }
        aligner._fitted = True
        return aligner

    # ━━━━━━ 信息 ━━━━━━

    def summary(self) -> pd.DataFrame:
        """返回每个说话人的基线摘要"""
        rows = []
        for speaker, info in self.baselines_.items():
            rows.append({
                "speaker": speaker,
                "n_samples": info["n"],
                "using_global": info["use_global"],
                "jitter_mean": info["mean"].get("jitterLocal_sma3nz_amean", np.nan),
                "shimmer_mean": info["mean"].get("shimmerLocaldB_sma3nz_amean", np.nan),
            })
        return pd.DataFrame(rows).set_index("speaker")


# ━━━━━━ 辅助 ━━━━━━

_META_COLS = {"start_sec", "end_sec", "duration_sec", "speaker_label",
              "quality_flag", "low_confidence", "segment_id"}


def _detect_feature_cols(df: pd.DataFrame) -> List[str]:
    """自动检测特征列（排除元数据列）"""
    return [c for c in df.columns if c not in _META_COLS]
