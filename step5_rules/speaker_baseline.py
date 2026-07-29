"""个体基线管理 — 无监督自学习，无需注册阶段

每个说话人的基线随时间推移自动建立：
    < 30 段  → 完全用群体基线（冷启动）
    30-100 段 → 群体+个体加权混合过渡
    > 100 段 → 完全个体基线

使用：
    baseline = SpeakerBaseline()
    result = baseline.normalize("speaker_A", arousal=0.72, exertion=0.91)
    # → {"arousal_z": -0.3, "exertion_z": 0.1, "confidence": "high", ...}
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import numpy as np


@dataclass
class BaselineResult:
    """归一化后的单段结果"""
    arousal_z: float          # 归一化 arousal（个体内 z-score）
    exertion_z: float         # 归一化 exertion
    arousal_raw: float        # 原始值
    exertion_raw: float       # 原始值
    using_individual: bool    # 是否使用了个体基线
    confidence: str           # high / medium / low
    speaker_n: int            # 该说话人已积累的段数


class SpeakerBaseline:
    """个体基线管理器

    维护每个说话人的运行均值/标准差，段数不够时回退群体基线。
    支持持久化到 JSON 文件。

    Args:
        global_mean: 群体 arousal 均值
        global_std: 群体 arousal 标准差
        cold_n: 冷启动阈值（< 此数完全用群体基线）
        warm_n: 完全个体基线阈值（> 此数完全用个体基线）
    """

    def __init__(
        self,
        global_mean: float = 0.72,
        global_std: float = 0.19,
        cold_n: int = 30,
        warm_n: int = 100,
    ):
        self.global_mean = global_mean
        self.global_std = global_std
        self.cold_n = cold_n
        self.warm_n = warm_n

        # 每个说话人的运行统计
        self._speakers: Dict[str, dict] = {}

    # ━━━ 核心方法 ━━━

    def normalize(
        self,
        speaker_id: str,
        arousal: float,
        exertion: float,
    ) -> BaselineResult:
        """对单个预测值做个体归一化

        Args:
            speaker_id: 说话人标识（如"controller_001"）
            arousal: XGBoost 预测的原始 arousal ∈ [0,1]
            exertion: XGBoost 预测的原始 exertion ∈ [0,1]

        Returns:
            BaselineResult，含归一化后的 z 值
        """
        spk = self._get_or_create(speaker_id)
        n = spk["n"]

        # 确定使用个体 vs 群体基线的权重
        if n < self.cold_n:
            # 冷启动：完全群体基线
            mu_a, sigma_a = self.global_mean, self.global_std
            using_individual = False
            confidence = "low"
        elif n < self.warm_n:
            # 过渡期：个体+群体加权混合
            alpha = (n - self.cold_n) / (self.warm_n - self.cold_n)
            mu_a = alpha * spk["sum_a"] / n + (1 - alpha) * self.global_mean
            # std 的混合（取 max 避免早期方差偏小）
            ind_std = np.sqrt(max(spk["sum_sq_a"] / n - (spk["sum_a"] / n) ** 2, 1e-8))
            sigma_a = alpha * ind_std + (1 - alpha) * self.global_std
            using_individual = True
            confidence = "medium"
        else:
            # 个体基线稳定
            mu_a = spk["sum_a"] / n
            ind_std = np.sqrt(max(spk["sum_sq_a"] / n - mu_a ** 2, 1e-8))
            sigma_a = max(ind_std, 0.05)  # 防止 std 过小导致 z 值爆炸
            using_individual = True
            confidence = "high"

        # 计算 z-score
        arousal_z = (arousal - mu_a) / sigma_a if sigma_a > 1e-8 else 0.0
        # exertion 目前用同样的规则（exertion 方差小，群体 std 需要单独设置）
        exertion_z = (exertion - 0.95) / 0.03 if exertion < 0.95 else 0.0

        return BaselineResult(
            arousal_z=round(float(arousal_z), 4),
            exertion_z=round(float(exertion_z), 4),
            arousal_raw=arousal,
            exertion_raw=exertion,
            using_individual=using_individual,
            confidence=confidence,
            speaker_n=n,
        )

    def update(
        self,
        speaker_id: str,
        arousal: float,
        exertion: float,
    ):
        """用新数据更新说话人的运行统计

        Args:
            speaker_id: 说话人标识
            arousal: 原始预测值
            exertion: 原始预测值
        """
        spk = self._get_or_create(speaker_id)
        spk["sum_a"] += arousal
        spk["sum_sq_a"] += arousal ** 2
        spk["n"] += 1

    def batch_process(
        self,
        speaker_id: str,
        arousal_values: np.ndarray,
        exertion_values: np.ndarray,
        update: bool = True,
    ) -> list:
        """批量处理同一说话人的多段预测

        先计算归一化，再用这批数据更新基线。

        Args:
            speaker_id: 说话人标识
            arousal_values: shape (n,) 的 arousal 数组
            exertion_values: shape (n,) 的 exertion 数组
            update: 是否用这批数据更新基线

        Returns:
            [BaselineResult, ...] 列表
        """
        results = []
        for a, e in zip(arousal_values, exertion_values):
            result = self.normalize(speaker_id, a, e)
            results.append(result)
            if update:
                self.update(speaker_id, a, e)
        return results

    # ━━━ 持久化 ━━━

    def save(self, path: str):
        """保存当前基线到 JSON 文件"""
        data = {
            "global": {"mean": self.global_mean, "std": self.global_std},
            "config": {"cold_n": self.cold_n, "warm_n": self.warm_n},
            "speakers": self._speakers,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """从 JSON 文件恢复基线, 文件不存在则初始化空状态"""
        if not os.path.exists(path):
            self._speakers = {}
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.global_mean = data.get("global", {}).get("mean", self.global_mean)
            self.global_std = data.get("global", {}).get("std", self.global_std)
            self.cold_n = data.get("config", {}).get("cold_n", self.cold_n)
            self.warm_n = data.get("config", {}).get("warm_n", self.warm_n)
            self._speakers = data.get("speakers", {})
        except (json.JSONDecodeError, KeyError) as e:
            import shutil
            bak = path + ".bak"
            if os.path.exists(path):
                shutil.copy2(path, bak)
                print(f"[SpeakerBaseline] JSON 损坏, 已备份到 {bak}, 重建空状态")
            self._speakers = {}
        self.global_mean = data["global"]["mean"]
        self.global_std = data["global"]["std"]
        self.cold_n = data["config"]["cold_n"]
        self.warm_n = data["config"]["warm_n"]
        self._speakers = data["speakers"]

    # ━━━ 查询 ━━━

    def summary(self) -> list:
        """所有说话人基线状态"""
        rows = []
        for sid, spk in self._speakers.items():
            n = spk["n"]
            mu = spk["sum_a"] / n if n > 0 else 0
            var = spk["sum_sq_a"] / n - mu ** 2 if n > 0 else 0
            std = np.sqrt(max(var, 0))
            if n < self.cold_n:
                stage = "cold"
            elif n < self.warm_n:
                stage = "warm"
            else:
                stage = "stable"
            rows.append({
                "speaker_id": sid,
                "n": n,
                "mean": round(float(mu), 4),
                "std": round(float(std), 4),
                "stage": stage,
            })
        return sorted(rows, key=lambda r: r["n"], reverse=True)

    # ━━━ 内部 ━━━

    def _get_or_create(self, speaker_id: str) -> dict:
        if speaker_id not in self._speakers:
            self._speakers[speaker_id] = {
                "n": 0,
                "sum_a": 0.0,
                "sum_sq_a": 0.0,
            }
        return self._speakers[speaker_id]
