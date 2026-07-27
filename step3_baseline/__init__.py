"""基线对齐模块 — 个体内 z-score 标准化

模块：
    aligner  : BaselineAligner — 个体基线计算 + z-score 对齐 + 持久化
    pipeline : align_features / align_from_pipeline — 接 features 输出
"""

from .aligner import BaselineAligner
from .pipeline import align_features, align_from_pipeline

__all__ = [
    "BaselineAligner",
    "align_features",
    "align_from_pipeline",
]
