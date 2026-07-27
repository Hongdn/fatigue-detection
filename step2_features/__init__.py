"""特征提取模块 — openSMILE eGeMAPSv02 88 维生理声学特征

模块：
    extractor   : FeatureExtractor — openSMILE 单例提取器
    feature_set : 特征分组（疲劳核心指标 + 物理含义分组）
    pipeline    : 对接 preprocess 输出，端到端提取
"""

from .extractor import FeatureExtractor, get_extractor
from .pipeline import extract_from_pipeline, extract_from_segments, extract_batch

__all__ = [
    "FeatureExtractor",
    "get_extractor",
    "extract_from_pipeline",
    "extract_from_segments",
    "extract_batch",
]
