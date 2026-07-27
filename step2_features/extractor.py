"""openSMILE 特征提取器 — eGeMAPSv02 88 维生理声学特征

支持 numpy 数组输入（接 preprocess AudioSegment）和文件路径输入。

使用：
    from features.extractor import FeatureExtractor

    extractor = FeatureExtractor()
    features = extractor.extract(audio, sr=16000)     # numpy → Series
    df = extractor.extract_from_segments(segments)     # AudioSegment[] → DataFrame
"""

import warnings
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import opensmile


class FeatureExtractor:
    """openSMILE eGeMAPSv02 特征提取器（单例模式）"""

    def __init__(self):
        self._smile = None

    @property
    def smile(self):
        if self._smile is None:
            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        return self._smile

    @property
    def feature_names(self) -> List[str]:
        """88 维特征名列表"""
        # 跑一次空音频获取列名（缓存用）
        if not hasattr(self, "_feature_names"):
            dummy = np.zeros(16000, dtype=np.float32)
            df = self.extract(dummy, sr=16000)
            self._feature_names = list(df.index)
        return self._feature_names

    def extract(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> pd.Series:
        """单段音频 → 88 维特征 Series

        Args:
            audio: 音频数据，float32，单声道
            sr: 采样率（默认 16000）

        Returns:
            pd.Series，索引为 88 维特征名
        """
        # 确保 float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 确保单声道
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        # 极短音频处理
        if len(audio) < sr * 0.05:  # < 50ms
            return pd.Series(np.full(self._n_features(), np.nan), index=self.feature_names)

        # openSMILE process_signal 接受 (n,) 或 (n, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = self.smile.process_signal(audio, sr)
        return df.iloc[0]

    def extract_from_file(self, filepath: str) -> pd.Series:
        """从音频文件提取

        Args:
            filepath: WAV 文件路径

        Returns:
            pd.Series，88 维特征
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = self.smile.process_file(filepath)
        return df.iloc[0]

    def extract_from_segments(
        self,
        segments,  # List[AudioSegment]
        add_metadata: bool = True,
    ) -> pd.DataFrame:
        """从 preprocess AudioSegment 列表提取特征

        Args:
            segments: process_audio() 返回的 AudioSegment 列表
            add_metadata: 是否附加元数据列

        Returns:
            DataFrame: index=segment_id, columns=88维特征 [+ 元数据列]
        """
        rows = []
        indices = []

        for seg in segments:
            feats = self.extract(seg.audio, sr=seg.sample_rate)
            rows.append(feats)

            idx = seg.segment_id or f"seg_{len(indices):03d}"
            indices.append(idx)

        df = pd.DataFrame(rows, index=indices)
        df.index.name = "segment_id"

        if add_metadata:
            df["start_sec"] = [s.start_sec for s in segments]
            df["end_sec"] = [s.end_sec for s in segments]
            df["duration_sec"] = [s.duration_sec for s in segments]
            df["speaker_label"] = [s.speaker_label for s in segments]
            df["quality_flag"] = [s.quality_flag for s in segments]

            # 元数据列放前面
            meta_cols = ["start_sec", "end_sec", "duration_sec", "speaker_label", "quality_flag"]
            other_cols = [c for c in df.columns if c not in meta_cols]
            df = df[meta_cols + other_cols]

        return df

    def _n_features(self) -> int:
        """获取特征维度（首次调用时通过跑 dummy 音频得到）"""
        return len(self.feature_names)


# 全局单例
_extractor = None


def get_extractor() -> FeatureExtractor:
    """获取全局 FeatureExtractor 实例"""
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor
