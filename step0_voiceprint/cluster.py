"""声纹提取与说话人聚类 — 基于 3D-Speaker ERes2NetV2 (ModelScope)

用法:
    from step0_voiceprint import SpeakerClusterer

    clusterer = SpeakerClusterer()
    speaker_ids = clusterer.fit_predict(audio_segments, sr)
    # → ["speaker_0", "speaker_0", "speaker_1", ...]

后端: ERes2NetV2 (达摩院, 20万中文说话人预训练, EER 0.61%)
模型: iic/speech_eres2netv2_sv_zh-cn_16k-common
"""

import numpy as np
from typing import List, Optional, Dict
from collections import Counter


class SpeakerClusterer:
    """声纹提取 + 聚类分组

    对每段音频提取 192 维声纹向量，
    然后用余弦相似度做层次聚类，自动分组。
    """

    MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"

    def __init__(self, threshold: float = 0.55):
        """
        Args:
            threshold: 余弦相似度阈值，高于此值认为是同一人
                       0.40 宽松（少分组） / 0.55 平衡 / 0.70 严格（多分组）
        """
        self.threshold = threshold
        self._pipeline = None
        self._embeddings: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def _ensure_model(self):
        """延迟加载模型（首次调用时下载）"""
        if self._pipeline is not None:
            return

        import tempfile, os, soundfile as sf

        # 兼容补丁: datasets 4.x 将 ALL_ALLOWED_EXTENSIONS 改名为 _ALL_ALLOWED_EXTENSIONS
        import datasets.load as _dl
        if not hasattr(_dl, 'ALL_ALLOWED_EXTENSIONS') and hasattr(_dl, '_ALL_ALLOWED_EXTENSIONS'):
            _dl.ALL_ALLOWED_EXTENSIONS = _dl._ALL_ALLOWED_EXTENSIONS

        # 用 ModelScope pipeline 加载
        from modelscope.pipelines import pipeline as ms_pipeline
        self._pipeline = ms_pipeline(
            task='speaker-verification',
            model=self.MODEL_ID,
        )
        # 保存 soundfile 引用供 extract 用
        self._sf = sf
        self._tempfile = tempfile
        self._os = os

    def extract_embeddings(
        self,
        audio_segments: List[np.ndarray],
        sr: int = 16000,
    ) -> np.ndarray:
        """逐段提取声纹向量

        Args:
            audio_segments: 音频数组列表，每段为 float32/float64 一维数组
            sr: 采样率

        Returns:
            (n_segments, 192) 的 embedding 矩阵
        """
        self._ensure_model()

        embeddings = []
        for i, seg in enumerate(audio_segments):
            if len(seg) < sr * 0.3:
                # 太短的段填零
                embeddings.append(np.zeros(192, dtype=np.float32))
                continue

            # ModelScope pipeline 需要文件路径，写临时 wav
            tmp = self._tempfile.NamedTemporaryFile(
                suffix='.wav', delete=False, dir=self._os.environ.get('TEMP', '/tmp')
            )
            tmp_path = tmp.name
            tmp.close()

            try:
                seg_data = seg.astype(np.float32) if seg.dtype != np.float32 else seg
                self._sf.write(tmp_path, seg_data, sr)
                emb = self._pipeline([tmp_path], output_emb=True)
                # pipeline 返回 dict: {'outputs': ..., 'embs': np.ndarray shape (1,192)}
                embeddings.append(emb['embs'].squeeze())
            except Exception as e:
                # fallback: 填零
                embeddings.append(np.zeros(192, dtype=np.float32))
            finally:
                try:
                    self._os.unlink(tmp_path)
                except:
                    pass

        self._embeddings = np.array(embeddings, dtype=np.float32)
        return self._embeddings

    def cluster(self) -> np.ndarray:
        """对已提取的 embedding 做聚类

        贪婪层次聚类：从第一段开始，
        依次判断后续段与已分组段的最小余弦距离，
        低于阈值则归入该组，否则新建一组。

        Returns:
            labels, shape (n,), 从 0 开始
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            raise RuntimeError("请先调用 extract_embeddings()")

        E = self._embeddings
        n = len(E)

        # L2 归一化
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E_norm = E / (norms + 1e-8)

        labels = np.full(n, -1, dtype=int)
        group_centroids = []

        for i in range(n):
            # 跳过零向量（太短的段）
            if norms[i] < 1e-6:
                labels[i] = -1  # 标记为无效
                continue

            best_group = None
            best_sim = -1.0

            for g, centroid in enumerate(group_centroids):
                sim = float(np.dot(E_norm[i], centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_group = g

            if best_group is not None and best_sim >= self.threshold:
                labels[i] = best_group
                # 更新质心
                members = (labels == best_group)
                group_centroids[best_group] = E_norm[members].mean(axis=0)
            else:
                labels[i] = len(group_centroids)
                group_centroids.append(E_norm[i].copy())

        # 无效段(-1)归到第一组
        labels[labels == -1] = 0 if len(group_centroids) > 0 else 0

        self._labels = labels
        return labels

    def fit_predict(
        self,
        audio_segments: List[np.ndarray],
        sr: int = 16000,
        prefix: str = "speaker",
    ) -> List[str]:
        """端到端：提取 embedding + 聚类 → 返回说话人标签

        Args:
            audio_segments: 音频段列表
            sr: 采样率
            prefix: 标签前缀

        Returns:
            每段对应的说话人 ID 列表
        """
        self.extract_embeddings(audio_segments, sr)
        labels = self.cluster()
        return [f"{prefix}_{l}" for l in labels]

    def summary(self) -> Dict:
        """聚类摘要"""
        if self._labels is None:
            return {"status": "not clustered"}
        counter = Counter(self._labels.tolist())
        return {
            "n_speakers": len(counter),
            "segments_per_speaker": dict(sorted(counter.items())),
            "total_segments": len(self._labels),
            "threshold": self.threshold,
            "model": self.MODEL_ID,
        }
