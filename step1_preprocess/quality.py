"""质量门控 — 对 VAD 切出的语音段做质量评估和过滤

规则（按设计文档 3.3）：
    - SNR < 阈值 → low_snr，降权或丢弃（REC-4 教训）
    - 段长 < 0.3s → too_short，丢弃
    - TSE 置信度 < 0.6 → uncertain，降权
    - 其他 → ok，正常喂特征提取
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class AudioSegment:
    """单个语音段，携带元数据"""
    audio: np.ndarray                # 音频数据 (n_samples,)，float32，16kHz
    sample_rate: int = 16000
    start_sec: float = 0.0
    end_sec: float = 0.0
    duration_sec: float = 0.0
    segment_id: str = ""
    speaker_label: str = "unknown"   # MVP 用数据集标签，二期用 TSE 判定
    tse_confidence: float = 1.0      # TSE 提取置信度，MVP 为 1.0
    quality_flag: str = "ok"         # ok / low_snr / too_short / uncertain


def estimate_snr(audio: np.ndarray) -> float:
    """简化 SNR 估计：用语音段能量 vs 底部 10% 分位噪声能量

    Args:
        audio: 16kHz float32 音频

    Returns:
        估计 SNR (dB)，纯静音返回 -inf
    """
    if len(audio) == 0:
        return float("-inf")

    # 帧级能量
    frame_len = 512  # 32ms at 16kHz
    hop = 256
    n_frames = (len(audio) - frame_len) // hop + 1

    if n_frames < 2:
        return float("-inf")

    energies = np.array([
        np.sum(audio[i * hop : i * hop + frame_len] ** 2)
        for i in range(n_frames)
    ])

    # 噪声水平：底部 10% 帧能量
    noise_floor = np.percentile(energies, 10)
    signal_level = np.percentile(energies, 90)

    if noise_floor < 1e-12:
        return 30.0  # 非常干净

    snr = 10 * np.log10(max(signal_level / noise_floor, 1e-6))
    return float(snr)


def quality_filter(
    segments: List[AudioSegment],
    min_duration: float = 0.3,
    snr_threshold: float = 3.0,
    tse_confidence_threshold: float = 0.6,
) -> List[AudioSegment]:
    """对语音段列表做质量评估和过滤

    Args:
        segments: VAD 切出的语音段列表
        min_duration: 最小有效段长（秒），低于此值丢弃
        snr_threshold: SNR 阈值（dB），低于此值标记 low_snr
        tse_confidence_threshold: TSE 置信度阈值，低于此值标记 uncertain

    Returns:
        过滤后的语音段列表（too_short 被丢弃，其余保留但可能标记）
    """
    filtered = []

    for i, seg in enumerate(segments):
        seg.segment_id = seg.segment_id or f"seg_{i:03d}"
        seg.duration_sec = len(seg.audio) / seg.sample_rate

        # 1. 段长检查
        if seg.duration_sec < min_duration:
            seg.quality_flag = "too_short"
            continue  # 丢弃

        # 2. SNR 检查
        snr = estimate_snr(seg.audio)
        if snr < snr_threshold:
            seg.quality_flag = "low_snr"
        else:
            seg.quality_flag = "ok"

        # 3. TSE 置信度检查
        if seg.tse_confidence < tse_confidence_threshold:
            seg.quality_flag = "uncertain"

        filtered.append(seg)

    return filtered


def quality_summary(segments: List[AudioSegment]) -> dict:
    """输出质量统计摘要"""
    flags = [s.quality_flag for s in segments]
    return {
        "total": len(segments),
        "ok": flags.count("ok"),
        "low_snr": flags.count("low_snr"),
        "too_short": flags.count("too_short"),
        "uncertain": flags.count("uncertain"),
    }
