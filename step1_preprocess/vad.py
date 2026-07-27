"""多后端 VAD — 语音活动检测与切分

后端：
    silero  — Silero VAD（默认），torch.hub 加载，CPU 实时
    ten     — TEN VAD（可选），pip install ten-vad，更高精度更低延迟

使用：
    from preprocess.vad import vad_detect, vad_segment

    # 默认 Silero
    segments = vad_detect(audio, sr=16000)
    segments = vad_detect(audio, sr=16000, backend="ten")

    # 切分
    chunks = vad_segment(audio, sr=16000)
"""

from dataclasses import dataclass
from typing import List, Optional, Literal
import numpy as np


VadBackend = Literal["silero", "ten"]


@dataclass
class VadSegment:
    """VAD 检测出的语音段时间戳"""
    start_sec: float
    end_sec: float
    duration_sec: float = 0.0
    backend: str = "silero"

    def __post_init__(self):
        self.duration_sec = self.end_sec - self.start_sec


# ━━━ Silero VAD 后端 ━━━

_silero_model = None
_silero_utils = None


def _get_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is not None and _silero_utils is not None:
        return _silero_model, _silero_utils

    import torch
    _silero_model, _silero_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return _silero_model, _silero_utils


def _vad_detect_silero(
    audio: np.ndarray,
    sr: int,
    speech_threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    window_size_ms: int,
) -> List[VadSegment]:
    import torch
    model, utils = _get_silero_vad()
    get_speech_timestamps = utils[0]

    if audio.dtype != np.int16:
        audio_int16 = (audio * 32767).astype(np.int16)
    else:
        audio_int16 = audio

    tensor = torch.from_numpy(audio_int16).float() / 32768.0
    timestamps = get_speech_timestamps(
        tensor, model,
        threshold=speech_threshold,
        sampling_rate=sr,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        window_size_samples=window_size_ms * sr // 1000,
    )
    return [
        VadSegment(start_sec=ts["start"] / sr, end_sec=ts["end"] / sr, backend="silero")
        for ts in timestamps
    ]


# ━━━ TEN VAD 后端 ━━━

_ten_vad = None


def _get_ten_vad():
    global _ten_vad
    if _ten_vad is not None:
        return _ten_vad
    try:
        from ten_vad import TENVAD
        _ten_vad = TENVAD()
        return _ten_vad
    except ImportError:
        raise ImportError(
            "TEN VAD 未安装。请执行:\n"
            "  pip install git+https://github.com/TEN-framework/ten-vad.git"
        )


def _vad_detect_ten(
    audio: np.ndarray,
    sr: int,
    speech_threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
) -> List[VadSegment]:
    """TEN VAD 逐帧检测，合并相邻语音帧为语音段"""
    vad = _get_ten_vad()

    # TEN VAD 期望 int16
    if audio.dtype != np.int16:
        audio_int16 = (audio * 32767).astype(np.int16)
    else:
        audio_int16 = audio

    # 逐帧检测（512 样本 = 32ms at 16kHz）
    chunk_size = 512
    n_chunks = len(audio_int16) // chunk_size
    speech_frames = np.zeros(n_chunks, dtype=bool)

    for i in range(n_chunks):
        chunk = audio_int16[i * chunk_size : (i + 1) * chunk_size]
        prob = vad.process(chunk)
        speech_frames[i] = prob > speech_threshold

    # 帧合并为段
    frame_duration = chunk_size / sr  # 秒
    min_speech_frames = max(1, int(min_speech_duration_ms / 1000 / frame_duration))
    min_silence_frames = max(1, int(min_silence_duration_ms / 1000 / frame_duration))

    segments = []
    in_speech = False
    speech_start = 0
    silence_count = 0

    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            if not in_speech:
                speech_start = i
                in_speech = True
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    # 检查段长
                    seg_duration = (i - silence_count - speech_start) * frame_duration
                    if seg_duration * 1000 >= min_speech_duration_ms:
                        segments.append(VadSegment(
                            start_sec=speech_start * frame_duration,
                            end_sec=(i - silence_count + 1) * frame_duration,
                            backend="ten",
                        ))
                    in_speech = False

    # 尾部未闭合段
    if in_speech:
        seg_duration = (n_chunks - speech_start) * frame_duration
        if seg_duration * 1000 >= min_speech_duration_ms:
            segments.append(VadSegment(
                start_sec=speech_start * frame_duration,
                end_sec=n_chunks * frame_duration,
                backend="ten",
            ))

    return segments


# ━━━ 统一入口 ━━━

def vad_detect(
    audio: np.ndarray,
    sr: int = 16000,
    backend: VadBackend = "silero",
    min_speech_duration_ms: int = 300,
    min_silence_duration_ms: int = 200,
    speech_threshold: float = 0.5,
    window_size_ms: int = 30,
) -> List[VadSegment]:
    """检测语音活动段，返回时间戳列表

    Args:
        audio: 音频数据，float32，单声道
        sr: 采样率
        backend: "silero"（默认）或 "ten"
        min_speech_duration_ms: 最小语音段长度（ms）
        min_silence_duration_ms: 最小静音间隔（ms）
        speech_threshold: 语音概率阈值（0~1）
        window_size_ms: 滑窗大小（仅 Silero 有效）

    Returns:
        语音段时间戳列表
    """
    if backend == "ten":
        return _vad_detect_ten(
            audio, sr, speech_threshold,
            min_speech_duration_ms, min_silence_duration_ms,
        )
    else:
        return _vad_detect_silero(
            audio, sr, speech_threshold,
            min_speech_duration_ms, min_silence_duration_ms, window_size_ms,
        )


def vad_segment(
    audio: np.ndarray,
    sr: int = 16000,
    backend: VadBackend = "silero",
    min_speech_duration_ms: int = 300,
    min_silence_duration_ms: int = 200,
    speech_threshold: float = 0.5,
) -> List[np.ndarray]:
    """检测并切分语音段，返回音频段列表

    Args:
        audio: 音频数据，float32
        sr: 采样率
        backend: "silero" 或 "ten"
        min_speech_duration_ms: 最小语音段长度（ms）
        min_silence_duration_ms: 最小静音间隔（ms）
        speech_threshold: 语音概率阈值

    Returns:
        切分后的音频段列表
    """
    segments = vad_detect(
        audio=audio, sr=sr, backend=backend,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_threshold=speech_threshold,
    )
    chunks = []
    for seg in segments:
        start_sample = int(seg.start_sec * sr)
        end_sample = int(seg.end_sec * sr)
        chunks.append(audio[start_sample:end_sample].astype(np.float32))
    return chunks


def vad_merge_segments(
    segments: List[VadSegment],
    gap_threshold_sec: float = 0.3,
) -> List[VadSegment]:
    """合并间隔小于阈值的相邻段"""
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.start_sec - last.end_sec < gap_threshold_sec:
            last.end_sec = seg.end_sec
            last.duration_sec = last.end_sec - last.start_sec
        else:
            merged.append(seg)
    return merged
