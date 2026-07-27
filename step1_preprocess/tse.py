"""TSE 目标说话人提取 — 二期模块（代完成）

用管制员参考语音（3-5 秒）从混合录音中一步提取目标说话人语音。

触发条件：多人重叠/换班/跨频道关联/串扰过滤
触发时机：真实多管制员录音到位后

二期集成：
    from preprocess.tse import tse_extract
    target_audio = tse_extract(mixed_audio, ref_audio, sr=16000)

依赖：
    ClearerVoice-Studio: task='target_speaker_extraction'（纯音频 8kHz 模型）
"""

from typing import Optional
import numpy as np


# ---------- 采样率桥接 ----------

def _resample_16k_to_8k(audio: np.ndarray) -> np.ndarray:
    """16kHz → 8kHz 降采样"""
    import librosa
    return librosa.resample(audio.astype(np.float64), orig_sr=16000, target_sr=8000).astype(np.float32)


def _resample_8k_to_16k(audio: np.ndarray) -> np.ndarray:
    """8kHz → 16kHz 升采样"""
    import librosa
    return librosa.resample(audio.astype(np.float64), orig_sr=8000, target_sr=16000).astype(np.float32)


# ---------- TSE 核心 ----------

def tse_extract(
    audio: np.ndarray,
    ref_audio: np.ndarray,
    sr: int = 16000,
) -> np.ndarray:
    """从混合音频中提取目标说话人纯净语音

    内部自动处理 16kHz⇄8kHz 桥接。

    Args:
        audio: 混合音频（可能包含多个说话人），16kHz float32
        ref_audio: 目标说话人参考语音（3-5 秒干净样本），16kHz float32
        sr: 采样率（当前仅支持 16000）

    Returns:
        目标说话人纯净语音，16kHz float32

    Raises:
        NotImplementedError: MVP 阶段暂未实现
        ImportError: ClearerVoice 未安装
    """
    if sr != 16000:
        raise ValueError(f"TSE 输入需要 16kHz，收到 {sr}Hz")

    # 1. 16kHz → 8kHz 降采样（纯音频 TSE 模型仅支持 8kHz）
    audio_8k = _resample_16k_to_8k(audio)
    ref_8k = _resample_16k_to_8k(ref_audio)

    # 2. 调用 ClearerVoice TSE
    audio_extracted_8k = _run_tse(audio_8k, ref_8k)

    # 3. 8kHz → 16kHz 升采样
    audio_extracted_16k = _resample_8k_to_16k(audio_extracted_8k)

    return audio_extracted_16k.astype(np.float32)


def _run_tse(mixture_8k: np.ndarray, ref_8k: np.ndarray) -> np.ndarray:
    """实际调用 ClearerVoice TSE 模型

    二期需确认 ClearerVoice 的 TSE Python API 接口。
    目前框架已就位，待 API 确认后填充。
    """
    raise NotImplementedError(
        "TSE 目标说话人提取是二期功能，MVP 阶段暂不需要。\n"
        "公开数据集（RECOLA/RAVDESS）是单说话人录音，VAD 切分后直接使用。\n\n"
        "二期接入步骤：\n"
        "  1. 采集管制员参考语音（3-5s），保存为 16kHz WAV\n"
        "  2. 确认 ClearerVoice 的 TSE Python API\n"
        "  3. 填充 _run_tse() 函数\n"
        "  4. 测试: tse_extract(mixed_16k, ref_16k)\n\n"
        "采样率桥接已就位：16kHz→8kHz→TSE→16kHz\n"
        "（VHF 窄带 0~3.4kHz，8kHz 覆盖，桥接无损）"
    )


# ---------- 参考语音库 ----------

def build_speaker_library(
    speaker_files: dict,
) -> dict:
    """构建说话人参考语音库

    Args:
        speaker_files: {"管制员A": "path/to/ref_a.wav", ...}

    Returns:
        {"管制员A": np.ndarray, ...} 16kHz float32
    """
    import librosa
    library = {}
    for speaker_id, path in speaker_files.items():
        audio, sr = librosa.load(path, sr=16000, mono=True)
        library[speaker_id] = audio.astype(np.float32)
    return library
