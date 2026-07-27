"""前置处理 pipeline — 串联重采样→降噪→VAD→质量门控

主入口：process_audio()

流程：
    原始音频(任意sr) → 重采样16kHz → MossFormerGAN降噪 → VAD切分 → 质量门控
    → 返回 AudioSegment 列表

MVP 阶段：不上 TSE，VAD 切分后直接使用。

使用：
    from preprocess import process_audio

    segments = process_audio("rec1.wav")
    for seg in segments:
        print(f"{seg.segment_id}: {seg.start_sec:.1f}-{seg.end_sec:.1f}s "
              f"quality={seg.quality_flag}")
"""

import os
import time
from typing import List, Optional

import numpy as np
import librosa

from .denoise import denoise, _resample_to_16k
from .vad import vad_detect, vad_segment
from .quality import AudioSegment, quality_filter, quality_summary


def process_audio(
    audio_input: str | np.ndarray,
    sr: Optional[int] = None,
    use_tse: bool = False,
    ref_speaker_path: Optional[str] = None,
    min_duration: float = 0.3,
    snr_threshold: float = 3.0,
    speaker_label: str = "unknown",
    verbose: bool = True,
) -> List[AudioSegment]:
    """前置处理主入口：音频 → 降噪 → VAD → 质量过j滤

    使用方式一：从文件
        segments = process_audio("REC-1.WAV")

    使用方式二：从 numpy 数组
        audio, sr = librosa.load("REC-1.WAV", sr=None)
        segments = process_audio(audio, sr=sr)

    Args:
        audio_input: WAV 文件路径 或 numpy 数组
        sr: 采样率。如果 audio_input 是 numpy 数组，必须提供；如果是文件路径则自动检测
        use_tse: 是否启用 TSE（MVP 阶段为 False）
        ref_speaker_path: TSE 参考语音路径（仅 use_tse=True 时有效）
        min_duration: 语音段最小有效时长（秒），低于此值的段被丢弃
        snr_threshold: SNR 质量阈值（dB）
        speaker_label: 说话人标签（MVP 用数据集自带的标签）
        verbose: 是否打印处理进度

    Returns:
        AudioSegment 列表（按时间排序，质量问题段已标 quality_flag）

    Raises:
        ImportError: ClearerVoice 未安装
        FileNotFoundError: 音频文件不存在
    """
    t0 = time.time()

    # ━━━ 0. 加载 + 重采样到 16kHz ━━━
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"音频文件不存在: {audio_input}")
        if verbose:
            print(f"[0/4] 加载音频: {os.path.basename(audio_input)}")
        audio, sr = librosa.load(audio_input, sr=None, mono=True)
    elif isinstance(audio_input, np.ndarray):
        audio = audio_input
        if sr is None:
            raise ValueError("传入 numpy 数组时必须提供 sr 参数")
    else:
        raise TypeError(f"audio_input 必须是文件路径或 numpy 数组，收到 {type(audio_input)}")

    original_sr = sr
    audio, sr = _resample_to_16k(audio, original_sr)
    if verbose and original_sr != sr:
        print(f"  → 重采样 {original_sr}Hz → {sr}Hz")
    if verbose:
        duration = len(audio) / sr
        print(f"  → 时长 {duration:.1f}s，{len(audio)} 样本")

    # ━━━ 1. 降噪 ━━━
    if verbose:
        print("[1/4] ClearerVoice 降噪 (MossFormerGAN_SE_16K) ...")
    try:
        audio_denoised = denoise(audio, sr=16000)
    except ImportError as e:
        print(f"  ⚠ 降噪模块不可用: {e}")
        print("  → 跳过降噪，使用原始音频（特征可能受噪声影响）")
        audio_denoised = audio
    if verbose:
        print(f"  → 降噪完成")

    # ━━━ 2. VAD 切分 ━━━
    if verbose:
        print("[2/4] Silero VAD 语音活动检测 ...")
    timestamps = vad_detect(audio_denoised, sr=sr)
    if verbose:
        total_speech = sum(t.end_sec - t.start_sec for t in timestamps)
        print(f"  → 检测到 {len(timestamps)} 个语音段，总语音 {total_speech:.1f}s")

    # ━━━ 3. TSE（二期，MVP 跳过） ━━━
    if use_tse:
        if verbose:
            print("[3/4] TSE 目标说话人提取 ...")
        # 二期实现
        from .tse import tse_extract
        # ... 对每段做 TSE 提取

    # ━━━ 4. 质量门控 ━━━
    if verbose:
        print("[4/4] 质量门控 ...")

    # 构建 AudioSegment 列表
    segments = []
    for i, ts in enumerate(timestamps):
        start_samp = int(ts.start_sec * sr)
        end_samp = int(ts.end_sec * sr)
        seg_audio = audio_denoised[start_samp:end_samp].astype(np.float32)

        segments.append(AudioSegment(
            audio=seg_audio,
            sample_rate=sr,
            start_sec=ts.start_sec,
            end_sec=ts.end_sec,
            segment_id=f"seg_{i:03d}",
            speaker_label=speaker_label,
            tse_confidence=1.0,  # MVP 为 1.0
        ))

    # 质量过滤
    segments = quality_filter(
        segments,
        min_duration=min_duration,
        snr_threshold=snr_threshold,
    )

    summary = quality_summary(segments)
    elapsed = time.time() - t0
    if verbose:
        print(f"  → 保留 {summary['ok']} ok, "
              f"{summary['low_snr']} low_snr, "
              f"{summary['uncertain']} uncertain")
        print(f"  → 耗时 {elapsed:.1f}s\n")

    return segments


def process_batch(
    file_list: List[str],
    output_dir: Optional[str] = None,
    **kwargs,
) -> dict:
    """批量处理多个音频文件

    Args:
        file_list: 文件路径列表
        output_dir: 输出目录（可选，暂未实现段级写入）
        **kwargs: 传参给 process_audio() 的参数

    Returns:
        {"results": {path: [AudioSegment, ...], ...},
         "summary": {"total_files": N, "success": N, "failed": N, "errors": [...]}}
    """
    results = {"results": {}, "summary": {"total_files": len(file_list), "success": 0, "failed": 0, "errors": []}}

    for f in file_list:
        try:
            segments = process_audio(f, verbose=False, **kwargs)
            results["results"][f] = segments
            results["summary"]["success"] += 1
        except Exception as e:
            results["summary"]["failed"] += 1
            results["summary"]["errors"].append(f"{f}: {e}")

    return results
