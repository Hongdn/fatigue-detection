"""前置处理层 — 音频预处理模块

模块：
    denoise  : ClearerVoice MossFormerGAN_SE_16K 降噪
    vad      : Silero VAD 语音活动检测
    tse      : TSE 目标说话人提取（二期）
    quality  : 质量门控过滤
    pipeline : 串联上述模块的完整入口
"""

from .pipeline import process_audio

__all__ = ["process_audio"]
