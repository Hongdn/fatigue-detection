"""ClearerVoice 降噪 — MossFormerGAN_SE_16K 封装

自动处理任意采样率：检测 → 必要时重采样到16kHz → 降噪 → 返回16kHz结果

安装：
    git clone https://github.com/modelscope/ClearerVoice-Studio
    cd ClearerVoice-Studio && pip install -r requirements.txt

使用：
    from preprocess.denoise import denoise, denoise_file
    audio = denoise(np_array, sr=8000)  # numpy 输入
    denoise_file("input.wav", "output.wav")  # 文件输入
"""

import os
import tempfile
import warnings
from typing import Optional

import numpy as np
import librosa


# 延迟导入，避免未安装时报错
_CLEARVOICE_AVAILABLE = None


def _check_clearvoice():
    """检查 ClearerVoice 是否可用"""
    global _CLEARVOICE_AVAILABLE
    if _CLEARVOICE_AVAILABLE is not None:
        return _CLEARVOICE_AVAILABLE

    try:
        from clearvoice import ClearVoice  # noqa: F401
        _CLEARVOICE_AVAILABLE = True
    except ImportError:
        _CLEARVOICE_AVAILABLE = False
    return _CLEARVOICE_AVAILABLE


def _find_output_wav(output_dir: str) -> str:
    """在 ClearVoice 输出目录中递归找到 wav 文件

    ClearVoice 输出结构: tmp_out/<ModelName>/output_xxx.wav
    """
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".wav"):
                return os.path.join(root, f)
    raise RuntimeError(f"ClearerVoice 输出目录中未找到 wav 文件: {output_dir}")


# 模型单例，避免重复加载
_enhancer = None


def _get_enhancer():
    """获取 ClearVoice 增强器（单例）"""
    global _enhancer
    if _enhancer is not None:
        return _enhancer

    if not _check_clearvoice():
        raise ImportError(
            "ClearerVoice-Studio 未安装。请执行：\n"
            "  git clone https://github.com/modelscope/ClearerVoice-Studio\n"
            "  cd ClearerVoice-Studio && pip install -r requirements.txt"
        )

    from clearvoice import ClearVoice

    _enhancer = ClearVoice(
        task="speech_enhancement",
        model_names=["MossFormerGAN_SE_16K"],
    )
    return _enhancer


def _resample_to_16k(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """重采样到 16kHz（如果是其他采样率）"""
    if sr == 16000:
        return audio.astype(np.float32), 16000

    return librosa.resample(
        y=audio.astype(np.float64),
        orig_sr=sr,
        target_sr=16000,
    ).astype(np.float32), 16000


def denoise(
    audio: np.ndarray,
    sr: int = 16000,
    use_tempfile: bool = True,
) -> np.ndarray:
    """降噪单段音频

    Args:
        audio: 输入音频，shape=(n,) 或 (n, channels)，支持任意采样率
        sr: 原始采样率（如果不是 16kHz 会自动重采样）
        use_tempfile: True 用临时文件，False 尝试直接调用（实验性）

    Returns:
        降噪后音频，float32，16kHz

    Raises:
        ImportError: ClearerVoice 未安装
        RuntimeError: 降噪处理失败
    """
    cv = _get_enhancer()

    # 1. 转单声道
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    # 2. 重采样到 16kHz
    audio_16k, _ = _resample_to_16k(audio, sr)

    # 3. 写入临时文件
    if use_tempfile:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
            # 写入输入
            import soundfile as sf
            sf.write(tmp_in.name, audio_16k, 16000, subtype="PCM_16")
            tmp_in.flush()

            # 降噪
            cv(
                input_path=tmp_in.name,
                online_write=True,
                output_path=tmp_out,
            )

            # 读取输出（ClearVoice 输出结构：tmp_out/<ModelName>/output_xxx.wav）
            output_path = _find_output_wav(tmp_out)
            result, _ = librosa.load(output_path, sr=16000, mono=True)

        # 清理临时输入
        os.unlink(tmp_in.name)
    else:
        # 实验性：直接调用（依赖 ClearVoice 内部 API）
        raise NotImplementedError(
            "直接调用模式暂未实现，请使用 use_tempfile=True"
        )

    return result.astype(np.float32)


def denoise_file(input_path: str, output_path: str) -> None:
    """从文件降噪（便捷方法）

    Args:
        input_path: 输入 WAV 文件路径
        output_path: 输出 WAV 文件路径
    """
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    result = denoise(audio, sr=sr)

    import soundfile as sf
    sf.write(output_path, result, 16000, subtype="PCM_16")


# ---- 批处理 ----


def denoise_directory(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.wav",
    recursive: bool = False,
) -> dict:
    """批量降噪目录下所有音频

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件名匹配模式
        recursive: 是否递归子目录

    Returns:
        {"success": N, "failed": M, "errors": [msg, ...]}
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, pattern))
    if recursive:
        files += glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
    files = list(set(files))

    results = {"success": 0, "failed": 0, "errors": []}

    for f in files:
        rel = os.path.relpath(f, input_dir)
        out = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            denoise_file(f, out)
            results["success"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{rel}: {e}")

    return results
