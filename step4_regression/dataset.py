"""数据集加载 — ExpressiveSpeech（HuggingFace parquet）

数据格式：parquet 文件，音频嵌入为 WAV bytes，每条约含：
    score_arousal / score_prosody / score_nature / score_expressive
    emotion / from / value / length / audio-path

使用：
    from step4_regression.dataset import load_expressive_speech_metadata
    df = load_expressive_speech_metadata("data/train-00005-of-00010.parquet")
"""

import pyarrow.parquet as pq
from typing import Optional

import pandas as pd


def load_parquet_metadata(parquet_path: str) -> pd.DataFrame:
    """直接读取 ExpressiveSpeech parquet 文件的元数据列（不含音频）

    Args:
        parquet_path: parquet 文件路径

    Returns:
        DataFrame，含 No/from/value/emotion/length/score_arousal/score_prosody/
                      score_nature/score_expressive
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # 去除 audio 列（大，不需要）
    meta_cols = [c for c in df.columns if c != 'audio-path']
    return df[meta_cols]


def load_parquet_with_audio(parquet_path: str, n: Optional[int] = None) -> dict:
    """读取 parquet，返回元数据 + 解码后的音频数组

    Args:
        parquet_path: parquet 文件路径
        n: 读取条数（None = 全部）

    Returns:
        {"arousal": [...], "nature": [...], "audio": [[samples], ...], "emotion": [...]}
    """
    import io
    import soundfile as sf
    import numpy as np

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if n:
        df = df.head(n)

    arousal = []
    nature = []
    audio_list = []
    emotion = []

    for _, row in df.iterrows():
        try:
            audio, sr = sf.read(io.BytesIO(row['audio-path']['bytes']))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=16000)
            audio_list.append(audio.astype(np.float32))
            arousal.append(row['score_arousal'])
            nature.append(row['score_nature'])
            emotion.append(str(row.get('emotion', '')))
        except:
            continue

    return {
        "arousal": np.array(arousal, dtype=np.float32),
        "nature": np.array(nature, dtype=np.float32),
        "audio": audio_list,
        "emotion": emotion,
        "n_samples": len(audio_list),
    }


# ━━━ 已弃用的函数（保留以便将来可能复用） ━━━
# load_atc_corpus(), _scan_atc_directory(), ATCSample, leave_one_speaker_out_split()
# 原因：ATC Fatigue Corpus 需付费，改用 ExpressiveSpeech
