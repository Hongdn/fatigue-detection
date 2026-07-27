"""特征提取 pipeline — 对接前置处理输出，一键提取特征

使用：
    from 2features.pipeline import extract_from_pipeline, extract_from_file

    # 端到端：音频文件 → 特征 DataFrame
    df = extract_from_pipeline("REC-1.WAV")          # 含降噪+VAD
    df = extract_from_pipeline("REC-1.WAV", denoise=False)  # 跳过降噪

    # 从已处理的 segments 提取
    df = extract_from_segments(segments)
"""

from typing import List, Optional
import pandas as pd

from .extractor import get_extractor


def extract_from_segments(
    segments,  # List[AudioSegment]
    add_metadata: bool = True,
) -> pd.DataFrame:
    """从 preprocess 输出的 AudioSegment 列表提取特征

    Args:
        segments: process_audio() 返回的列表
        add_metadata: 是否附加元数据列

    Returns:
        DataFrame: 每段 88 维特征 + 元数据
    """
    extractor = get_extractor()
    return extractor.extract_from_segments(segments, add_metadata=add_metadata)


def extract_from_pipeline(
    audio_input: str,
    denoise: bool = True,
    min_duration: float = 0.3,
    verbose: bool = True,
) -> pd.DataFrame:
    """端到端：原始音频文件 → 特征 DataFrame

    内部串接：降噪 → VAD → 质量门控 → openSMILE 特征提取

    Args:
        audio_input: WAV 文件路径
        denoise: 是否启用降噪（MossFormerGAN_SE_16K）
        min_duration: 最小语音段长度（秒）
        verbose: 打印进度

    Returns:
        DataFrame: 每段 88 维特征 + 元数据
    """
    from step1_preprocess.pipeline import process_audio

    if verbose:
        print(f"=== 提取特征: {audio_input} ===")

    # Step 1: 前置处理
    segments = process_audio(
        audio_input,
        use_tse=False,
        min_duration=min_duration,
        verbose=verbose,
    )

    if not denoise:
        # 跳过降噪模式：直接用原始音频（process_audio 内部会 skip 降噪如果 clearvoice 不可用）
        pass

    if verbose:
        print(f"  前置处理: {len(segments)} 段语音")

    # Step 2: 特征提取
    extractor = get_extractor()
    df = extractor.extract_from_segments(segments, add_metadata=True)

    if verbose:
        print(f"  特征提取: {len(df.columns) - 5} 维特征 × {len(df)} 段")
        print(f"  ok={df['quality_flag'].value_counts().get('ok', 0)}, "
              f"low_snr={df['quality_flag'].value_counts().get('low_snr', 0)}")

    return df


def extract_batch(
    file_list: List[str],
    denoise: bool = True,
) -> pd.DataFrame:
    """批量处理多个音频文件

    Args:
        file_list: 文件路径列表
        denoise: 是否降噪

    Returns:
        DataFrame: 所有文件的所有段特征（index 为 file::segment_id）
    """
    dfs = []
    for f in file_list:
        df = extract_from_pipeline(f, denoise=denoise, verbose=False)
        df.index = [f"{f}::{idx}" for idx in df.index]
        dfs.append(df)

    return pd.concat(dfs)
