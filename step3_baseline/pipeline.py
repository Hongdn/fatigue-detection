"""基线对齐 pipeline — 接 features 输出，输出标准化特征

使用：
    from 3baseline.pipeline import align_features, align_from_pipeline

    # 从 features DataFrame 对齐
    df_z = align_features(df_features)

    # 端到端：音频 → 特征 → 基线对齐
    df_z = align_from_pipeline("REC-1.WAV")
"""

from typing import Optional
import pandas as pd

from .aligner import BaselineAligner


def align_features(
    df: pd.DataFrame,
    speaker_col: str = "speaker_label",
    min_samples: int = 3,
    aligner: Optional[BaselineAligner] = None,
) -> tuple[pd.DataFrame, BaselineAligner]:
    """对 features 输出的 DataFrame 做个体基线对齐

    Args:
        df: features 模块输出的 DataFrame（含 speaker_label + 88维特征）
        speaker_col: 说话人列名
        min_samples: 个体最少样本数
        aligner: 已有的 BaselineAligner（为 None 则新建并 fit）

    Returns:
        (df_z, aligner)
    """
    if aligner is None:
        aligner = BaselineAligner(min_samples=min_samples)

    df_z = aligner.fit_transform(df, speaker_col=speaker_col)
    return df_z, aligner


def align_from_pipeline(
    audio_input: str,
    denoise: bool = True,
    speaker_label: str = "unknown",
    min_samples: int = 3,
    verbose: bool = True,
) -> tuple[pd.DataFrame, BaselineAligner]:
    """端到端：音频文件 → 特征 → 基线对齐

    内部串接：preprocess → features → baseline

    Args:
        audio_input: WAV 文件路径
        denoise: 是否降噪
        speaker_label: 说话人标签（MVP 阶段手动指定）
        min_samples: 个体最少样本数
        verbose: 打印进度

    Returns:
        (df_z, aligner)
    """
    from step2_features.pipeline import extract_from_pipeline

    if verbose:
        print(f"=== 基线对齐: {audio_input} ===")

    # Step 1-2：前置处理 + 特征提取
    df = extract_from_pipeline(
        audio_input,
        denoise=denoise,
        verbose=verbose,
    )

    # 覆盖说话人标签（如果需要）
    if speaker_label != "unknown":
        df["speaker_label"] = speaker_label

    # Step 3：基线对齐
    if verbose:
        print("[5/5] 个体基线对齐 ...")
    aligner = BaselineAligner(min_samples=min_samples)
    df_z = aligner.fit_transform(df, speaker_col="speaker_label")

    if verbose:
        n_low = df_z["low_confidence"].sum()
        print(f"  → {len(df_z)} 样本, {n_low} 低置信度(用群体基线兜底)")
        summary = aligner.summary()
        for sp, row in summary.iterrows():
            flag = "⚠群体兜底" if row["using_global"] else "✓个体基线"
            print(f"  → 说话人 {sp}: {row['n_samples']}样本 {flag}")

    return df_z, aligner
