"""step5 pipeline — 接 step4 推理结果，输出状态判定

使用：
    from step5_rules.pipeline import classify_from_step4

    # 从 step4 推理结果判定
    df_state = classify_from_step4(df_step4)

    # 端到端：音频文件 → step4 推理 → step5 判定
    df_state = classify_from_audio("REC-1.WAV")
"""

import pandas as pd
from pathlib import Path
from .engine import classify_dataframe, state_summary


def classify_from_step4(
    df_step4: pd.DataFrame,
) -> pd.DataFrame:
    """从 step4 推理结果判定状态

    Args:
        df_step4: step4_regression.predict() 输出的 DataFrame

    Returns:
        DataFrame，新增 state / confidence / reason 列
    """
    df = classify_dataframe(df_step4)
    return df


def classify_from_audio(
    audio_path: str,
    model_dir: str = "step4_regression/models/",
) -> pd.DataFrame:
    """端到端：音频文件 → step4 推理 → step5 状态判定

    Args:
        audio_path: WAV 文件路径
        model_dir: XGBoost 模型目录

    Returns:
        DataFrame，包含 arousal/exertion/state/reason
    """
    from step4_regression.pipeline import predict_from_step3

    # step4 推理
    df_step4 = predict_from_step3(audio_path, model_dir=model_dir, verbose=False)

    # step5 判定
    df_state = classify_dataframe(df_step4)

    return df_state


def run_on_step4_results(
    results_dir: str = "output/",
    model_dir: str = "step4_regression/models/",
) -> pd.DataFrame:
    """对已有的 step4 训练结果 CSV 做状态判定演示

    Args:
        results_dir: 包含 step4 结果的目录

    Returns:
        带 state 列的 DataFrame
    """
    import os
    # 直接对已保存的结果做判定
    # 这里用实际跑过的结果来做演示

    # 从 step4 的 2000 条数据中取测试集的预测
    from step4_regression.model import load_models
    import numpy as np

    # 重新跑一次小批量推理来做演示
    import pyarrow.parquet as pq
    import soundfile as sf
    import io, warnings, os
    import opensmile

    parquet_path = os.path.join(results_dir, "..", "data", "train-00005-of-00010.parquet")
    parquet_path = os.path.abspath(parquet_path)
    if not os.path.exists(parquet_path):
        # fallback to project root
        parquet_path = "data/train-00005-of-00010.parquet"
    df_pq = pq.read_table(parquet_path).to_pandas()
    models = load_models(model_dir)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    features, y_arousal, y_nature = [], [], []
    for i in range(min(500, len(df_pq))):
        try:
            row = df_pq.iloc[i]
            audio, sr = sf.read(io.BytesIO(row['audio-path']['bytes']))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=16000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feat = smile.process_signal(audio.astype(np.float32), sr)
            features.append(feat.iloc[0].values)
            y_arousal.append(row['score_arousal'])
            y_nature.append(row['score_nature'])
        except:
            pass

    X = np.array(features, dtype=np.float32)
    pred_a = models["arousal"].predict(X)
    pred_e = models["exertion"].predict(X)
    fatigue = 1.0 - pred_a

    df = pd.DataFrame({
        "segment_id": [f"seg_{i:04d}" for i in range(len(X))],
        "arousal": np.clip(pred_a, 0, 1),
        "exertion": np.clip(pred_e, 0, 1),
        "fatigue_prob": np.clip(fatigue, 0, 1),
    })

    df_state = classify_dataframe(df)
    summary = state_summary(df_state)

    print("=== Step5 状态判定结果 ===")
    print(f"样本数: {summary['total']}")
    print(f"兴奋: {summary['兴奋']} ({summary['兴奋_pct']}%)")
    print(f"稳定: {summary['稳定']} ({100 - summary['兴奋_pct'] - summary['疲劳_pct']:.1f}%)")
    print(f"疲劳: {summary['疲劳']} ({summary['疲劳_pct']}%)")
    print(f"平均 arousal: {summary['avg_arousal']:.4f}")
    print(f"平均 exertion: {summary['avg_exertion']:.4f}")

    # 输出几个示例
    print("\n=== 判定示例 ===")
    for s in ["疲劳", "兴奋", "稳定"]:
        samples = df_state[df_state["state"] == s].head(2)
        for _, row in samples.iterrows():
            print(f"  [{row['state']}]({row['confidence']}) a={row['arousal']:.3f} e={row['exertion']:.3f}")
            print(f"    → {row['reason']}")

    # 保存结果
    out_path = os.path.join(results_dir, "step5_state_results.csv")
    df_state.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至 {out_path}")

    return df_state
