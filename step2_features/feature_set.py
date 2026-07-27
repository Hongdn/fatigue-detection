"""eGeMAPSv02 特征分组 — 按物理含义分类，标注疲劳相关性

总 88 维，分 9 组：
    pitch        基频  — 疲劳↓（声带张力减弱）
    voice_quality 嗓音  — 疲劳↑（jitter/shimmer↑, HNR↓）⭐核心
    energy       能量  — 疲劳↓（呼吸变浅）
    spectral     频谱  — 辅佐（谱通量↓）
    mfcc         MFCC  — 辅佐
    spectral_tilt 谱倾斜 — 辅佐（H1-H2↓）
    formant     共振峰 — 辅佐（声道松弛）
    rhythm      节奏  — 疲劳↓（说话变慢）
    other       其他  — 辅助
"""

import re
from typing import List, Dict

# ━━━ 疲劳核心指标 ━━━

FATIGUE_CORE = {
    "F0": {
        "patterns": ["F0semitoneFrom27.5Hz"],
        "direction": "↓ 疲劳时下降",
        "desc": "基频 — 声带振动速率",
    },
    "jitter": {
        "patterns": ["jitterLocal"],
        "direction": "↑ 疲劳时上升",
        "desc": "基频微扰 — 声带控制不稳",
    },
    "shimmer": {
        "patterns": ["shimmerLocaldB"],
        "direction": "↑ 疲劳时上升",
        "desc": "振幅微扰 — 声带控制不稳",
    },
    "HNR": {
        "patterns": ["HNRdBACF"],
        "direction": "↓ 疲劳时下降",
        "desc": "谐噪比 — 嗓音噪声成分",
    },
    "loudness": {
        "patterns": ["loudness_sma3"],
        "direction": "↓ 疲劳时下降",
        "desc": "响度 — 说话能量",
    },
    "voice_rate": {
        "patterns": ["VoicedSegmentsPerSec"],
        "direction": "↓ 疲劳时下降",
        "desc": "每秒发声段数 — 语速/停顿",
    },
}

# ━━━ 特征分组（按 eGeMAPSv02 命名规范） ━━━

FEATURE_GROUPS: Dict[str, dict] = {
    "pitch": {
        "label": "基频 (F0)",
        "patterns": ["F0semitoneFrom27.5Hz"],
        "fatigue": "↓ 声带张力减弱",
        "weight": 4,  # 重要性权重（5=核心, 1=辅助）
    },
    "voice_quality": {
        "label": "嗓音质量",
        "patterns": ["jitterLocal", "shimmerLocaldB", "HNRdBACF"],
        "fatigue": "↑ jitter/shimmer, ↓ HNR — 声带疲劳核心证据",
        "weight": 5,
    },
    "energy": {
        "label": "能量/响度",
        "patterns": ["loudness_sma3"],
        "fatigue": "↓ 呼吸变浅",
        "weight": 4,
    },
    "spectral": {
        "label": "频谱特征",
        "patterns": ["spectralFlux_sma3"],
        "fatigue": "↓ 谱通量 — 说话更平",
        "weight": 2,
    },
    "mfcc": {
        "label": "MFCC",
        "patterns": ["mfcc1_sma3", "mfcc2_sma3", "mfcc3_sma3", "mfcc4_sma3"],
        "fatigue": "辅佐 — 个体差异大",
        "weight": 2,
    },
    "spectral_tilt": {
        "label": "频谱倾斜",
        "patterns": [
            "logRelF0-H1-H2", "logRelF0-H1-A3",
            "alphaRatio", "hammarbergIndex",
            "slopeV0-500", "slopeV500-1500",
        ],
        "fatigue": "H1-H2↓ — 声带闭合松",
        "weight": 3,
    },
    "formant": {
        "label": "共振峰 (F1-F3)",
        "patterns": [
            "F1frequency", "F1bandwidth", "F1amplitudeLogRelF0",
            "F2frequency", "F2bandwidth", "F2amplitudeLogRelF0",
            "F3frequency", "F3bandwidth", "F3amplitudeLogRelF0",
        ],
        "fatigue": "频率↓、带宽↑ — 声道肌肉松弛",
        "weight": 2,
    },
    "rhythm": {
        "label": "发声节奏",
        "patterns": [
            "VoicedSegmentsPerSec",
            "MeanVoicedSegmentLengthSec",
            "StddevVoicedSegmentLengthSec",
            "MeanUnvoicedSegmentLength",
            "StddevUnvoicedSegmentLength",
            "loudnessPeaksPerSec",
        ],
        "fatigue": "↓ 说话变慢/变少",
        "weight": 4,
    },
    "other": {
        "label": "其他",
        "patterns": [
            "equivalentSoundLevel_dBp",
            "spectralFluxV",
            "alphaRatioV",
            "hammarbergIndexV",
            "slopeV0-500",
            "slopeV500-1500",
            "spectralFluxUV",
            "alphaRatioUV",
            "hammarbergIndexUV",
        ],
        "fatigue": "辅助参考",
        "weight": 1,
    },
}


def match_features(feature_names: List[str], patterns: List[str]) -> List[str]:
    """从特征名列表中匹配指定模式"""
    matched = []
    for name in feature_names:
        for pat in patterns:
            if pat in name:
                matched.append(name)
                break
    return matched


def classify_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """将 88 维特征按分组归类

    Returns:
        {"pitch": ["F0_amean", "F0_stddev", ...], ...}
    """
    result = {}
    for group, info in FEATURE_GROUPS.items():
        result[group] = match_features(feature_names, info["patterns"])
    return result


def get_fatigue_features(feature_names: List[str]) -> List[str]:
    """返回所有疲劳相关特征名"""
    fat = set()
    for core_info in FATIGUE_CORE.values():
        for name in feature_names:
            for pat in core_info["patterns"]:
                if pat in name:
                    fat.add(name)
    return sorted(fat)


def short_name(full: str) -> str:
    """长特征名 → 简短可读名

    Example:
        F0semitoneFrom27.5Hz_sma3nz_amean → F0_mean
        jitterLocal_sma3nz_amean → jitter_mean
    """
    # 常见替换
    replacements = [
        ("F0semitoneFrom27.5Hz", "F0"),
        ("jitterLocal", "jitter"),
        ("shimmerLocaldB", "shimmer"),
        ("HNRdBACF", "HNR"),
        ("loudness_sma3", "loudness"),
        ("spectralFlux_sma3", "specFlux"),
        ("mfcc1_sma3", "MFCC1"),
        ("mfcc2_sma3", "MFCC2"),
        ("mfcc3_sma3", "MFCC3"),
        ("mfcc4_sma3", "MFCC4"),
        ("logRelF0-H1-H2", "H1H2"),
        ("logRelF0-H1-A3", "H1A3"),
        ("alphaRatioV", "alphaRatioV"),
        ("alphaRatioUV", "alphaRatioUV"),
        ("hammarbergIndexV", "hammarbergV"),
        ("EquivalentSoundLevel_dBp", "SPL"),
        ("VoicedSegmentsPerSec", "voiceSegPerSec"),
        ("MeanVoicedSegmentLengthSec", "meanVoiceSegLen"),
        ("F1frequency", "F1freq"),
        ("F2frequency", "F2freq"),
        ("F3frequency", "F3freq"),
        ("F1bandwidth", "F1bw"),
        ("F2bandwidth", "F2bw"),
        ("F3bandwidth", "F3bw"),
        ("_sma3nz_", "_"),
        ("_sma3_", "_"),
        ("_sma3nn_", "_"),
    ]
    short = full
    for old, new in replacements:
        short = short.replace(old, new)
    return short
