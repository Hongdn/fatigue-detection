# 疲劳识别 - 使用指南

> 版本：v1.1 | 日期：2026-07-27

## 一、环境

```bash
conda activate funasr              # Python 3.12
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    xgboost shap opensmile soundfile librosa gradio modelscope sortedcontainers
```

## 二、当前完成状态

```
step0_voiceprint (声纹提取+聚类)   ✅ 编码完成 (ERes2NetV2)
step1_preprocess (降噪 + VAD)     ✅ 编码完成
step2_features   (openSMILE 88维) ✅ 编码完成
step3_baseline   (特征级z-score)  🔜 代码保留，当前不接入
step4_regression (XGBoost+SHAP)   ✅ 编码 + 全量训练完成
step5_rules      (规则引擎+基线)  ✅ 编码完成
step6_demo       (Gradio 界面)    ✅ 可运行
```

已训练模型位于 `step4_regression/models/`：

| 文件 | 说明 |
|------|------|
| `model_arousal.json` | arousal 回归，CCC=0.873 |
| `model_exertion.json` | exertion 回归，CCC=0.617 |
| `feature_cols.json` | 88 维特征名 |

训练数据：ExpressiveSpeech (HuggingFace)，已下载于 `data/` (10 个 parquet，~12GB)。

## 三、快速体验：Gradio Demo

```bash
cd C:\Users\admin\Downloads\疲劳识别
conda activate funasr
python step6_demo/app.py
```

浏览器打开 `http://127.0.0.1:7860`，上传 WAV 音频 -> 自动分析。

输出：arousal/exertion 曲线图、状态分布饼图、逐段时间线、说��人分组。

内置示例音频：`output/REC-1_denoised.wav` 和 `REC-2_denoised.wav`。

## 四、Python API

### 4.1 端到端分析

```python
import sys
sys.path.insert(0, r'C:\Users\admin\Downloads\疲劳识别')
from step6_demo.app import analyze_audio

df, summary, fig, _ = analyze_audio("my_audio.wav")
# df: 含 段号/说话人/arousal/exertion/状态/置信度/arousal_z
```

### 4.2 分步调用

```python
from step1_preprocess.vad import vad_detect
from step0_voiceprint import SpeakerClusterer
from step4_regression.model import load_models
from step5_rules import SpeakerBaseline
from step5_rules.engine import classify_state, _compute_thresholds
import opensmile, soundfile as sf, numpy as np

# 加载 (只做一次)
models = load_models("step4_regression/models/")
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
clusterer = SpeakerClusterer(threshold=0.55)
baseline = SpeakerBaseline(global_mean=0.72, global_std=0.19)

# step1: VAD
audio, sr = sf.read("my_audio.wav")
if audio.ndim > 1: audio = audio.mean(axis=1)
segs = vad_detect(audio, sr, backend="silero")
seg_audios = [audio[int(s.start_sec*sr):int(s.end_sec*sr)] for s in segs]

# step0: 声纹聚类
speaker_ids = clusterer.fit_predict(seg_audios, sr)

# step2: openSMILE
features = [smile.process_signal(s.astype(np.float32), sr).iloc[0].values for s in seg_audios]

# step4: XGBoost
X = np.array(features, dtype=np.float32)
arousal = np.clip(models["arousal"].predict(X), 0, 1)
exertion = np.clip(models["exertion"].predict(X), 0, 1)

# step5: 个体基线 + 状态判定
at = _compute_thresholds(arousal)
for i, (a, e) in enumerate(zip(arousal, exertion)):
    sid = speaker_ids[i]
    br = baseline.normalize(sid, a, e)
    baseline.update(sid, a, e)
    state, conf, reason = classify_state(a, e, at, et)
    print(f"{sid}: a={a:.3f} e={e:.3f} z={br.arousal_z:.1f} -> {state}")
```

## 五、处理流程

```
音频 -> step1 VAD分段 -> step0 声纹聚类 -> step2 openSMILE 88维
                                                    |
                                         step4 XGBoost 双任务回归
                                                    |
                                         step5 规则引擎 + 个体基线
                                                    |
                                         状态 + SHAP溯源 + 说话人分组
```

- **step0**: ERes2NetV2 (达摩院, 20万中文说话人预训练, EER 0.61%)
- **step1**: Silero VAD 分段
- **step2**: openSMILE eGeMAPSv02 88维 (F0/jitter/shimmer/HNR/MFCC/共振峰)
- **step4**: XGBoost 双任务回归 (CCC_arousal=0.873, CCC_exertion=0.617)
- **step5**: 规则引擎 + SpeakerBaseline (输出级个体归一化, 无注册冷启动)

## 六、注意事项

1. 当前模型用 ExpressiveSpeech 训练，标注是 emotional arousal，不是真实操作疲劳。用于技术验证。
2. step3 (特征级 z-score) 当前不接入。XGBoost 是树模型，对特征尺度不敏感。
3. 音频需为 16kHz 单声道 WAV。其他格式会自动重采样。
4. 结果仅供参考，不得直接用于安全决策。

## 七、相关文档

| 文档 | 内容 |
|------|------|
| `README.md` | 项目总览、技术栈、进度表 |
| `docs/系统架构与选型方案.md` | 架构设计和各层选型理由 |
| `docs/维度回归模块-XGBoost设计.md` | step4 详细设计 |
| `docs/说话人基线对齐方案.md` | step0 声纹 + step5 个体基线设计 |
| `output/step4_full_training_results.xlsx` | 全量 28k 训练详细结果 |
