# 疲劳识别 - 使用指南

> 版本：v1.4 | 日期：2026-07-29

## 一、环境

```bash
conda activate funasr              # Python 3.12
pip install -r requirements.txt    # 或: -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 二、当前完成状态

```
step0_voiceprint (声纹提取+持久化声纹库) ✅ 编码完成 (ERes2NetV2 + VoiceprintDB)
step1_preprocess (降噪 + VAD)     ✅ 编码完成
step2_features   (openSMILE 88维) ✅ 编码完成
step3_baseline   (特征级z-score)  🔜 代码保留，当前不接入
step4_regression (XGBoost+SHAP)   ✅ 编码 + 全量训练完成
step5_rules      (规则引擎+基线)  ✅ 编码完成（SpeakerBaseline 持久化）
step6_demo       (Gradio 界面)    ✅ 可运行（声纹库管理Tab）
```

已训练模型位于 `step4_regression/models/`：

| 文件 | 说明 |
|------|------|
| `model_arousal.json` | arousal 回归，CCC=0.873 |
| `model_exertion.json` | exertion 回归，CCC=0.617 |
| `feature_cols.json` | 88 维特征名 |

训练数据：ExpressiveSpeech (HuggingFace: FreedomIntelligence/ExpressiveSpeech)，已下载于 `data/` (10 个 parquet，~12GB)。

## 三、快速体验：Gradio Demo

```bash
cd C:\Users\admin\Downloads\疲劳识别
conda activate funasr
python step6_demo/app.py
```

浏览器打开 `http://127.0.0.1:7860`，上传 WAV 音频 -> 自动分析。

### 界面功能

- **8 个实时指标卡片**：文件数、总时长、VAD 段数、arousal/exertion 均值、疲劳段数、兴奋段数、处理耗时
- **进度条**：分步显示 VAD -> 特征提取 -> XGBoost 推理 -> 声纹匹配 -> 状态判定
- **Tabs 结果区**：概览 Tab（可视化图表 + 分析摘要）+ 逐段详情 Tab + 声纹库管理 Tab
- **四象限图表**：arousal/exertion 时间线、状态分布饼图、状态时间线、逐说话人唤醒度/费力程度对比
- **Trace 日志**：每次推理自动生成 `trace_id`，结构化 JSON 保存至 `output/.traces/`
- **声纹库管理**：查看已注册说话人、重命名、合并、删除（联动清理 SpeakerBaseline）
- **持久化声纹库**：跨请求关联同一说话人，个体基线持续积累（`output/voiceprint_db.json` + `output/speaker_baseline.json`）

### 内置示例音频

`output/REC-1_denoised.wav` ~ `REC-4_denoised.wav`（四段管制录音降噪后）。

## 四、Python API

### 4.1 端到端分析

```python
import sys
sys.path.insert(0, r'C:\Users\admin\Downloads\疲劳识别')
from step6_demo.app import analyze_audio

df, summary, fig, _ = analyze_audio("my_audio.wav")
# df: 含 段号/说话人/arousal/exertion/状态/置信度
# summary: Markdown 分析摘要（含 trace_id、模型版本、阈值信息）
# fig: matplotlib Figure（四象限图表）
# 可选参数 progress=None，传入 gr.Progress() 可在 Gradio 中显示进度条
```

### 4.2 分步调用

```python
from step1_preprocess.vad import vad_detect
from step0_voiceprint import SpeakerClusterer, VoiceprintDB
from step4_regression.model import load_models
from step5_rules import SpeakerBaseline
from step5_rules.engine import classify_state, _compute_thresholds
import opensmile, soundfile as sf, numpy as np

# 加载 (只做一次)
models = load_models("step4_regression/models/")
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
extractor = SpeakerClusterer(threshold=0.55)  # 仅用于 embedding 提取
vpdb = VoiceprintDB("output/voiceprint_db.json")
vpdb.load()
baseline = SpeakerBaseline(global_mean=0.72, global_std=0.19)
baseline.load("output/speaker_baseline.json")

# step1: VAD
audio, sr = sf.read("my_audio.wav")
if audio.ndim > 1: audio = audio.mean(axis=1)
segs = vad_detect(audio, sr, backend="silero")
seg_audios = [audio[int(s.start_sec*sr):int(s.end_sec*sr)] for s in segs]

# step0: 逐段 embedding → VoiceprintDB 匹配
embeddings = extractor.extract_embeddings(seg_audios, sr)
speaker_ids = [vpdb.match_and_update(emb) for emb in embeddings]

# step2: openSMILE
features = [smile.process_signal(s.astype(np.float32), sr).iloc[0].values for s in seg_audios]

# step4: XGBoost
X = np.array(features, dtype=np.float32)
arousal = np.clip(models["arousal"].predict(X), 0, 1)
exertion = np.clip(models["exertion"].predict(X), 0, 1)

# step5: 个体基线 + 状态判定
at = _compute_thresholds(arousal)
et = _compute_thresholds(exertion)
for i, (a, e) in enumerate(zip(arousal, exertion)):
    sid = speaker_ids[i]
    br = baseline.normalize(sid, a, e)
    baseline.update(sid, a, e)
    state, conf, reason = classify_state(a, e, at, et)
    print(f"{sid}: a={a:.3f} e={e:.3f} -> {state} ({conf})")

# 持久化
vpdb.save()
baseline.save("output/speaker_baseline.json")
```

## 五、处理流程

```
音频 -> step1 VAD分段 -> step0 逐段embedding提取 -> VoiceprintDB匹配
                                                         |
                                              step2 openSMILE 88维
                                                         |
                                              step4 XGBoost 双任务回归
                                                         |
                                              step5 规则引擎 + 个体基线(持久化)
                                                         |
                                              状态 + SHAP溯源 + 持久speaker_id
```

- **step0**: ERes2NetV2 (达摩院, 20万中文说话人预训练, EER 0.61%) + VoiceprintDB 持久化声纹库
- **step1**: Silero VAD 分段
- **step2**: openSMILE eGeMAPSv02 88维 (F0/jitter/shimmer/HNR/MFCC/共振峰)
- **step4**: XGBoost 双任务回归 (CCC_arousal=0.873, CCC_exertion=0.617)
- **step5**: 规则引擎 + SpeakerBaseline (输出级个体归一化, 持久化基线, 冷启动降级)

## 六、注意事项

1. 当前模型用 ExpressiveSpeech 训练，标注是 emotional arousal，不是真实操作疲劳。用于技术验证。
2. step3 (特征级 z-score) 当前不接入。XGBoost 是树模型，对特征尺度不敏感。
3. 音频需为 16kHz 单声道 WAV。其他格式会自动重采样。
4. 结果仅供参考，不得直接用于安全决策。

## 七、测试

```bash
pytest tests/ -v                 # 34 个单元测试
```

## 八、相关文档

| 文档 | 内容 |
|------|------|
| `README.md` | 项目总览、技术栈、进度表 |
| `docs/系统架构与选型方案.md` | 架构设计和各层选型理由 |
| `docs/维度回归模块-XGBoost设计.md` | step4 详细设计 |
| `docs/说话人基线对齐方案.md` | step0 声纹 + step5 个体基线设计 |
| `docs/声纹库设计方案.md` | 持久化声纹库：跨文件说话人关联（已实现 v0.3） |
| `output/step4_full_training_results.xlsx` | 全量 28k 训练详细结果 |
