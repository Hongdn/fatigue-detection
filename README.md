# 管制员工作状态分析 — 项目文档

> 课题来源：中兵 & 大兴机场运行管理部 AI 需求课题 5.2《智能语音识别分析》
> 聚焦：进阶功能二 — 管制员工作状态（兴奋/稳定/疲劳）分析
> 版本：v2.2 | 日期：2026-07-27

---

## 一、项目概述

通过空管电台通话录音，分析管制员的工作状态（兴奋/稳定/疲劳），辅助值班长进行疲劳风险管理。录音场景为空管+飞行员混合对话。

**核心方案**（v2.2 — 全链路编码并验证）：

```
step0_voiceprint      step1_preprocess     step2_features         step4_regression       step5_rules              step6_demo
ERes2NetV2声纹提取 +  MossFormerGAN降噪 → openSMILE 88维 → XGBoost+SHAP双任务 → 规则引擎+个体基线判状态 → Gradio Web分析
持久化声纹库匹配      + VAD切分           + 文本认知指标(二期) → arousal/exertion  → 兴奋/稳定/疲劳+溯源    → 指标卡片+声纹库管理Tab
                                                                                    +逐说话人对比+Trace日志

          step3_baseline（特征级z-score: 代码保留，当前不接入。XGBoost树模型对尺度不敏感）

**核心创新点**：
- 不走情绪分类路线（Emotion2Vec+ 分类已实测失效）
- arousal + exertion 双维度回归，SHAP 可溯源告警
- 声纹识别自动分组说话人（ERes2NetV2, EER 0.61%）
- 用 ExpressiveSpeech（28k 条）全量训练，CCC_arousal=0.873, CCC_exertion=0.617
- 双维度特征模式分离：arousal 靠 F0/频谱，exertion 靠 shimmer/jitter/嗓音质量

---

## 二、文件清单

```
疲劳识别/
├── README.md
├── data/                        ← 训练数据（10个parquet, ~12GB）
├── step0_voiceprint/            ← ⓪声纹识别（已编码✅）
│   ├── cluster.py               ERes2NetV2提取+余弦聚类
│   └── db.py                    VoiceprintDB 持久化声纹库
├── step1_preprocess/            ← ①前置处理（已编码✅）
│   ├── denoise.py              MossFormerGAN降噪 + 自动重采样
│   ├── vad.py                  Silero/TEN VAD 多后端
│   ├── tse.py                  TSE目标提取（二期，8k桥接）
│   ├── quality.py              四级质量门控
│   └── pipeline.py             串联入口
├── step2_features/             ← ②特征提取（已编码✅）
│   ├── extractor.py            openSMILE eGeMAPSv02 88维
│   ├── feature_set.py           9组分类 + 疲劳核心27维
│   └── pipeline.py             接step1输出
├── step3_baseline/             ← ③基线对齐（已编码✅）
│   ├── aligner.py              个体z-score + 群体兜底 + 持久化
│   └── pipeline.py             接step2输出
├── step4_regression/           ← ④维度回归（已编码+全量训练✅）
│   ├── model.py                XGBoost双任务 + SHAP解释 + 持久化
│   ├── dataset.py              Parquet加载器
│   ├── pipeline.py             训练/推理/CV
│   └── models/                 model_arousal.json + model_exertion.json
├── step5_rules/                ← ⑤状态判定（已编码✅）
│   ├── engine.py               规则引擎 + SHAP溯源
│   ├── speaker_baseline.py     个体基线管理（无注册自学习）
│   └── pipeline.py             端到端判定
├── checkpoints/                 ← MossFormerGAN 模型权重
├── output/                      ← 降噪结果 + 训练结果Excel
├── docs/                        ← 方案文档（8篇）
├── figures/                     ← 架构图（4张）
├── scripts/                     ← 训练脚本
└── references/                  ← 外部参考
```

### 核心方案文档

| 文件 | 内容 |
|------|------|
| `docs/系统架构与选型方案.md` | 六阶段架构总览、各层技术选型与理由 |
| `docs/工作状态分析模块-技术实现方案.md` | 聚焦状态分析的完整实现方案（pipeline、特征、模型、规则） |
| `docs/课题5.2-工作状态分析-核心难点分析.md` | 六个核心难点拆解与破局思路 |
| `docs/一期MVP实现方案.md` | MVP 验证计划（5个假设、9周规划） |
| `docs/前置处理层-模块设计.md` | 降噪→VAD→TSE 前置处理详细设计 |
| `docs/维度回归模块-XGBoost设计.md` | XGBoost+SHAP 双任务回归详细设计 |
| `docs/说话人基线对齐方案.md` | per-speaker 基线自学习：无注册，冷启动→切换 |
| `docs/声纹库设计方案.md` | 持久化声纹库：跨文件说话人关联 + 个体基线积累（已实现 v0.3） |

### 架构图

| 文件 | 内容 |
|------|------|
| `figures/系统架构图.svg` | 六阶段流水线：数据采集→前置处理→特征提取→模型推理→融合决策→输出 |
| `figures/一期MVP模块架构图.svg` | MVP 五模块：数据准备→降噪+特征→基线→回归→判定 |
| `figures/工作状态分析实现流程图.svg` | 状态分析模块内部数据流 |

### 验证实验

| 文件 | 内容 |
|------|------|
| `figures/opensmile_fatigue_compare.png` | REC-1~4 四段管制录音特征对比图 |
| `output/step4_full_training_results.xlsx` | 全量训练详细结果（7 sheets，含指标/SHAP/特征组/对比） |
| `output/step5_state_results.csv` | 500条端到端状态判定示例 |

### 参考文档

| 文件 | 内容 |
|------|------|
| `docs/国内竞品分析报告.md` | 国内空管/航司机疲劳监控竞品全景 |
| `figures/竞品定位矩阵.svg` | 竞品技术路线×成熟度定位图 |
| `references/中兵&大兴机场运行管理部AI需求课题汇总.xlsx` | 课题原始需求（含5.2原文） |

---

## 三、技术栈（v2.0）

| 层 | 选型 | 状态 |
|----|------|------|
| 降噪 | ClearerVoice-Studio MossFormerGAN_SE_16K | ✅ |
| VAD | Silero VAD（默认）/ TEN VAD（可选） | ✅ |
| 目标说话人提取 | ClearerVoice 纯音频 TSE（8kHz，二期） | 🔜 |
| 声纹校验 | ECAPA-TDNN（后置，二期） | 🔜 |
| 持久化声纹库 | VoiceprintDB：逐段匹配+自适应阈值+EMA质心+match_history | ✅ |
| 生理声学特征 | openSMILE eGeMAPSv02（88维，9组分类） | ✅ |
| 个体基线对齐 | z-score + 群体均值兜底 + 持久化 | ✅ |
| 个体基线对齐 | per-speaker 运行均值/标准差，自适应权重混合 | ✅ |
| 维度回归 | XGBoost 双任务回归 + SHAP 可解释 | ✅ |
| 状态判定 | 规则引擎 + percentile自适应阈值 + SHAP溯源 | ✅ |
| Demo | Gradio 端到端界面（指标卡片/Progress/Tabs/Trace） | ✅ |

---

## 四、核心方案要点

### 4.1 技术路线修正过程

| 阶段 | 尝试 | 结论 |
|------|------|------|
| 初始 | wav2vec2 + TCN 多信号融合 | 需验证 |
| 实测验证 | Emotion2Vec+ 分类输出 | **失效** |
| 方向修正 | openSMILE 生理声学特征回归 | **有效** |
| 数据集选定 | ATC Fatigue Corpus（付费放弃）→ ExpressiveSpeech | **确定** |
| 训练方案 | 2000条验证 → 28190条全量训练 | **完成** |
| 全链路集成 | step1→step2→step3→step4→step5 | **跑通** |

### 4.2 全量训练结果

| 指标 | arousal | exertion |
|------|---------|----------|
| 样本数 | 28,190 | 28,190 |
| 测试 CCC | **0.873** | **0.617** |
| 测试 R² | 0.782 | 0.467 |
| 测试 Pearson | 0.885 | 0.686 |
| 训练时间 | 2.2s | 2.0s |
| 主导特征组 | F0基频 (0.108) | loudness响度 (0.014) |
| 次要特征组 | MFCC (0.068) | shimmer振幅 (0.010) |

### 4.3 当前 MVP 五个假设

| # | 假设 | 通过标准 | 状态 |
|---|------|---------|------|
| H1 | 语音特征与 arousal 相关 | CCC > 0.5 | ✅ 0.873 |
| H2 | arousal+exertion 映射三态 | 状态可分 | ✅ 规则引擎已跑通 |
| H3 | 个体基线有效 | 个体内优于跨人 | 🔜 待真实数据验证 |
| H4 | 端到端 pipeline 可跑通 | Demo 可演示 | ✅ 已实现 |
| H5 | 降噪是硬前提 | 降噪后特征更稳定 | ✅ 已实测 |

### 4.4 已知前提

- Emotion2Vec+ 分类路线不可行（已实测证实）
- 情绪分类 ≠ 工作状态分析，是两个不同的概念空间
- 当前模型用 emotional arousal 训练，真实疲劳数据需二期补齐

---

## 五、已验证的结论与教训

1. **openSMILE 生理声学特征有效**：四段管制录音实测，全量 28k 条训练验证
2. **Emotion2Vec+ 分类失效**：通用情感模型不适用于管制职业语音
3. **谱减法降噪不可用**：需 ClearerVoice MossFormerGAN
4. **XGBoost+SHAP 链路成立**：arousal CCC=0.873，特征模式符合理论
5. **双维度特征分离**：arousal 靠 F0，exertion 靠 shimmer/jitter，两个模型独立
6. **jitter/shimmer 对录音质量极敏感**——不能脱离录音质量单独解读

---

## 六、当前进度

| 步骤 | 内容 | 状态 |
|------|------|------|
| step0 | ERes2NetV2 声纹提取 + VoiceprintDB 持久化声纹库 | ✅ |
| step1 | MossFormerGAN降噪 + Silero VAD + 质量门控 | ✅ |
| step2 | openSMILE eGeMAPSv02 88维特征提取 | ✅ |
| step3 | 个体 z-score 基线对齐 + 群体兜底 | 🔜 保留代码（XGBoost树模型对尺度不敏感，当前不接入） |
| step4 | XGBoost+SHAP 双任务回归（28k条全量训练） | ✅ |
| step5 | 规则引擎 + SpeakerBaseline个体基线 + SHAP溯源 | ✅ |
| step6 | Gradio Demo | ✅ |

## 七、文件状态

| 文件 | 状态 |
|------|------|
| step0_voiceprint/ | v2.0 声纹提取 + VoiceprintDB 持久化声纹库 |
| step1_preprocess/ | v1.0 编码通过 |
| step2_features/ | v1.0 编码通过 |
| step3_baseline/ | v1.0 编码通过 |
| step4_regression/ | v2.2 编码+全量训练完成 |
| step5_rules/ | v1.1 编码+验证通过（SpeakerBaseline 持久化） |
| step6_demo/ | v3.0 声纹库管理Tab + 批量上传 + 指标卡片 + Trace |
| docs/系统架构与选型方案.md | v1.2 有效 |
| docs/一期MVP实现方案.md | v1.5 有效 |
| docs/维度回归模块-XGBoost设计.md | v2.2 有效 |
| docs/国内竞品分析报告.md | 参考 |

---

*README.md — v2.3，2026-07-29*
