# 管制员工作状态分析

> 课题来源：中兵 & 大兴机场运行管理部 AI 需求课题 5.2《智能语音识别分析》
> 聚焦：通过空管电台通话录音，分析管制员工作状态（兴奋/稳定/疲劳），辅助值班长疲劳风险管理
> 版本：v2.4 | 2026-07-29

---

## 一、处理流程

| 步骤 | 模块 | 功能 | 状态 |
|------|------|------|:--:|
| 0 | **声纹识别** | ERes2NetV2 提取 192d embedding → VoiceprintDB 逐段匹配 → 持久化 speaker_id | ✅ |
| 1 | **前置处理** | MossFormerGAN 降噪 + Silero VAD 分段 + 质量门控 | ✅ |
| 2 | **特征提取** | openSMILE eGeMAPSv02 88 维（F0/jitter/shimmer/HNR/MFCC/共振峰） | ✅ |
| 3 | **基线对齐** | 特征级 z-score（代码保留，XGBoost 树模型对尺度不敏感，当前不接入） | 🔜 |
| 4 | **维度回归** | XGBoost 双任务回归：arousal (CCC=0.873) + exertion (CCC=0.617)，SHAP 可解释 | ✅ |
| 5 | **状态判定** | 规则引擎 + SpeakerBaseline 个体基线（自适应阈值、冷启动降级、持久化） | ✅ |
| 6 | **Demo** | Gradio Web：批量上传、指标卡片、Tabs、声纹库管理、Trace 日志 | ✅ |

**核心思路**：不走情绪分类路线（Emotion2Vec+ 已实测失效），走生理声学 XGBoost + SHAP 可解释路线。双维度特征分离——arousal 靠 F0/频谱，exertion 靠 shimmer/jitter/嗓音质量。

训练数据：ExpressiveSpeech 28k 条（中英双语），已全量训练。当前模型用 emotional arousal 标注，真实疲劳数据需二期补齐。

---

## 二、技术栈

| 层 | 选型 | 状态 |
|----|------|:--:|
| 降噪 | ClearerVoice-Studio MossFormerGAN_SE_16K | ✅ |
| VAD | Silero VAD（默认）/ TEN VAD（可选） | ✅ |
| 声纹嵌入 | ERes2NetV2（达摩院，20万中文说话人预训练，EER 0.61%） | ✅ |
| 持久化声纹库 | VoiceprintDB：逐段匹配、自适应阈值、EMA 质心、match_history | ✅ |
| 目标说话人提取 | ClearerVoice 纯音频 TSE（8kHz） | 🔜 |
| 声纹校验 | ECAPA-TDNN（后置，二期） | 🔜 |
| 声学特征 | openSMILE eGeMAPSv02（88 维，9 组分类） | ✅ |
| 个体基线 | SpeakerBaseline：per-speaker 运行统计 + 群体兜底 + JSON 持久化 | ✅ |
| 回归模型 | XGBoost 双任务回归 + SHAP 可解释 | ✅ |
| 状态判定 | 规则引擎 + percentile 自适应阈值 + SHAP 溯源 | ✅ |
| Demo | Gradio（指标卡片 / Progress / Tabs / 声纹库管理 / Trace） | ✅ |

---

## 三、文件结构

```
疲劳识别/
├── step0_voiceprint/           声纹识别 + 持久化声纹库
│   ├── cluster.py              ERes2NetV2 嵌入提取 + 余弦聚类
│   ├── db.py                   VoiceprintDB 持久化声纹库
├── step1_preprocess/           前置处理
│   ├── denoise.py              MossFormerGAN 降噪 + 自动重采样
│   ├── vad.py                  Silero / TEN VAD 多后端
│   ├── tse.py                  TSE 目标提取（二期）
│   └── quality.py              四级质量门控
├── step2_features/             特征提取
│   └── extractor.py            openSMILE eGeMAPSv02 88 维
├── step3_baseline/             基线对齐（当前不接入）
├── step4_regression/           维度回归
│   ├── model.py                XGBoost 双任务 + SHAP + 持久化
│   ├── dataset.py              Parquet 加载器
│   └── models/                 已训练模型（JSON）
├── step5_rules/                状态判定
│   ├── engine.py               规则引擎 + SHAP 溯源
│   └── speaker_baseline.py     个体基线管理 + JSON 持久化
├── step6_demo/                 Gradio Demo
├── checkpoints/                MossFormerGAN 模型权重
├── output/                     降噪结果 / 训练结果 / 声纹库 JSON
├── docs/                       方案文档（8 篇）
├── figures/                    架构图（4 张）
├── scripts/                    训练脚本
├── data/                       训练数据（10 parquet, ~12GB）
└── references/                 外部参考
```

---

## 四、核心文档

| 文件 | 内容 |
|------|------|
| `docs/系统架构与选型方案.md` | 六阶段架构总览、各层技术选型与理由 |
| `docs/工作状态分析模块-技术实现方案.md` | 状态分析完整实现方案 |
| `docs/课题5.2-工作状态分析-核心难点分析.md` | 六个核心难点拆解与破局 |
| `docs/一期MVP实现方案.md` | MVP 验证计划（5 个假设，9 周规划） |
| `docs/前置处理层-模块设计.md` | 降噪 → VAD → TSE 详细设计 |
| `docs/维度回归模块-XGBoost设计.md` | XGBoost + SHAP 双任务回归详细设计 |
| `docs/说话人基线对齐方案.md` | per-speaker 基线：无注册自学习，冷启动 → 切换 |
| `docs/声纹库设计方案.md` | VoiceprintDB：逐段匹配、自适应阈值、质心更新（已实现 v0.3） |

**架构图**：`figures/系统架构图.svg`、`figures/一期MVP模块架构图.svg`、`figures/工作状态分析实现流程图.svg`、`figures/竞品定位矩阵.svg`

**关键产出**：`output/step4_full_training_results.xlsx`（全量训练 7 sheets）、`figures/opensmile_fatigue_compare.png`（REC-1~4 特征对比）

**参考**：`docs/国内竞品分析报告.md`、`references/中兵&大兴机场运行管理部AI需求课题汇总.xlsx`

---

## 五、全量训练结果

| 指标 | arousal | exertion |
|------|---------|----------|
| 样本数 | 28,190 | 28,190 |
| 测试 CCC | **0.873** | **0.617** |
| 测试 R² | 0.782 | 0.467 |
| 测试 Pearson | 0.885 | 0.686 |
| 主导特征组 | F0 基频 (0.108) | loudness 响度 (0.014) |
| 次要特征组 | MFCC (0.068) | shimmer 振幅 (0.010) |

---

## 六、MVP 验证状态

| # | 假设 | 通过标准 | 状态 |
|---|------|---------|:--:|
| H1 | 语音特征与 arousal 相关 | CCC > 0.5 | ✅ 0.873 |
| H2 | arousal + exertion 映射三态 | 状态可分 | ✅ |
| H3 | 个体基线有效 | 个体内优于跨人 | 🔜 待真实数据 |
| H4 | 端到端 pipeline 可跑通 | Demo 可演示 | ✅ |
| H5 | 降噪是硬前提 | 降噪后特征更稳定 | ✅ |

---

## 七、已知结论与限制

1. **Emotion2Vec+ 分类失效** — 通用情感模型不适用于管制职业语音
2. **openSMILE 生理声学特征有效** — 28k 条训练验证，双维度特征分离
3. **谱减法降噪不可用** — 需 ClearerVoice MossFormerGAN
4. **XGBoost + SHAP 成立** — arousal CCC=0.873，特征模式符合理论
5. **jitter/shimmer 对录音质量极敏感** — 不能脱离录音质量单独解读
6. **当前模型用 emotional arousal 训练** — 真实管制疲劳数据需二期补齐
7. **声纹属敏感个人信息** — 需遵守《个人信息保护法》，提供删除选项

---

*README.md — v2.4，2026-07-29*
