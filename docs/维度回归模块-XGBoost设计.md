# 维度回归模块 — XGBoost + SHAP 设计文档

> 对应系统架构第 4 层：回归 arousal + exertion 连续值
> 版本：v2.2 | 日期：2026-07-21

---

## 一、模块定位

```
step1_preprocess     step2_features        step3_baseline        step4_regression
音频 → 降噪+VAD  →  openSMILE 88维  →  个体 z-score  →  XGBoost 双任务回归
                                                              ├── arousal ∈ [0,1]
                                                              └── exertion ∈ [0,1]
                                                              └── SHAP 特征贡献 → 可解释告警
```

输入 step3_baseline 的 z-scored DataFrame，输出每段的 arousal 和 exertion 连续值，同时输出 SHAP 特征贡献度用于告警溯源。结果喂给下一步规则引擎判三态。

---

## 二、核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 模型 | **XGBoost** 双任务回归 | 表格数据（88维）上的 SOTA；训练秒级；原生特征重要度 |
| 可解释性 | **SHAP** | 对每次预测输出每个 eGeMAPS 维度的贡献值——告警可溯源到具体声学指标 |
| 主链路 | **eGeMAPS 88维不变** | 生理声学因果链：jitter→疲劳→告警，航空安全场景必须可解释 |
| 否决 Wav2Vec2 | **不换端到端方案** | 黑盒不可解释 + 英语播客→中文 VHF 域迁移风险 |
| 评价指标 | **CCC**（一致性相关系数） | SER 回归标准指标，同时衡量相关性和偏差 |
| exertion 标签 | **DeEAR score_nature**（说话自然度） | 与 arousal 相关系数仅 0.36，独立维度；疲劳时说话更费力→自然度下降 |
| 训练数据 | **ExpressiveSpeech**（HuggingFace，parquet 嵌入音频） | ~14k 条，51h，中英双语；DeEAR 四维分数，音频直接嵌在 parquet 中无需单独下载 |
| 训练方式 | **单句级独立回归** | 每条语音独立；后续管制员连续录音可加入时间窗口特征 |

### 2.1 DeEAR 四维分数的选取

ExpressiveSpeech 每条数据有四个 DeEAR 分数：

| 分数 | 含义 | 与 arousal 相关系数 | 用途 |
|------|------|-------------------|------|
| `score_arousal` | 唤醒度（激动程度） | 1.00 | **arousal 回归目标** |
| `score_prosody` | 韵律丰富度 | 0.92 | ❌ 与 arousal 高度冗余 |
| `score_nature` | 自然度（机械 ← → 自然） | 0.36 | **exertion 回归目标**（独立维度） |
| `score_expressive` | 综合表现力 | 0.86 | ❌ 与 arousal 高度冗余 |

**关键发现**：nature 与 arousal 几乎独立（r=0.36），疲劳时说话更费力→更不自然→nature↓，是理想的 exertion 代理标签。prosody 和 expressive 因与 arousal 高度重合而弃用。

### 2.2 为什么不用 Wav2Vec2 / 端到端方案

| 维度 | Wav2Vec2 等端到端 | XGBoost + eGeMAPS（选定） |
|------|-------------------|-------------------------|
| 可解释性 | ❌ 黑盒——不知道为什么预测 0.32 | ✅ SHAP 输出"jitter 贡献 +0.15，shimmer 贡献 +0.12" |
| 域适配 | ❌ 英语播客→中文 VHF，需大量标注 | ✅ jitter 就是 jitter，跨语言跨场景含义一致 |
| 验证方式 | ❌ 无法逐特征校验 | ✅ 可和人工听感对齐（"这段声音沙哑，jitter 确实高"） |
| 告警溯源 | ❌ "模型说疲劳" | ✅ "jitter 偏离个体基线 2.3σ，shimmer 偏离 1.8σ" |
| 精度 | 更高（CCC ~0.7） | 中等但可解释（CCC ~0.5-0.6，可通过特征工程提升） |
| 资源 | GPU 必需 | **CPU 可用，训练秒级** |

**航空安全场景的约束决定了不可解释的模型不可用。精度差距可通过特征工程（加入 VAD 段长/间隔等时间特征）缩小。**

---

## 三、模型架构

```
输入: step3 输出的 z-scored 88维 eGeMAPS
  │
  ├─── 直接作为 XGBoost 的输入特征
  │
  ▼
┌─────────────────────────────────────────────┐
│  XGBoost Regressor × 2（独立训练）           │
│                                              │
│  XGBoost-arousal:                            │
│    n_estimators=200, max_depth=6             │
│    learning_rate=0.05, subsample=0.8         │
│    objective='reg:squarederror'              │
│    → 输出 arousal ∈ [0, 1]                  │
│                                              │
│  XGBoost-exertion（代理标签训练）:            │
│    同上结构                                   │
│    → 输出 exertion ∈ [0, 1]                  │
│                                              │
│  共同输出:                                    │
│    ├── feature_importances_ （原生）          │
│    └── SHAP values           （可解释层）     │
└─────────────────────────────────────────────┘
```

**参考实现**：
- XGBoost：https://github.com/dmlc/xgboost | https://xgboost.readthedocs.io/
- SHAP：https://github.com/shap/shap
- eGeMAPS+XGBoost 案例：https://github.com/Hein-HtetSan/depression-analysis-model（openSMILE→XGBoost 抑郁检测，链路最接近）

### 3.1 为什么单独训练两个 XGBoost 而非共享

XGBoost 没有"共享 backbone"的概念。双任务共享表征的优势（如 TCN 中）在表格数据上不明显——88 维已经高度结构化，每个维度的物理含义独立。两个独立模型训练速度更快（可并行），且各自的特征重要度互不干扰。

---

## 四、训练策略

### 4.1 训练数据：ExpressiveSpeech

| 项 | 内容 |
|----|------|
| 数据 | ExpressiveSpeech（FreedomIntelligence, HuggingFace） |
| 格式 | Parquet 文件，音频直接作为 WAV bytes 嵌入，无需单独下载音频文件 |
| 规模 | ~14,000 条，51h，中英双语（约 1:1） |
| 标注 | DeEAR 四维连续分数：score_arousal / score_prosody / score_nature / score_expressive |
| 采样率 | 16kHz 或 44.1kHz（已含重采样逻辑） |
| 下载 | 直链下载 parquet 文件（共 10 个分片，12.3GB），建议先下 2-3 个分片（~3GB） |
| 预处理 | step2 FeatureExtractor → 逐段提取 eGeMAPS 88维 → step3 z-score |
| 切分 | 80% 训练 / 20% 测试，随机 split |
| arousal 标签 | DeEAR `score_arousal` ∈ [0, 1] |
| exertion 标签 | DeEAR `score_nature` ∈ [0, 1]（自然度→费力程度逆映射） |
| 模型 1 | XGBoost 回归 arousal（score_arousal 为目标） |
| 模型 2 | XGBoost 回归 exertion（score_nature 为目标） |
| 调参 | GridSearchCV：n_estimators=[100,200,300], max_depth=[4,6,8] |
| 训练时间 | **CPU 20-60 秒**（单 parquet 分片 ~2800 条） |
| 输出 | `model_arousal.json` + `model_exertion.json`（XGBoost 原生格式） |

### 4.2 已确认的初步结果（100 条验证）

| 指标 | 结果 |
|------|------|
| CCC_arousal（80 train / 20 test） | 0.626 |
| R²（测试集） | 0.375 |
| SHAP top 特征 | F0 方差、共振峰带宽、响度——符合 emotional arousal 理论 |
| 结论 | openSMILE 88维特征确实携带 arousal 信号，小样本即获得有意义的相关性 |

### 4.3 完整数据集（待运行）

当下载 ≥2 个 parquet 分片（~5000 条）后，扩大训练集至 4000+ 条，预期 CCC 可提升至 >0.7。

### 验证目标

| 指标 | 目标 |
|------|------|
| CCC_arousal | > 0.6（ExpressiveSpeech DeEAR score_arousal） |
| CCC_exertion | > 0.4（ExpressiveSpeech DeEAR score_nature，nature 方差小故放低门槛） |
| SHAP top-5 特征 | arousal 模型以 F0/频谱为主；exertion 模型以嗓音质量为主（与理论预期一致） |

---

## 五、资源估算

```
XGBoost 不做矩阵乘法做决策树分裂：
  数据: ~14k 行 × 88 列 = ~5MB（ExpressiveSpeech 全量）
  训练: CPU 30-120 秒
  推理: 每段 < 1ms
  内存: < 200MB
  硬盘: 模型文件 < 2MB（两个模型各 <1MB）
```

不需要 GPU。任何机器都够。

---

## 六、损失函数与评估

### 训练损失

XGBoost 内置的 `reg:squarederror`（MSE）即可。最终评估用 CCC 而非训练损失。

### CCC 评估（推理阶段）

```python
def ccc_score(y_true, y_pred):
    """推理后评估用，不参与训练"""
    mu_t, mu_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    cov = ((y_true - mu_t) * (y_pred - mu_p)).mean()
    return (2 * cov) / (var_t + var_p + (mu_t - mu_p)**2 + 1e-8)
```

### 可解释性输出（每次推理均输出）

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_current_segment)

# 对单段预测给出解释
# → jitter: +0.15, shimmer: +0.12, HNR: -0.08, F0: -0.03, ...
# → "该段 arousal=0.32（偏低），主要贡献：jitter↑（+0.15）、shimmer↑（+0.12）"
```

---

## 七、exertion 标签说明

v2.2 起使用 ExpressiveSpeech 的 DeEAR `score_nature` 作为 exertion 回归目标：

```python
# ExpressiveSpeech 直接提供 score_nature，无需构造代理标签
y_exertion = df_parquet["score_nature"].values  # ∈ [0, 1]
```

**为什么 nature → exertion？**
- nature 衡量说话自然度：机械朗读（0）← → 自然流畅（1）
- 疲劳时说话更费力 → 声音更不自然 → nature 降低
- 与 arousal 独立（r=0.36），捕捉的是"费力程度"而非"激动程度"

**旧方案（已弃用）**：jitter/shimmer/HNR 加权构造代理标签——在无外部标注时是妥协方案；有了 DeEAR nature 后替换为真实评分。

---

## 八、数据流与接口

### 8.1 模块间传递

```
step3_baseline 输出           →    step4_regression 输入
─────────────────                   ─────────────────
DataFrame(z-scored, 88维)     →    XGBoost 逐段推理
  seg_000: [z_F0, z_jitter, ...]  →  arousal=0.72, exertion=0.31
  seg_001: [z_F0, z_jitter, ...]  →  arousal=0.65, exertion=0.35
  ...

每次输出附带 SHAP 贡献向量，用于告警溯源
```

### 8.2 核心 API

```python
from step4_regression import (
    train_models,
    predict_with_explanation,
)

# 训练
models = train_models(df_ccsemo, output_dir="step4_regression/models/")
# → 保存 model_arousal.json + model_exertion.json

# 推理
result = predict_with_explanation(models, df_z)
# → DataFrame: segment_id, arousal, exertion, shap_top_features, shap_values
```

### 8.3 SHAP 可解释输出

```python
# 单段解释
explanation = explain_segment(model_arousal, df_z.iloc[0])
print(explanation)
# "该段 arousal=0.32。主要贡献：jitter +0.15（高于基线）、shimmer +0.12（高于基线）、
#  HNR -0.08（低于基线）、F0 -0.03（低于基线）。综合判断：嗓音质量下降，疑似疲劳趋势。"
```

---

## 九、H 假设验证

| 假设 | 验证方式 |
|------|---------|
| H1: 特征与疲劳相关 | CCC_arousal > 0.5 + SHAP top-3 包含 jitter/shimmer/HNR |
| H2: arousal+exertion 映射三态 | 规则引擎后续验证 |
| H3: 个体基线有效 | 对比个体基线 vs 群体基线的 CCC 差异 + SHAP 贡献稳定性 |

---

## 十、后续升级路径

| 阶段 | 动作 | 触发条件 |
|------|------|---------|
| MVP | XGBoost + 88维 eGeMAPS | 当前方案 |
| 增强 | 加入时间窗口特征（最近 N 段的趋势：Δjitter、ΔF0） | CCC < 0.5 |
| 增强 | 加入 VAD 节律特征（段间隔、语速变化） | 精度不足 |
| 二期 | 加入 Wav2Vec2 embedding 作为辅助特征（768维），但主路仍保持 eGeMAPS 可解释 | 管制员数据就位 |

---

## 十一、代码文件结构

```
step4_regression/
├── __init__.py           # 包入口
├── model.py              # XGBoost 训练 + 推理 + SHAP 解释
│   ├── train_models()     # 训练 arousal + exertion 双模型
│   ├── predict()          # 推理
│   └── explain_segment()  # SHAP 单段解释
├── dataset.py            # 数据加载
│   └── ExpressiveSpeechLoader  # Parquet 加载 + step2 特征提取
├── pipeline.py           # 接 step3，端到端推理 + 解释
├── models/               # 训练产出（.json 文件）
│   ├── model_arousal.json
│   ├── model_exertion.json
│   └── feature_cols.json
└── scripts/              # 运行脚本
    └── run_step4_expressivespeech.py   # 端到端训练+评估
```

---

## 十二、风险

| 风险 | 缓解 | 状态 |
|------|------|------|
| nature 方差小（std=0.075），回归可能欠拟合 | exertion 模型放低目标（CCC>0.4）；增大样本量 | 需验证 |
| 情感 arousal ≠ 操作疲劳 arousal | 当前为技术验证阶段；二期用真实空管数据微调 | 明确边界 |
| 英语为主，中文管制域迁移风险 | eGeMAPS 语言无关；但韵律模式可能不同 | 可接受 |
| XGBoost 对 88 维全量可能过拟合 | GridSearch 调参 + max_depth 限制 | 可控 |

---

*设计文档完。编码时以此为接口契约。*
