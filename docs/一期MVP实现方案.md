# 工作状态分析 — 一期 MVP 实现方案

> **定位**：内部技术验证，用公开数据证明"语音能反映管制员工作状态"，为二期立项提供依据
> **不是**：给机场方的交付物，不接真实管制录音
> **版本**：v1.6（一期完成） | **日期**：2026-07-22
> **周期**：7-9 周

---

## 一、MVP 定位与目标

### 1.1 MVP 要回答的核心问题

二期立项前，必须先证明四件事（MVP 的四个核心假设）：

| # | 核心假设 | 验证方式 | 通过标准 |
|---|---------|---------|---------|
| H1 | 语音特征与疲劳相关 | 公开疲劳数据集上回归 | 相关性 r > 0.5 |
| H2 | arousal + exertion 两维能映射三态 | 连续情感数据集上聚类 | 三态可分性 > 70% |
| H3 | 个体基线对齐有效（vs 跨人） | 对比实验 | 个体内 r 比跨人 r 高 ≥ 0.15 |
| H4 | 端到端 pipeline 可跑通 | 上传音频→输出状态轨迹 | Demo 可演示 |

**MVP 失败的判定**：H1 或 H3 不达标 → 说明"语音判疲劳"路线不可行，需重新评估，二期不要立项。

### 1.2 MVP 明确不做的事（边界）

- ❌ 不接真实管制录音（拿不到，且合规未走完）
- ❌ 不做声纹识别（用数据集自带说话人标签代替）
- ❌ 不做多信号融合（执勤/航班密度等情境信号缺）
- ❌ 不做实时处理（离线批处理即可）
- ❌ 不追求高准确率（证明相关性即可）
- ❌ 不做告警引擎（只输出状态轨迹）

### 1.3 MVP 的产出物

1. 可运行的 Python pipeline（端到端）
2. 验证报告（H1-H4 指标）
3. Gradio/Streamlit 演示界面（上传音频→状态曲线）
4. 内部立项依据文档

---

## 二、技术栈

| 层 | 选型 | 理由 |
|----|------|------|
| 语言 | Python 3.10+ | 生态成熟 |
| **降噪** | **ClearerVoice-Studio** | **实测教训：不降噪则jitter/shimmer失真；谱减法(noisereduce)反破坏语音** |
| 特征①生理（核心） | **openSMILE eGeMAPSv02** | 生理声学测量，实测能区分Emotion2Vec区分不了的状态 |
| 特征②认知（核心） | **口误率/复诵准确率/填充词/ASR置信度** | 复用课题ASR，认知状态，零采集成本 |
| 特征③嵌入（可选） | Emotion2Vec+ embedding（FunASR） | 取768维不取标签，验证增益后决定 |
| 深度框架 | PyTorch 2.x | 主流，TCN 实现简单 |
| 模型 | XGBoost + SHAP（双任务回归） | 表格数据 SOTA，训练秒级，原生 SHAP 可解释，免 GPU |
| 数据处理 | pandas + numpy + librosa | 标准栈 |
| Demo | Gradio | 几行代码出界面 |
| 可视化 | matplotlib + plotly | 状态轨迹图 |

**关键依赖**：
```
opensmile>=2.6
# ClearerVoice-Studio: git clone https://github.com/modelscope/ClearerVoice-Studio + pip install -r requirements.txt
funasr>=1.0          # 可选：Emotion2Vec+ embedding（验证增益后启用）
torch>=2.0
librosa>=0.10
gradio>=4.0
pandas numpy scikit-learn matplotlib
```

---

## 三、数据集策略（v1.5 确定）

### 3.1 选型原则

MVP 阶段约束：**只用纯音频数据，不依赖多模态（视频/ECG/EDA 等）**。疲劳判断的唯一输入是音频信号。

### 3.2 主数据源：ATC 管制员疲劳语音库

| 维度 | 详情 |
|------|------|
| 来源 | 沈志远 / 南京航空航天大学民航学院 |
| 发布 | IEEE Dataport, 2021. DOI: `10.21227/70sz-mj23` |
| 数据 | 真实空管陆空通话，~700k 段，纯音频 |
| 语言 | 中文（民航局批准，华东空管局） |
| 说话人 | 7 位管制员，含性别/年龄/级别/岗位维度 |
| 时段 | 凌晨 0200-0700 / 上午 1000-1200 / 下午 1330-1530 |
| 标注 | 民航专家标注 fatigue / non-fatigue（二分类） |
| 许可 | CC BY，学术可用 |
| 技术栈对齐 | 同一团队已用 openSMILE + XGBoost + SHAP 在该数据上发表多篇论文 |

**MVP 用途**：
- H1 验证：疲劳二分类（fatigue vs non-fatigue），验证 openSMILE 特征与疲劳的相关性
- H2 验证：用三时段数据构造 fatigue→稳定→兴奋 的三态分类（凌晨疲劳、上午兴奋、下午稳定）
- H3 验证：7 人 per-speaker 基线对齐，对比个体内 vs 跨人效果

### 3.3 辅助验证：ExpressiveSpeech

| 维度 | 详情 |
|------|------|
| 来源 | FreedomIntelligence, HuggingFace / ModelScope |
| 数据 | ~14,000 条，51h，纯音频，16kHz |
| 语言 | 中文 + 英文（约 1:1） |
| 标注 | DeEAR 四维连续分数（arousal/prosody/nature/expressive）+ 情感标签 |
| 许可 | CC BY-NC-SA 4.0 |

**MVP 用途（辅助验证，非主训练）**：
- 用 DeEAR `score_arousal` 作为连续 arousal 回归目标，验证 step4 XGBoost 回归通路
- 交叉验证：openSMILE 88维→XGBoost 预测的 arousal 与 DeEAR 评分的相关性
- 中文表达力语音的预训练辅助

### 3.4 数据策略概览

| 数据集 | 角色 | 标注类型 | MVP 验证目标 |
|--------|------|---------|-------------|
| ATC Fatigue Corpus | **主训练数据** | 疲劳二分类（专家标注） | H1、H2、H3 |
| ExpressiveSpeech | 辅助验证 | arousal 连续分数（DeEAR） | 回归通路验证 |

### 3.5 已排除的数据集

| 数据集 | 排除原因 |
|--------|---------|
| RECOLA | 多模态（视频+ECG+EDA），非纯语音 |
| DAIC-WOZ | 多模态（视频），临床访谈场景不匹配 |
| BESST | 多模态（ECG+EDA+视频），捷克语 |
| RAVDESS/MESD | 表演性离散情感，无疲劳/唤醒度标注 |
| SUSAS | 8kHz 低采样、无疲劳标注、需付费（LDC） |

---

## 四、模块实现（按周拆解）

### 4.1 模块架构

```
┌──────────┐ ┌─────────┐ ┌──────────────┐ ┌─────────┐ ┌──────────┐ ┌────────┐
│ 数据准备  │→│ 降噪    │→│ 特征提取     │→│基线对齐 │→│ XGBoost  │→│状态判定│
│ATC Fatigue│ │ClearVoice│ │①openSMILE   │ │个体     │ │回归+SHAP │ │规则+Demo│
│+Expressive│ │门控      │ │②文本认知    │ │z-score  │ │          │ │        │
│Speech     │ │          │ │③emb(可选)   │ │         │ │          │ │        │
└──────────┘ └─────────┘ └──────────────┘ └─────────┘ └──────────┘ └────────┘
```

### 4.2 实现步骤

#### Step 1：环境与数据准备（W1）

**环境**：
```bash
python -m venv venv && source venv/bin/activate
pip install opensmile torch librosa gradio pandas scikit-learn matplotlib funasr
# ClearerVoice-Studio: git clone + pip install -r requirements.txt（阿里ModelScope）
```

**数据解析**（以 ATC Fatigue Corpus 为例）：
- 从 IEEE Dataport 下载 7 个子数据集（ATCs_1 ~ ATCs_7）
- 读取音频 + fatigue/non-fatigue 标签 + 时段信息
- 说话人 ID 即为 ATCs_1~7（用于个体基线）
- 训练/验证/测试按说话人划分（**绝不跨集泄漏**）

**关键产出**：`data_loader.py`，输出 `(audio, features_ready, speaker_id, labels)`

#### Step 2：降噪 + 特征提取 pipeline（W2-W3）

**降噪前置（ClearerVoice-Studio）— 实测教训，必须做**：
```python
# 实测验证：不降噪则 jitter/shimmer 受噪声干扰失真；
# noisereduce(谱减法)反而破坏语音，必须用 ClearerVoice-Studio
from clearvoice import ClearVoice
cv = ClearVoice(model='MossFormerGAN_SE_16K')  # 16kHz适配电台窄带
audio_enhanced = cv.enhance(audio_path)  # 降噪后音频
```

**录音质量门控**（降噪后评估）：
- 计算降噪后 SNR / 嗓音段占比
- 质量过差的段（SNR < 阈值）标记"低置信度"或丢弃
- 避免噪声伪影被误读为疲劳（REC-4 教训）

**openSMILE eGeMAPS（核心，88维）**：
```python
import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
features = smile.process_file(audio_enhanced_path)  # 88维生理声学特征
```

**文本认知指标（核心路②，复用ASR转写文本）**：
- 口误率 = 口误次数 / 指令总数
- 复诵准确率 = 正确复诵 / 总复诵（金信号，复用课题5.2进阶一）
- 填充词频率 = 嗯/啊/那个 / 通话时长
- ASR平均置信度（疲劳时下降，声学-认知桥梁）
- 全部做个体内 z-score + 时序趋势，与openSMILE统一处理

**Emotion2Vec+ embedding（可选扩展③，仅取特征不取标签）**：
```python
from funasr import AutoModel
model = AutoModel(model="iic/emotion2vec_plus_large")
result = model.generate(audio_path, extract_embedding=True)
embedding = result[0]["feats"]  # 768维
```

**MVP 决策**：
- **先纯 openSMILE + 文本认知两路跑通**（核心，轻量快）
- Emotion2Vec embedding 作为 W4 可选实验（H5a），验证增益后再决定是否纳入

**关键产出**：`denoise.py` + `feature_extraction.py` + `text_cognitive.py`，输出特征矩阵（核心：88生理+~6认知；可选：+768嵌入）

#### Step 3：个体基线对齐（W3）

**逻辑**：
```python
# 每个说话人，取前 N 个样本作为基线（模拟"值班前2小时清醒段"）
for speaker in speakers:
    baseline_features = features[speaker][:N]
    mean, std = baseline_features.mean(), baseline_features.std()
    features[speaker] = (features[speaker] - mean) / std  # z-score
```

**对比实验**（验证 H3）：
- 方案A：个体内 z-score（实验组）
- 方案B：全局 z-score（对照组，跨人）
- 分别训练回归模型，比较 arousal 预测的 r 值

**关键产出**：`baseline_alignment.py` + H3 验证报告

#### Step 4：维度回归模型（W4）

**模型**：XGBoost 双任务回归（详见 `docs/维度回归模块-XGBoost设计.md`）

**训练配置**：
- **arousal**：ATC Fatigue Corpus 的 fatigue/non-fatigue 标签 → 概率值作为 arousal ≈ 1-fatigue_prob
- **exertion**：代理标签（jitter/shimmer/HNR z-score 加权→Sigmoid→[0,1]）
- Loss：`reg:squarederror`（XGBoost 内置 MSE）
- 调参：GridSearchCV（n_estimators=[100,200,300], max_depth=[4,6,8]）
- 输出：`model_arousal.json` + `model_exertion.json` + SHAP 解释

**可选辅助验证**：
- 用 ExpressiveSpeech 的 DeEAR `score_arousal` 作为连续回归目标，验证 openSMILE→XGBoost→arousal 通路的泛化性

**关键产出**：`model.py` + `train.py` + 训练好的 checkpoint + SHAP 特征贡献报告

#### Step 5：状态判定 + 趋势（W5）

**趋势计算**：
```python
# 滑窗斜率：近 30 分钟 arousal 的线性回归斜率
slope_a = linear_regression(arousal_series[-window:]).slope
slope_e = linear_regression(exertion_series[-window:]).slope
```

**规则引擎**（实现方案文档第四章 4.6 的规则）：
```python
def classify_state(arousal_z, exertion_z, slope_a, slope_e):
    if abs(arousal_z) < 0.5 and exertion_z < 0.5:
        return "稳定"
    elif slope_a < -0.02 and slope_e > 0.02:  # 趋势疲劳
        return "疲劳"
    elif arousal_z < -1.0 and exertion_z > 1.5:
        return "疲劳(晚期)"
    elif arousal_z > 0.8:
        return "兴奋"
    return "稳定"
```

**关键产出**：`state_engine.py`

#### Step 6：Demo 界面（W6）

**Gradio 界面**：
- 上传音频文件
- 显示：openSMILE 特征曲线 + arousal/exertion 轨迹 + 状态标注
- 输出：状态轨迹图（matplotlib）

```python
import gradio as gr
def analyze(audio):
    # 端到端：特征→基线→回归→判定
    return state_trajectory_plot
gr.Interface(fn=analyze, inputs=gr.Audio(), outputs=gr.Image()).launch()
```

**关键产出**：`app.py` + 可演示 demo

---

## 五、验证实验设计

### 5.1 H1 验证：openSMILE 特征与疲劳相关

**数据**：ATC Fatigue Corpus（fatigue vs non-fatigue，7 位管制员）
**方法**：提取 openSMILE 88维，训练 XGBoost 二分类（fatigue/non-fatigue），在留出的管制员上测试
**通过**：分类准确率 > 75% 且 SHAP top-5 特征包含 jitter/shimmer/HNR 中至少 3 个

### 5.2 H2 验证：三态可分

**数据**：ATC Fatigue Corpus（三时段：凌晨疲劳、上午兴奋、下午稳定）+ ExpressiveSpeech（sleepy 等疲劳近似标签辅助）
**方法**：
- 主验证：ATC 三时段疲劳分均值差异显著性检验（ANOVA + 事后比较）
- 辅助验证：ExpressiveSpeech 的 sleepy/bored vs happy/angry vs calm/default 三组 openSMILE 特征可分性
**通过**：三时段 arousal 均值差异显著（p < 0.05），或三组分类准确率 > 70%

### 5.3 H3 验证：个体基线有效

**数据**：ATC Fatigue Corpus（7 人 per-speaker）
**方法**：A/B 对比实验
- A：个体内 z-score → 训练 → 测试准确率
- B：全局标准化 → 训练 → 测试准确率
**通过**：A 的准确率比 B 高 ≥ 10 个百分点

### 5.4 H4 验证：端到端可跑

**方法**：上传任意音频，10 秒内输出状态轨迹图
**通过**：Demo 稳定运行，内部可演示

### 5.5 H5 验证：降噪是硬前提

**方法**：用现有 REC-1~4 实测：降噪前 vs ClearerVoice-Studio 降噪后的 jitter/shimmer 变化
**预期**：降噪后特征更稳定，印证"降噪前置是硬前提"
**已知反例**：noisereduce 谱减法破坏语音（已实测），不可用
**通过**：降噪后 jitter/shimmer 明显改善或更稳定

### 5.6 已知前提（非假设）：Emotion2Vec 分类路线不可行

- 已实测 REC-1~4：REC-2/4 标准管制通话被判 happy 99%+，sad 始终 0.04~0.07
- 离散情绪分类与工作状态（兴奋/稳定/疲劳）是不同概念空间，映射逻辑不成立
- 本方案基于此前提设计：不走情绪分类路线，走生理声学（openSMILE）+ 认知指标（文本）路线
- **不设为待验证假设**，作为立项依据的实证支撑

---

## 六、时间规划

| 周 | 任务 | 产出 | 假设验证 |
|----|------|------|---------|
| W1 | 环境 + 数据获取/解析 | data_loader.py | — |
| W2 | **ClearerVoice降噪** + openSMILE特征 | denoise.py + feature_extraction.py | H5降噪验证 |
| W3 | 个体基线对齐 | baseline_alignment.py | **H3初步** |
| W4 | arousal/exertion 回归训练 | model.py + checkpoint | **H1/H2** |
| W5 | 状态判定规则 + 趋势 | state_engine.py | — |
| W6 | Demo 界面 + 端到端联调 | app.py | **H4** |
| W7 | 验证实验 + 调优 | 验证报告 | H1-H5复核 |
| W8 | 文档 + 内部演示 | 立项依据 | — |
| W9 | 缓冲 | — | — |

**关键里程碑**：
- W3 末：H3 初步结论（个体基线是否有效）→ 决定是否继续
- W4 末：H1/H2 结论（核心假设是否成立）→ 决定二期是否立项
- W6 末：可演示 demo

---

## 七、风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| ATC Fatigue Corpus 标注质量参差 | 低 | 同一团队已发表验证结果（准确率 100% XGBoost），标注可靠性有论文支撑 |
| fatigue/non-fatigue 二分类粒度粗 | 中 | 利用三时段信息构造 coarse-grained 三态映射；ExpressiveSpeech DeEAR arousal 补充连续维度 |
| exertion 无标注 | 中 | MVP 用嗓音质量特征（jitter/shimmer/HNR）构造代理标签，标注为已知妥协 |
| 真实管制环境与数据集差异 | 中 | ATC Fatigue Corpus 就是真实空管录音，域匹配度已最大化；但 VHF 窄带 vs 16kHz 降噪链路需验证 |
| 7 人样本偏少 | 低 | 700k 段数量充足；ExpressiveSpeech 中英双语补充泛化性 |

---

## 八、MVP 成功标准（一句话）

> **W4 末，在 ATC Fatigue Corpus 上证明 openSMILE 特征能有效区分 fatigue/non-fatigue（分类准确率 > 75%，SHAP 确认 jitter/shimmer/HNR 为主要贡献），且个体基线比全局基线准确率高 ≥ 10% —— 则二期值得立项。**

MVP 不追求准确率，追求"相关性证明 + 链路跑通"。准确率是二期的事。

---

## 九、与二期衔接

| MVP 产出 | 二期复用 |
|---------|---------|
| 特征提取 pipeline | 直接复用，换真实空管+飞行员混合录音 |
| TCN 模型架构 | 微调，加 exertion 真实标注 |
| 个体基线逻辑 | 扩展为分班次基线 |
| 状态判定规则 | 扩展为多信号融合（+执勤时长+航班密度） |
| 验证方法论 | 复用，数据换为机坪管制语料 |
| **目标说话人提取** | **新增：ClearerVoice TSE（参考语音）+ ECAPA-TDNN身份校验** |
| Demo | 升级为实时版本 + 多人录音支持 |

**MVP 是二期的技术降风险手段**：用最小成本验证最大不确定性（语音能否判状态），通过后才投入真实数据采集和合规流程。

---

*方案完。如需展开某一 Step 的完整代码实现，或调整数据集策略，可单独深化。*
