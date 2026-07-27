# 项目长期记忆

## 数据集策略（v2.2 确定，2026-07-21）

- **训练数据**：ExpressiveSpeech（FreedomIntelligence, HuggingFace）
  - ~14k 条，51h，中英双语，parquet 格式，音频嵌入无需单独下载
  - DeEAR 四维分数（arousal/prosody/nature/expressive）
  - arousal 作为 arousal 回归目标，nature 作为 exertion 回归目标
  - prosody/expressive 与 arousal 高度冗余，弃用
- **已放弃**：ATC Fatigue Corpus（IEEE Dataport 需付费）
- **约束**：纯语音判断（排除 RECOLA/BESST 等多模态数据集）

## 技术路线

- 项目分 6 步：step1(降噪+VAD) → step2(openSMILE 88维) → step3(z-score基线) → step4(XGBoost+SHAP) → step5(规则引擎) → step6(Gradio Demo)
- step1-5 已编码完成，全量训练 CCC_arousal=0.873, CCC_exertion=0.617
- 核心创新：不走情绪分类路线（Emotion2Vec+ 已实测失效），走生理声学 XGBoost+SHAP 可解释路线

## 环境

- 使用 funasr conda env（Python 3.12），pip 用清华源
- 关键包：xgboost 3.3.0, shap 0.52.0, opensmile 2.6.0, datasets 4.8.5
