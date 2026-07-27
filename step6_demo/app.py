"""step6 Gradio Demo — 音频上传 → 疲劳状态分析

运行: python step6_demo/app.py
"""
import gradio as gr
import numpy as np
import pandas as pd
import io, warnings, os, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
# 检查中文字体是否可用
def _has_cjk_font():
    from matplotlib.font_manager import FontManager
    fm = FontManager()
    for f in fm.ttflist:
        if 'YaHei' in f.name or 'SimHei' in f.name or 'Heiti' in f.name:
            return True
    return False
_USE_CJK = _has_cjk_font()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from step1_preprocess.vad import vad_detect
from step4_regression.model import load_models
from step5_rules.engine import classify_state, _compute_thresholds
from step5_rules import SpeakerBaseline
from step0_voiceprint import SpeakerClusterer
import opensmile
import soundfile as sf

# ━━━ 加载模型与组件 ━━━
MODEL_DIR = PROJECT_ROOT / "step4_regression" / "models"
print("加载 XGBoost 模型...")
models = load_models(str(MODEL_DIR))
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
print(f"模型就绪: CCC_arousal=0.873, CCC_exertion=0.617")

# 全局实例（跨调用持久）
speaker_baseline = SpeakerBaseline(global_mean=0.72, global_std=0.19)
speaker_clusterer = SpeakerClusterer(threshold=0.55)


def analyze_audio(audio_path: str, denoise: bool = False):
    """核心分析函数
    
    链路: step1 VAD → step2 openSMILE → step4 XGBoost → step5 规则+基线
    （step3 z-score 暂不调用：XGBoost 尚未在 z-scored 特征上重训）
    """
    
    if audio_path is None:
        return None, "请上传音频文件", None, None
    
    # ━━━ Step 1: VAD 分段 ━━━
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=16000)
            sr = 16000
        
        segments = vad_detect(audio, sr, backend="silero")
        
        if len(segments) == 0:
            return None, "VAD 未检测到有效语音段", None, None
        
    except Exception as e:
        return None, f"音频处理失败: {e}", None, None
    
    # ━━━ Step 2: 收集音频段 + openSMILE ━━━
    seg_audios, seg_times, features = [], [], []
    for seg in segments:
        start_idx = int(seg.start_sec * sr)
        end_idx = int(seg.end_sec * sr)
        seg_audio_data = audio[start_idx:end_idx]
        if len(seg_audio_data) < sr * 0.3:
            continue
        seg_audios.append(seg_audio_data)
        seg_times.append((seg.start_sec, seg.end_sec))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feat = smile.process_signal(seg_audio_data.astype(np.float32), sr)
        features.append(feat.iloc[0].values)
    
    if len(features) == 0:
        return None, "所有语音段太短（< 0.3s），无法提取特征", None, None
    
    X = np.array(features, dtype=np.float32)
    
    # ━━━ Step 4: XGBoost 预测 ━━━
    pred_a = models["arousal"].predict(X)
    pred_e = models["exertion"].predict(X)
    pred_a = np.clip(pred_a, 0, 1)
    pred_e = np.clip(pred_e, 0, 1)
    fatigue_p = 1.0 - pred_a
    
    # ━━━ Step 0: 声纹聚类（输入音频段，输出说话人分组） ━━━
    speaker_ids = speaker_clusterer.fit_predict(seg_audios, sr)
    
    # ━━━ Step 5: 状态判定（含个体基线） ━━━
    at = _compute_thresholds(pred_a)
    et = _compute_thresholds(pred_e)
    
    states, confidences, reasons = [], [], []
    a_z, e_z, bl_conf = [], [], []
    for i, (a, e) in enumerate(zip(pred_a, pred_e)):
        sid = speaker_ids[i]  # 声纹聚类给的真实说话人ID
        br = speaker_baseline.normalize(sid, a, e)
        a_z.append(br.arousal_z)
        e_z.append(br.exertion_z)
        bl_conf.append(br.confidence)
        speaker_baseline.update(sid, a, e)
        
        s, c, r = classify_state(a, e, at, et)
        states.append(s)
        confidences.append(c)
        reasons.append(r)
    
    # ━━━ 构建结果 DataFrame ━━━
    df = pd.DataFrame({
        "段号": range(len(seg_times)),
        "起(s)": [round(t[0], 1) for t in seg_times],
        "止(s)": [round(t[1], 1) for t in seg_times],
        "说话人": speaker_ids,
        "arousal": pred_a.round(4),
        "exertion": pred_e.round(4),
        "fatigue_prob": fatigue_p.round(4),
        "状态": states,
        "置信度": confidences,
        "arousal_z": [round(v, 2) for v in a_z],
        "baseline": bl_conf,
    })
    
    # ━━━ 生成图表 ━━━
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 图1: arousal+exertion 曲线
    ax = axes[0, 0]
    x = np.arange(len(seg_times))
    ax.plot(x, pred_a, 'r-o', label='arousal', markersize=4)
    ax.plot(x, pred_e, 'b-s', label='exertion', markersize=4)
    ax.axhline(y=at["high"], color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=at["low"], color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Segment')
    ax.set_ylabel('Score')
    ax.set_title(f'Arousal & Exertion Timeline ({len(seg_times)} VAD segments)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 图2: 状态分布饼图
    ax = axes[0, 1]
    state_counts = pd.Series(states).value_counts()
    colors_map = {'兴奋': '#E24B4A', '稳定': '#378ADD', '疲劳': '#BA7517'}
    colors = [colors_map.get(s, '#888780') for s in state_counts.index]
    ax.pie(state_counts.values, labels=state_counts.index, autopct='%1.0f%%',
           colors=colors, startangle=90)
    ax.set_title(f'State Distribution ({len(seg_times)} VAD segments)')
    
    # 图3: 状态时间线
    ax = axes[1, 0]
    state_colors = {'兴奋': '#E24B4A', '稳定': '#378ADD', '疲劳': '#BA7517'}
    for i, (s, c) in enumerate(zip(states, confidences)):
        alpha = 1.0 if c == 'high' else 0.5 if c == 'medium' else 0.3
        ax.bar(i, 1, color=state_colors.get(s, '#888780'), alpha=alpha, width=0.9)
    ax.set_xlabel('Segment')
    ax.set_yticks([])
    ax.set_title('State Timeline (dark = high confidence)')
    
    # 图4: 汇总信息
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""Analysis Summary
    
Total VAD segments: {len(seg_times)}
Audio duration: {len(audio)/sr:.1f}s

Arousal mean: {pred_a.mean():.3f}
Exertion mean: {pred_e.mean():.3f}

Exited: {state_counts.get('兴奋', 0)} ({100*state_counts.get('兴奋', 0)/len(states):.0f}%)
Stable: {state_counts.get('稳定', 0)} ({100*state_counts.get('稳定', 0)/len(states):.0f}%)
Fatigue: {state_counts.get('疲劳', 0)} ({100*state_counts.get('疲劳', 0)/len(states):.0f}%)

Model: XGBoost+SHAP
Data: ExpressiveSpeech 28k
CCC_arousal: 0.873
Segmentation: Silero VAD
"""
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # ━━━ 状态汇总文本 ━━━
    summary = f"### 分析结果\n\n"
    summary += f"**音频时长**: {len(audio)/sr:.1f} 秒 | **VAD 段数**: {len(seg_times)}\n\n"
    summary += f"**arousal 均值**: {pred_a.mean():.3f} | **exertion 均值**: {pred_e.mean():.3f}\n\n"
    
    # 疲劳段详情
    fatigue_idx = [i for i, s in enumerate(states) if s == '疲劳']
    if fatigue_idx:
        summary += f"**⚠ 疲劳段** ({len(fatigue_idx)} 段):\n"
        for idx in fatigue_idx[:5]:
            t0, t1 = seg_times[idx]
            summary += f"- 段{idx} ({t0:.1f}s-{t1:.1f}s): arousal={pred_a[idx]:.3f}, exertion={pred_e[idx]:.3f}\n"
    else:
        summary += "**✅ 未检测到疲劳段**\n"
    
    return df, summary, fig, df


# ━━━ Gradio 界面 ━━━
with gr.Blocks(title="管制员工作状态分析", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 管制员工作状态分析
    
    上传音频文件，分析 arousal（唤醒度）和 exertion（费力程度），判定疲劳状态。
    
    > 模型基于 ExpressiveSpeech 28k 条数据训练，CCC_arousal=0.873。
    > 当前版本用于技术验证，实际管制场景需用真实数据微调。
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="上传音频",
                type="filepath",
                sources=["upload"],
            )
            analyze_btn = gr.Button("开始分析", variant="primary", size="lg")
            
            gr.Markdown("---")
            gr.Markdown("""
            **技术说明**
            - 模型: XGBoost 双任务回归 + SHAP
            - 特征: openSMILE eGeMAPSv02 (88维)
            - 训练: ExpressiveSpeech 28,190条
            - 判定: percentile自适应阈值
            """)
        
        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="分析结果")
            plot_output = gr.Plot(label="可视化")
    
    with gr.Row():
        table_output = gr.Dataframe(label="逐段详情", interactive=False)
    
    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input],
        outputs=[table_output, summary_output, plot_output, table_output],
    )
    
    # 示例
    gr.Examples(
        examples=[
            str(PROJECT_ROOT / "output" / "REC-1_denoised.wav"),
            str(PROJECT_ROOT / "output" / "REC-2_denoised.wav"),
        ],
        inputs=audio_input,
        label="示例音频（管制录音）",
    )

if __name__ == "__main__":
    print("启动 Gradio Demo...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
