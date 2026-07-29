"""step6 Gradio Demo — 管制员工作状态分析
运行: python step6_demo/app.py
"""
import gradio as gr
import numpy as np
import pandas as pd
import io, warnings, os, sys, time, json, uuid
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from step1_preprocess.vad import vad_detect
from step4_regression.model import load_models
from step5_rules.engine import classify_state, _compute_thresholds
from step5_rules import SpeakerBaseline
from step0_voiceprint import SpeakerClusterer, VoiceprintDB
import opensmile
import soundfile as sf

MODEL_DIR = PROJECT_ROOT / "step4_regression" / "models"
MODEL_VERSION = "v2.2"
CCC_AROUSAL = 0.873
CCC_EXERTION = 0.617

print(f"[init] 加载模型 {MODEL_VERSION} ...")
models = load_models(str(MODEL_DIR))
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
print(f"[init] 就绪  CCC_arousal={CCC_AROUSAL}  CCC_exertion={CCC_EXERTION}")

TRACE_DIR = PROJECT_ROOT / "output" / ".traces" / datetime.now().strftime("%Y-%m-%d")
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# VoiceprintDB + SpeakerBaseline: 模块级持久化, 跨请求累积
VPDB_PATH = PROJECT_ROOT / "output" / "voiceprint_db.json"
BL_PATH = PROJECT_ROOT / "output" / "speaker_baseline.json"
voiceprint_db = VoiceprintDB(str(VPDB_PATH))
voiceprint_db.load()
print(f"[init] 声纹库: {len(voiceprint_db)} 个说话人")

speaker_baseline = SpeakerBaseline(global_mean=0.72, global_std=0.19)
speaker_baseline.load(str(BL_PATH))
print(f"[init] SpeakerBaseline: {len(speaker_baseline._speakers)} 个说话人")

# 删除声纹库记录时同步清理 SpeakerBaseline
def _on_speaker_delete(sid):
    if sid in speaker_baseline._speakers:
        del speaker_baseline._speakers[sid]
        speaker_baseline.save(str(BL_PATH))
voiceprint_db.on_delete(_on_speaker_delete)

# SpeakerClusterer: 仅用于 embedding 提取, 不做聚类
embedding_extractor = SpeakerClusterer(threshold=0.55)


# ===========================================================================
#  核心分析 — 单文件处理
# ===========================================================================

def _process_single_file(audio_path, baseline, trace_id, progress=None):
    """处理单个音频文件，返回 (df, metadata_dict)

    baseline: 全局持久化 SpeakerBaseline
    逐段 embedding → VoiceprintDB 匹配 → 持久 speaker_id
    progress: 可选 gr.Progress 实例
    """
    def _p(frac, desc):
        if progress:
            progress(frac, desc=desc)

    timing = {}
    t_total = time.time()

    _p(0.05, desc="加载音频...")

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=16000)
        sr = 16000
    audio_dur = len(audio) / sr

    # VAD
    _p(0.10, desc="VAD 语音分段...")
    t0 = time.time()
    segments = vad_detect(audio, sr, backend="silero")
    timing["vad_ms"] = int((time.time() - t0) * 1000)

    if len(segments) == 0:
        return None, {"audio_dur": audio_dur, "n_seg": 0, "error": "no_speech", "timing": timing}

    # openSMILE
    _p(0.20, desc="提取声学特征...")
    t0 = time.time()
    seg_audios, seg_times, features = [], [], []
    for seg in segments:
        si = int(seg.start_sec * sr)
        ei = int(seg.end_sec * sr)
        seg_data = audio[si:ei]
        if len(seg_data) < sr * 0.3:
            continue
        seg_audios.append(seg_data)
        seg_times.append((seg.start_sec, seg.end_sec))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feat = smile.process_signal(seg_data.astype(np.float32), sr)
        features.append(feat.iloc[0].values)
    timing["opensmile_ms"] = int((time.time() - t0) * 1000)

    if len(features) == 0:
        return None, {"audio_dur": audio_dur, "n_seg": 0, "error": "too_short", "timing": timing}

    X = np.array(features, dtype=np.float32)
    n_seg = len(features)

    # XGBoost
    _p(0.40, desc="XGBoost 推理...")
    t0 = time.time()
    pred_a = models["arousal"].predict(X)
    pred_e = models["exertion"].predict(X)
    pred_a = np.clip(pred_a, 0, 1)
    pred_e = np.clip(pred_e, 0, 1)
    fatigue_p = 1.0 - pred_a
    timing["xgb_ms"] = int((time.time() - t0) * 1000)

    # 声纹: 逐段提取 embedding → VoiceprintDB 匹配
    _p(0.60, desc="声纹匹配...")
    t0 = time.time()
    embeddings = embedding_extractor.extract_embeddings(seg_audios, sr)
    speaker_ids = []
    for emb in embeddings:
        sid = voiceprint_db.match_and_update(emb)
        speaker_ids.append(sid)
    timing["voiceprint_ms"] = int((time.time() - t0) * 1000)

    # 状态判定（全局持久 SpeakerBaseline + 基线置信度降级）
    _p(0.80, desc="状态判定...")
    t0 = time.time()
    at = _compute_thresholds(pred_a)
    et = _compute_thresholds(pred_e)

    states, confidences, reasons = [], [], []
    a_z, e_z, bl_conf, bl_using = [], [], [], []
    for i, (a, e) in enumerate(zip(pred_a, pred_e)):
        sid = speaker_ids[i]
        br = baseline.normalize(sid, a, e)
        a_z.append(br.arousal_z)
        e_z.append(br.exertion_z)
        bl_conf.append(br.confidence)
        bl_using.append(br.using_individual)
        baseline.update(sid, a, e)
        s, c, r = classify_state(a, e, at, et)
        # 基线不可靠时降级
        if br.confidence == "low":
            c = "low" if c == "low" else ("medium" if c == "high" else "low")
        elif br.confidence == "medium" and c == "high":
            c = "medium"
        states.append(s)
        confidences.append(c)
        reasons.append(r)
    timing["state_ms"] = int((time.time() - t0) * 1000)
    timing["total_ms"] = int((time.time() - t_total) * 1000)

    fname = os.path.basename(audio_path)
    df = pd.DataFrame({
        "来源文件": [fname] * n_seg,
        "段号": range(n_seg),
        "起(s)": [round(t[0], 1) for t in seg_times],
        "止(s)": [round(t[1], 1) for t in seg_times],
        "时长(s)": [round(t[1] - t[0], 1) for t in seg_times],
        "说话人": speaker_ids,
        "arousal": pred_a.round(3),
        "exertion": pred_e.round(3),
        "fatigue_prob": fatigue_p.round(3),
        "状态": states,
        "置信度": confidences,
        "基线状态": bl_conf,
        "使用个体基线": bl_using,
    })

    meta = {
        "audio_dur": audio_dur, "n_seg": n_seg,
        "pred_a": pred_a, "pred_e": pred_e, "fatigue_p": fatigue_p,
        "states": states, "confidences": confidences,
        "speaker_ids": speaker_ids, "seg_times": seg_times,
        "bl_conf": bl_conf, "bl_using": bl_using,
        "at": at, "et": et, "timing": timing,
    }
    return df, meta


# ===========================================================================
#  批量分析入口
# ===========================================================================

def analyze_audio(audio_path, progress=gr.Progress()):
    """统一入口：单个路径或路径列表均可"""
    if audio_path is None:
        raise gr.Error("请先上传音频文件")

    def _to_path(fp):
        if isinstance(fp, str):
            return fp
        if hasattr(fp, "name"):
            return fp.name
        return str(fp)

    if isinstance(audio_path, str):
        return _analyze_single(audio_path, progress)
    # 列表: 统一为字符串路径
    paths = [_to_path(fp) for fp in audio_path]
    if len(paths) == 1:
        return _analyze_single(paths[0], progress)
    return _analyze_batch(paths, progress)


def _analyze_single(audio_path, progress):
    trace_id = uuid.uuid4().hex[:8]
    progress(0.05, desc="加载音频...")

    df, meta = _process_single_file(audio_path, speaker_baseline, trace_id, progress)
    if df is None:
        _save_trace(trace_id, meta["audio_dur"], 0, meta["timing"],
                    error=meta.get("error", "unknown"))
        return _empty_result(meta["audio_dur"], trace_id,
                            "VAD 未检测到有效语音段" if meta.get("error") == "no_speech"
                            else "所有语音段太短（< 0.3s）")

    _save_trace(trace_id, meta["audio_dur"], meta["n_seg"], meta["timing"],
                arousal_mean=float(meta["pred_a"].mean()),
                exertion_mean=float(meta["pred_e"].mean()),
                states=meta["states"])

    fig = _build_figure(meta["seg_times"], meta["pred_a"], meta["pred_e"],
                        meta["at"], meta["et"], meta["states"],
                        meta["confidences"], meta["speaker_ids"],
                        meta["audio_dur"], meta["n_seg"],
                        file_labels=None, n_files=1)

    summary = _build_summary(meta["audio_dur"], meta["n_seg"],
                             meta["pred_a"], meta["pred_e"],
                             meta["states"], meta["confidences"],
                             meta["seg_times"], meta["speaker_ids"],
                             meta["bl_conf"], meta["bl_using"],
                             [meta["timing"]], trace_id, meta["at"], meta["et"],
                             n_files=1)

    # 批量持久化: 请求结束后写一次
    voiceprint_db.save()
    speaker_baseline.save(str(BL_PATH))

    progress(1.0, desc="完成")
    return df, summary, fig, df


def _analyze_batch(file_paths, progress):
    n_files = len(file_paths)
    trace_id = uuid.uuid4().hex[:8]
    progress(0.02, desc=f"处理 {n_files} 个文件...")

    all_dfs, all_metas, errors = [], [], []
    total_dur = 0

    for idx, fp in enumerate(file_paths):
        pct = 0.05 + 0.90 * (idx / n_files)
        fname = os.path.basename(fp)
        progress(pct, desc=f"[{idx+1}/{n_files}] {fname[:30]}...")

        df, meta = _process_single_file(fp, speaker_baseline, trace_id)
        if df is None:
            errors.append(f"{fname}: {meta.get('error', 'unknown')}")
            total_dur += meta["audio_dur"]
        else:
            all_dfs.append(df)
            all_metas.append(meta)
            total_dur += meta["audio_dur"]

    if not all_dfs:
        progress(1.0, desc="完成")
        return _empty_result(total_dur, trace_id,
                            f"{n_files} 个文件均无有效语音段\n" + "\n".join(errors))

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["段号"] = range(len(df_all))  # 重新编号

    # 合并统计数据
    pred_a_all = np.concatenate([m["pred_a"] for m in all_metas])
    pred_e_all = np.concatenate([m["pred_e"] for m in all_metas])
    states_all = [s for m in all_metas for s in m["states"]]
    conf_all = [c for m in all_metas for c in m["confidences"]]
    sp_ids_all = [s for m in all_metas for s in m["speaker_ids"]]
    seg_times_all = [t for m in all_metas for t in m["seg_times"]]
    bl_conf_all = [b for m in all_metas for b in m["bl_conf"]]
    bl_using_all = [u for m in all_metas for u in m["bl_using"]]
    at = _compute_thresholds(pred_a_all)
    et = _compute_thresholds(pred_e_all)
    timings = [m["timing"] for m in all_metas]

    # 标注每个段属于哪个文件
    file_labels = []
    for m in all_metas:
        file_labels.extend([os.path.basename(fp) for fp in file_paths
                           if os.path.basename(fp) in df_all["来源文件"].values][:1]
                           * m["n_seg"])

    timing_total = sum(t.get("total_ms", 0) for t in timings)
    _save_trace(trace_id, total_dur, len(df_all),
                {"total_ms": timing_total, "n_files": n_files},
                arousal_mean=float(pred_a_all.mean()),
                exertion_mean=float(pred_e_all.mean()),
                states=states_all, errors=errors)

    fig = _build_figure(None, pred_a_all, pred_e_all, at, et,
                        states_all, conf_all, sp_ids_all,
                        total_dur, len(df_all),
                        file_labels=df_all["来源文件"].tolist() if n_files > 1 else None,
                        n_files=n_files)

    summary = _build_summary(total_dur, len(df_all),
                             pred_a_all, pred_e_all,
                             states_all, conf_all,
                             seg_times_all, sp_ids_all,
                             bl_conf_all, bl_using_all,
                             timings, trace_id, at, et,
                             n_files=n_files, errors=errors)

    # 批量持久化: 请求结束后写一次
    voiceprint_db.save()
    speaker_baseline.save(str(BL_PATH))

    progress(1.0, desc="完成")
    return df_all, summary, fig, df_all


# ===========================================================================
#  辅助函数
# ===========================================================================

def _empty_result(audio_dur, trace_id, msg):
    summary = f"### 分析结果\n\n> {msg}\n\n"
    summary += f"音频时长: {audio_dur:.1f}s | trace: `{trace_id}`"
    empty_df = pd.DataFrame(columns=[
        "来源文件", "段号", "起(s)", "止(s)", "时长(s)",
        "说话人", "arousal", "exertion", "fatigue_prob",
        "状态", "置信度", "基线状态", "使用个体基线",
    ])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14,
            transform=ax.transAxes, color="#888")
    ax.axis("off")
    return empty_df, summary, fig, empty_df


def _build_figure(seg_times, pred_a, pred_e, at, et, states,
                  confidences, speaker_ids, audio_dur, n_seg,
                  file_labels=None, n_files=1):
    x = np.arange(n_seg)
    state_cmap = {"兴奋": "#E24B4A", "稳定": "#378ADD", "疲劳": "#BA7517",
                   "疲劳趋势": "#EF9F27"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("white")

    # 图1: arousal + exertion 时间线
    ax = axes[0, 0]
    ax.plot(x, pred_a, "o-", color="#E24B4A", markersize=4, linewidth=1.2,
            label=f"arousal (mean={pred_a.mean():.2f})")
    ax.plot(x, pred_e, "s-", color="#378ADD", markersize=4, linewidth=1.2,
            label=f"exertion (mean={pred_e.mean():.2f})")
    ax.axhline(at["high"], color="#E24B4A", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(at["low"], color="#E24B4A", linestyle=":", alpha=0.4, linewidth=1)
    ax.fill_between(x, 0, 1,
                    where=(pred_a < at["low"]) & (pred_e > et["high"]),
                    color="#BA7517", alpha=0.08, label="fatigue zone")
    if file_labels and n_files > 1:
        # 标注文件边界
        prev_f = file_labels[0]
        for i, f in enumerate(file_labels):
            if f != prev_f:
                ax.axvline(x=i - 0.5, color="#888", linestyle="--", alpha=0.3, linewidth=0.8)
                prev_f = f
    ax.set_xlabel("Segment")
    ax.set_ylabel("Score")
    ax.set_title(f"Arousal / Exertion  ({n_seg} segments, {audio_dur:.0f}s audio, {n_files} files)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # 图2: 状态分布
    ax = axes[0, 1]
    state_counts = pd.Series(states).value_counts()
    pie_labels = [f"{k}\n({v})" for k, v in state_counts.items()]
    pie_colors = [state_cmap.get(s, "#888780") for s in state_counts.index]
    ax.pie(state_counts.values, labels=pie_labels, colors=pie_colors,
           startangle=90, textprops={"fontsize": 10})
    ax.set_title("State Distribution")

    # 图3: 状态时间线
    ax = axes[1, 0]
    for i in range(n_seg):
        s = states[i]
        alpha = 1.0 if confidences[i] == "high" else 0.55
        ax.bar(i, 1, color=state_cmap.get(s, "#888780"),
               alpha=alpha, width=0.85, edgecolor="white", linewidth=0.5)
    if file_labels and n_files > 1:
        prev_f = file_labels[0]
        for i, f in enumerate(file_labels):
            if f != prev_f:
                ax.axvline(x=i - 0.5, color="#888", linestyle="--", alpha=0.3, linewidth=0.8)
                ax.text(i - 0.3, 0.5, prev_f[:12], rotation=90, fontsize=7, color="#888", va="center")
                prev_f = f
        ax.text(n_seg - 0.3, 0.5, prev_f[:12], rotation=90, fontsize=7, color="#888", va="center")
    ax.set_xlabel("Segment")
    ax.set_yticks([])
    ax.set_title("State Timeline  (dark=high confidence)")
    ax.set_xlim(-0.5, n_seg - 0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # 图4: 逐说话人
    ax = axes[1, 1]
    sp_data = {}
    for i, sp in enumerate(speaker_ids):
        short_sp = sp  # persist_* ID already clean, no file prefix
        sp_data.setdefault(short_sp, {"a": [], "e": []})
        sp_data[short_sp]["a"].append(pred_a[i])
        sp_data[short_sp]["e"].append(pred_e[i])

    n_speakers = len(sp_data)
    bar_w = 0.35
    for j, (sp, d) in enumerate(sorted(sp_data.items())):
        a_mean = np.mean(d["a"])
        e_mean = np.mean(d["e"])
        ax.bar(j - bar_w/2, a_mean, bar_w, color="#E24B4A", alpha=0.7,
               label="arousal" if j == 0 else "")
        ax.bar(j + bar_w/2, e_mean, bar_w, color="#378ADD", alpha=0.7,
               label="exertion" if j == 0 else "")
        ax.text(j, max(a_mean, e_mean) + 0.03, f"n={len(d['a'])}",
                ha="center", fontsize=8, color="#666")
    ax.set_xticks(range(n_speakers))
    ax.set_xticklabels(sorted(sp_data.keys()), fontsize=8, rotation=45 if n_speakers > 4 else 0)
    ax.set_ylabel("Mean Score")
    ax.set_title(f"Per-Speaker  ({n_speakers} speakers)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout(pad=2)
    return fig


def _build_summary(audio_dur, n_seg, pred_a, pred_e, states,
                   confidences, seg_times, speaker_ids,
                   bl_conf, bl_using, timings, trace_id, at, et,
                   n_files=1, errors=None):
    state_counts = pd.Series(states).value_counts()
    n_fatigue = state_counts.get("疲劳", 0)
    n_excited = state_counts.get("兴奋", 0)
    n_trend = state_counts.get("疲劳趋势", 0)
    n_low_conf = sum(1 for c in confidences if c == "low")
    n_cold = sum(1 for u in bl_using if not u)
    timing_total = sum(t.get("total_ms", 0) for t in timings)

    lines = [
        "## 分析结果",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 文件数 | {n_files} |",
        f"| 总时长 | {audio_dur:.1f}s |",
        f"| VAD 段数 | {n_seg} |",
        f"| 说话人数 | {len(set(speaker_ids))} |",
        f"| arousal 均值 | {pred_a.mean():.3f} (low<{at['low']:.2f}, high>{at['high']:.2f}) |",
        f"| exertion 均值 | {pred_e.mean():.3f} |",
        f"| 处理耗时 | {timing_total/1000:.1f}s |",
        "",
    ]

    if errors:
        lines.append(f"> 部分文件处理异常：")
        for e in errors:
            lines.append(f"> - {e}")
        lines.append("")

    if n_fatigue > 0:
        lines.append(f"### 疲劳段 ({n_fatigue} 段)")
        lines.append("")
        for i, s in enumerate(states):
            if s == "疲劳":
                t0, t1 = seg_times[i]
                sp = speaker_ids[i]
                lines.append(
                    f"- 段{i} {t0:.1f}s-{t1:.1f}s ({sp})  "
                    f"arousal={pred_a[i]:.3f}  exertion={pred_e[i]:.3f}  "
                    f"conf={confidences[i]}"
                )
        lines.append("")

    if n_excited > 0:
        lines.append(f"### 兴奋段 ({n_excited} 段)")
        lines.append("")
        for i, s in enumerate(states):
            if s == "兴奋":
                t0, t1 = seg_times[i]
                sp = speaker_ids[i]
                lines.append(
                    f"- 段{i} {t0:.1f}s-{t1:.1f}s ({sp})  "
                    f"arousal={pred_a[i]:.3f}  conf={confidences[i]}"
                )
        lines.append("")

    if n_low_conf > 0:
        lines.append(f"> {n_low_conf}/{n_seg} 段置信度为 low，建议人工复核")
        lines.append("")

    if n_cold > 0:
        lines.append(f"> 说话人基线冷启动（{n_cold} 段使用群体基线），置信度已被降级")
        lines.append("")

    lines.append("---")
    lines.append(f"trace: `{trace_id}` | model: {MODEL_VERSION} "
                 f"(CCC_arousal={CCC_AROUSAL}, CCC_exertion={CCC_EXERTION})")
    lines.append("expressiveSpeech 28k | Silero VAD | openSMILE eGeMAPSv02 | XGBoost")

    return "\n".join(lines)


def _save_trace(trace_id, audio_dur, n_seg, timing, **kwargs):
    trace = {
        "trace_id": trace_id,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "model_version": MODEL_VERSION,
        "audio_duration_s": round(audio_dur, 1),
        "n_segments": n_seg,
        "timing_ms": timing,
        **kwargs,
    }
    trace_path = TRACE_DIR / f"{trace_id}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)


# ===========================================================================
#  Gradio 界面
# ===========================================================================

CSS = """
.metric-card {
    background: var(--background-fill-secondary);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    margin: 0;
}
.metric-label {
    font-size: 0.8rem;
    color: var(--body-text-color-subdued);
    margin: 4px 0 0 0;
}
.metric-warn { color: #BA7517; }
.metric-danger { color: #E24B4A; }
.metric-ok { color: #378ADD; }
"""

with gr.Blocks(
    title="管制员工作状态分析",
    theme=gr.themes.Soft(),
    css=CSS,
) as demo:

    gr.Markdown("""
    # 管制员工作状态分析

    上传空管通话录音（支持多文件批量），自动检测 **arousal（唤醒度）** 和 **exertion（费力程度）**，
    判定管制员工作状态。基于 ExpressiveSpeech 28k 条数据训练。
    """)

    gr.Markdown(
        f"> model: `{MODEL_VERSION}` | XGBoost + SHAP | openSMILE eGeMAPSv02 (88d) | "
        f"CCC_arousal=`{CCC_AROUSAL}` CCC_exertion=`{CCC_EXERTION}` | "
        f"Silero VAD | ERES2NetV2 | 声纹库 | 批量上传"
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            audio_input = gr.File(
                label="上传音频（支持多文件）",
                file_count="multiple",
                file_types=[".wav", ".mp3", ".flac", ".m4a"],
            )
            analyze_btn = gr.Button("开始分析", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Row():
                metric_files = gr.HTML("""<div class="metric-card">
                    <p class="metric-value">--</p>
                    <p class="metric-label">文件数</p></div>""")
                metric_audio = gr.HTML("""<div class="metric-card">
                    <p class="metric-value">--</p>
                    <p class="metric-label">总时长</p></div>""")
                metric_seg = gr.HTML("""<div class="metric-card">
                    <p class="metric-value">--</p>
                    <p class="metric-label">VAD 段数</p></div>""")
                metric_arousal = gr.HTML("""<div class="metric-card">
                    <p class="metric-value">--</p>
                    <p class="metric-label">arousal 均值</p></div>""")
            with gr.Row():
                metric_exertion = gr.HTML("""<div class="metric-card">
                    <p class="metric-value">--</p>
                    <p class="metric-label">exertion 均值</p></div>""")
                metric_fatigue = gr.HTML("""<div class="metric-card">
                    <p class="metric-value metric-ok">--</p>
                    <p class="metric-label">疲劳段</p></div>""")
                metric_excited = gr.HTML("""<div class="metric-card">
                    <p class="metric-value">--</p>
                    <p class="metric-label">兴奋段</p></div>""")
                metric_time = gr.HTML("""<div class="metric-card">
                    <p class="metric-value">--</p>
                    <p class="metric-label">处理耗时</p></div>""")

    with gr.Tabs():
        with gr.TabItem("概览"):
            plot_output = gr.Plot(label="可视化分析", container=True)
            summary_output = gr.Markdown()

        with gr.TabItem("逐段详情"):
            table_output = gr.Dataframe(
                label="逐段分析结果",
                interactive=False,
                wrap=True,
            )

        with gr.TabItem("声纹库"):
            gr.Markdown("### 已注册说话人")
            gr.Markdown(
                "> 声纹属于敏感个人信息（《个人信息保护法》）。"
                "删除操作不可恢复，请确认后执行。"
            )
            vp_table = gr.Dataframe(
                label="说话人列表",
                interactive=False,
                wrap=True,
            )
            refresh_btn = gr.Button("刷新列表", size="sm")

            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 重命名")
                    rename_dd = gr.Dropdown(label="说话人", choices=[], interactive=True)
                    rename_input = gr.Textbox(label="新名称", placeholder="如 controller_张")
                    rename_btn = gr.Button("确认重命名", size="sm")

                with gr.Column():
                    gr.Markdown("#### 合并")
                    merge_a_dd = gr.Dropdown(label="说话人 A（保留）", choices=[], interactive=True)
                    merge_b_dd = gr.Dropdown(label="说话人 B（删除）", choices=[], interactive=True)
                    merge_btn = gr.Button("确认合并", size="sm")

            gr.Markdown("---")
            gr.Markdown("#### 删除（合规）")
            with gr.Row():
                delete_dd = gr.Dropdown(label="选择说话人删除", choices=[], interactive=True)
                delete_confirm = gr.Checkbox(label="我确认删除此说话人（不可恢复）")
                delete_btn = gr.Button("删除", variant="stop", size="sm", interactive=False)

    gr.Examples(
        examples=[
            [str(PROJECT_ROOT / "output" / "REC-1_denoised.wav")],
            [str(PROJECT_ROOT / "output" / "REC-2_denoised.wav")],
            [str(PROJECT_ROOT / "output" / "REC-3_denoised.wav")],
            [str(PROJECT_ROOT / "output" / "REC-4_denoised.wav")],
        ],
        inputs=audio_input,
        label="示例音频（管制录音降噪后）",
    )

    gr.Markdown("""
    ---
    > **技术验证版本** — 训练数据为 ExpressiveSpeech 情绪语音，非真实管制疲劳标注。
    > 实际部署需用管制场景数据微调。声纹数据遵守《个人信息保护法》相关要求。
    """)

    def _wrap_analyze(file_list, progress=gr.Progress()):
        if file_list is None or (isinstance(file_list, list) and len(file_list) == 0):
            raise gr.Error("请先上传音频文件")

        # 统一为字符串路径列表 (gr.File 可能返回 _TemporaryFileWrapper)
        def _to_path(fp):
            if isinstance(fp, str):
                return fp
            if hasattr(fp, "name"):
                return fp.name
            return str(fp)

        if isinstance(file_list, str):
            paths = [file_list]
        else:
            paths = [_to_path(fp) for fp in file_list]

        try:
            df, summary, fig, df2 = analyze_audio(paths, progress)
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"分析失败: {e}")

        n_files = len(paths)
        if df is not None and len(df) > 0:
            n_seg = len(df)
            a_mean = df["arousal"].mean()
            e_mean = df["exertion"].mean()
            n_fatigue = (df["状态"] == "疲劳").sum()
            n_excited = (df["状态"] == "兴奋").sum()
            sp_set = set(df["说话人"])
            n_sp = len(sp_set)

            import re
            tm_match = re.search(r"处理耗时 \| ([\d.]+)s", summary)
            proc_time = tm_match.group(1) + "s" if tm_match else "--"
            dur_match = re.search(r"总时长 \| ([\d.]+)s", summary)
            dur_str = dur_match.group(1) + "s" if dur_match else "--"

            fatigue_class = "metric-danger" if n_fatigue > 0 else "metric-ok"
        else:
            n_seg, n_sp, a_mean, e_mean = 0, 0, 0, 0
            n_fatigue, n_excited = 0, 0
            dur_str, proc_time = "--", "--"
            fatigue_class = "metric-ok"

        def _card(val, label, cls=""):
            return f"""<div class="metric-card"><p class="metric-value {cls}">{val}</p><p class="metric-label">{label}</p></div>"""

        return (
            df, summary, fig, df2,
            _card(n_files, "文件数"),
            _card(dur_str, "总时长"),
            _card(n_seg, "VAD 段数"),
            _card(f"{a_mean:.3f}", "arousal 均值"),
            _card(f"{e_mean:.3f}", "exertion 均值"),
            _card(n_fatigue, "疲劳段", fatigue_class),
            _card(n_excited, "兴奋段"),
            _card(proc_time, "处理耗时"),
        )

    analyze_btn.click(
        fn=_wrap_analyze,
        inputs=[audio_input],
        outputs=[
            table_output, summary_output, plot_output, table_output,
            metric_files, metric_audio, metric_seg, metric_arousal,
            metric_exertion, metric_fatigue, metric_excited, metric_time,
        ],
    )

    # ━━━ 声纹库管理事件 ━━━

    def _refresh_vp_table():
        """刷新声纹库表格 + 下拉选项"""
        rows = voiceprint_db.summary()
        if not rows:
            df = pd.DataFrame(columns=["ID", "标签", "样本数", "阶段", "平均相似度", "首次出现", "最后活跃"])
            choices = []
            return df, choices, choices, choices, choices

        data = []
        choices = []
        for r in rows:
            sid = r["speaker_id"]
            label = r["label"]
            display = f"{sid} ({label})" if label != sid else sid
            choices.append(display)
            data.append([
                sid, label, r["n_samples"], r["stage"],
                f"{r['avg_similarity']:.4f}" if r["avg_similarity"] else "--",
                r["first_seen"], r["last_seen"],
            ])
        df = pd.DataFrame(data, columns=[
            "ID", "标签", "样本数", "阶段", "平均相似度", "首次出现", "最后活跃"
        ])
        return df, choices, choices, choices, choices

    def _parse_speaker_id(display_str):
        """从下拉选项中提取 speaker_id (如 'persist_001 (controller_张)' → 'persist_001')"""
        if not display_str:
            return ""
        return display_str.split(" ")[0].split("(")[0].strip()

    def _do_rename(speaker_display, new_label):
        if not speaker_display or not new_label.strip():
            raise gr.Error("请选择说话人并输入新名称")
        sid = _parse_speaker_id(speaker_display)
        voiceprint_db.rename(sid, new_label.strip())
        voiceprint_db.save()
        df, c1, c2, c3, c4 = _refresh_vp_table()
        gr.Info(f"已重命名: {sid} -> {new_label.strip()}")
        return df, c1, c2, c3, c4

    def _do_merge(a_display, b_display):
        if not a_display or not b_display:
            raise gr.Error("请选择两个说话人")
        if a_display == b_display:
            raise gr.Error("不能合并同一个说话人")
        id_a = _parse_speaker_id(a_display)
        id_b = _parse_speaker_id(b_display)
        voiceprint_db.merge(id_a, id_b)
        voiceprint_db.save()
        df, c1, c2, c3, c4 = _refresh_vp_table()
        gr.Info(f"已合并: {id_b} -> {id_a}")
        return df, c1, c2, c3, c4

    def _do_delete(speaker_display, confirmed):
        if not speaker_display:
            raise gr.Error("请选择要删除的说话人")
        if not confirmed:
            raise gr.Error("请先勾选确认框")
        sid = _parse_speaker_id(speaker_display)
        voiceprint_db.delete(sid)
        voiceprint_db.save()
        df, c1, c2, c3, c4 = _refresh_vp_table()
        gr.Info(f"已删除: {sid}")
        return df, c1, c2, c3, c4, False  # False = 取消勾选

    def _toggle_delete_btn(confirmed):
        return gr.update(interactive=confirmed)

    refresh_btn.click(
        fn=_refresh_vp_table,
        outputs=[vp_table, rename_dd, merge_a_dd, merge_b_dd, delete_dd],
    )
    rename_btn.click(
        fn=_do_rename,
        inputs=[rename_dd, rename_input],
        outputs=[vp_table, rename_dd, merge_a_dd, merge_b_dd, delete_dd],
    )
    merge_btn.click(
        fn=_do_merge,
        inputs=[merge_a_dd, merge_b_dd],
        outputs=[vp_table, rename_dd, merge_a_dd, merge_b_dd, delete_dd],
    )
    delete_confirm.change(fn=_toggle_delete_btn, inputs=[delete_confirm], outputs=[delete_btn])
    delete_btn.click(
        fn=_do_delete,
        inputs=[delete_dd, delete_confirm],
        outputs=[vp_table, rename_dd, merge_a_dd, merge_b_dd, delete_dd, delete_confirm],
    )

    # 页面加载时自动刷新声纹库
    demo.load(fn=_refresh_vp_table, outputs=[vp_table, rename_dd, merge_a_dd, merge_b_dd, delete_dd])


if __name__ == "__main__":
    print("[launch] Gradio Demo 启动 http://0.0.0.0:7860")
    demo.queue(concurrency_count=3).launch(
        server_name="0.0.0.0", server_port=7860, share=False
    )
