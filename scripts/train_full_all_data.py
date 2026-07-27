"""全量训练 — 全部 10 个 parquet → openSMILE → XGBoost → 详细结果 Excel
输出到 output/step4_full_training_results.xlsx
"""
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import io, warnings, os, glob, time
import soundfile as sf
import opensmile
from pathlib import Path

DATA_DIR  = Path(r'C:\Users\admin\Downloads\疲劳识别\data')
OUTPUT    = Path(r'C:\Users\admin\Downloads\疲劳识别\output')
MODEL_DIR = Path(r'C:\Users\admin\Downloads\疲劳识别\step4_regression\models')

os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ━━━ 辅助函数 ━━━
def _classify_feature(name: str) -> str:
    if "F0" in name or "F0semitone" in name:
        return "F0基频"
    if "jitter" in name.lower():
        return "jitter嗓音"
    if "shimmer" in name.lower():
        return "shimmer振幅"
    if "HNR" in name:
        return "HNR谐噪比"
    if "loudness" in name.lower():
        return "loudness响度"
    if "mfcc" in name.lower():
        return "MFCC"
    if "F1" in name or "F2" in name or "F3" in name:
        return "formant共振峰"
    if "Voiced" in name or "voice" in name.lower():
        return "rhythm节奏"
    if "slope" in name.lower() or "alphaRatio" in name or "H1" in name:
        return "spectralTilt谱倾斜"
    if "spectralFlux" in name.lower():
        return "spectral频谱"
    return "other其他"

def ccc(y_true, y_pred):
    mu_t, mu_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(ddof=0), y_pred.var(ddof=0)
    cov = ((y_true - mu_t) * (y_pred - mu_p)).mean()
    denom = var_t + var_p + (mu_t - mu_p)**2
    return float(2*cov/denom) if denom > 1e-12 else 0.0

# ━━━ 1. 加载所有 parquet ━━━
parquet_files = sorted(DATA_DIR.glob("train-*.parquet"))
print(f"找到 {len(parquet_files)} 个 parquet 文件")
total_rows = 0
for f in parquet_files:
    n = pq.read_metadata(str(f)).num_rows
    total_rows += n
    print(f"  {f.name}: {n} 行")
print(f"总计: {total_rows} 条\n")

# ━━━ 2. 全量提取 openSMILE ━━━
print("初始化 openSMILE eGeMAPSv02 ...")
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
dummy = np.zeros(16000, dtype=np.float32)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    feat_names = list(smile.process_signal(dummy, 16000).columns)

features = []
y_arousal = []
y_nature = []
skipped = 0
t_start = time.time()

print("全量提取特征中...")
for pf in parquet_files:
    df = pq.read_table(str(pf)).to_pandas()
    for i in range(len(df)):
        try:
            row = df.iloc[i]
            audio, sr = sf.read(io.BytesIO(row['audio-path']['bytes']))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=16000)
            if len(audio) < sr * 0.3:
                skipped += 1; continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feat = smile.process_signal(audio.astype(np.float32), sr)
            features.append(feat.iloc[0].values)
            y_arousal.append(row['score_arousal'])
            y_nature.append(row['score_nature'])
        except:
            skipped += 1
            continue
    elapsed = time.time() - t_start
    print(f"  {pf.name} ✓  累计 {len(features)} 条  ({elapsed:.0f}s)", flush=True)

X = np.array(features, dtype=np.float32)
ya = np.array(y_arousal, dtype=np.float32)
yn = np.array(y_nature, dtype=np.float32)

print(f"\n完成: {len(X)} 条, 跳过 {skipped}, 耗时 {time.time()-t_start:.0f}s")
print(f"arousal: [{ya.min():.3f}, {ya.max():.3f}], mean={ya.mean():.3f}")
print(f"nature:  [{yn.min():.3f}, {yn.max():.3f}], mean={yn.mean():.3f}")

# ━━━ 3. 训练双模型 + SHAP ━━━
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import shap

all_metrics = []
all_shap = {}       # 存每个 task 的 SHAP rows
all_group = {}      # 存每个 task 的特征组汇总
all_preds = {}      # 存每个 task 的预测 vs 真实

for task_name, y_all, label_name in [
    ("arousal", ya, "score_arousal"),
    ("exertion", yn, "score_nature"),
]:
    print(f"\n{'='*60}")
    print(f"训练 {task_name} 模型 ...")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_all, test_size=0.2, random_state=42)
    print(f"  训练: {len(X_tr)}, 测试: {len(X_te)}")

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, objective='reg:squarederror',
        random_state=42, verbosity=0,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    metrics = {
        "任务": task_name,
        "标签": label_name,
        "样本数": len(X),
        "训练集": len(X_tr),
        "测试集": len(X_te),
        "训练时间(s)": f"{train_time:.1f}",
        "训练R²": round(r2_score(y_tr, pred_tr), 4),
        "测试R²": round(r2_score(y_te, pred_te), 4),
        "训练MSE": round(mean_squared_error(y_tr, pred_tr), 6),
        "测试MSE": round(mean_squared_error(y_te, pred_te), 6),
        "训练MAE": round(mean_absolute_error(y_tr, pred_tr), 4),
        "测试MAE": round(mean_absolute_error(y_te, pred_te), 4),
        "训练CCC": round(ccc(y_tr, pred_tr), 4),
        "测试CCC": round(ccc(y_te, pred_te), 4),
        "训练Pearson": round(np.corrcoef(y_tr, pred_tr)[0,1], 4),
        "测试Pearson": round(np.corrcoef(y_te, pred_te)[0,1], 4),
        "预测均值": round(float(pred_te.mean()), 4),
        "真实均值": round(float(y_te.mean()), 4),
        "预测std": round(float(pred_te.std()), 4),
        "真实std": round(float(y_te.std()), 4),
    }
    all_metrics.append(metrics)
    print(f"  测试 CCC={metrics['测试CCC']}, R²={metrics['测试R²']}, MAE={metrics['测试MAE']}")

    # SHAP
    print(f"  计算 SHAP ...", flush=True)
    explainer = shap.TreeExplainer(model)
    n_shap = min(500, len(X_te))
    sv = explainer.shap_values(X_te[:n_shap])
    importance = np.abs(sv).mean(axis=0)
    top30 = np.argsort(importance)[::-1][:30]

    shap_rows = []
    for rank, idx in enumerate(top30, 1):
        name = feat_names[idx]
        val = importance[idx]
        corr = np.corrcoef(X_te[:n_shap, idx], sv[:, idx])[0, 1]
        shap_rows.append({
            "排名": rank,
            "特征": name,
            "SHAP贡献": round(float(val), 6),
            "方向": "+" if corr > 0 else "-",
            "特征组": _classify_feature(name),
        })

    all_shap[task_name] = shap_rows
    df_shap = pd.DataFrame(shap_rows)
    all_group[task_name] = df_shap.groupby("特征组")["SHAP贡献"].sum().sort_values(ascending=False)
    all_preds[task_name] = (pred_te, y_te)

    # 保存模型
    model.save_model(MODEL_DIR / f"model_{task_name}.json")
    print(f"  模型已保存")

# ━━━ 4. 输出 Excel ━━━
excel_path = OUTPUT / "step4_full_training_results.xlsx"
print(f"\n输出 Excel 到 {excel_path} ...", flush=True)

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Sheet 1: 汇总指标
    pd.DataFrame(all_metrics).to_excel(writer, sheet_name="汇总指标", index=False)

    # Sheet 2-3: SHAP top-30 per task
    for task_name in ["arousal", "exertion"]:
        pd.DataFrame(all_shap[task_name]).to_excel(
            writer, sheet_name=f"SHAP_{task_name}", index=False)

    # Sheet 4-5: 特征组汇总
    for task_name in ["arousal", "exertion"]:
        all_group[task_name].to_excel(
            writer, sheet_name=f"特征组_{task_name}", header=["SHAP贡献"])

    # Sheet 6: 预测 vs 真实
    pred_a, ya_true = all_preds["arousal"]
    pred_e, yn_true = all_preds["exertion"]
    df_pred = pd.DataFrame({
        "真实arousal": ya_true,
        "预测arousal": pred_a,
        "arousal误差": pred_a - ya_true,
        "真实nature": yn_true,
        "预测exertion": pred_e,
        "exertion误差": pred_e - yn_true,
    })
    df_pred.head(500).to_excel(writer, sheet_name="预测vs真实(前500)", index=False)

    # Sheet 7: 训练参数
    pd.DataFrame([
        {"参数": "n_estimators", "值": 300},
        {"参数": "max_depth", "值": 6},
        {"参数": "learning_rate", "值": 0.05},
        {"参数": "subsample", "值": 0.8},
        {"参数": "特征维度", "值": 88},
        {"参数": "特征集", "值": "eGeMAPSv02"},
        {"参数": "数据来源", "值": "ExpressiveSpeech (FreedomIntelligence)"},
        {"参数": "总样本数", "值": len(X)},
    ]).to_excel(writer, sheet_name="训练参数", index=False)

print(f"\n{'='*60}")
print("详细结果已保存至:", excel_path)
print(f"\n=== 全量训练汇总 ===")
for m in all_metrics:
    print(f"  {m['任务']:>10s}: CCC={m['测试CCC']}, R²={m['测试R²']}, Pearson={m['测试Pearson']}, MAE={m['测试MAE']}")
