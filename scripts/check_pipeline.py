"""全链路通顺性检查 — 从原录音到最终分析"""
import sys
sys.path.insert(0, '.')
import numpy as np, soundfile as sf, warnings
warnings.filterwarnings('ignore')

print('=== 1. 导入检查 ===')
from step0_voiceprint import SpeakerClusterer;      print(' step0: SpeakerClusterer ok')
from step1_preprocess.vad import vad_detect;         print(' step1: vad_detect ok')
from step1_preprocess.denoise import denoise;         print(' step1: denoise ok (TSE=代完成)')
from step2_features.extractor import get_extractor;  print(' step2: get_extractor ok')
from step3_baseline import BaselineAligner;           print(' step3: BaselineAligner ok (reserved)')
from step4_regression.model import load_models;       print(' step4: load_models ok')
from step5_rules import SpeakerBaseline;              print(' step5: SpeakerBaseline ok')
from step5_rules.engine import classify_state, _compute_thresholds; print(' step5: engine ok')
from step6_demo.app import analyze_audio;             print(' step6: demo app ok')

print('\n=== 2. 全链路执行 ===')
models = load_models('step4_regression/models/')
baseline = SpeakerBaseline()
clusterer = SpeakerClusterer(threshold=0.55)
import opensmile
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)

audio, sr = sf.read('output/REC-1_denoised.wav')
if audio.ndim > 1: audio = audio.mean(axis=1)
print(f' RAW audio: {len(audio)/sr:.0f}s {sr}Hz')

segs = vad_detect(audio, sr, backend='silero')
seg_audios = [audio[int(s.start_sec*sr):int(s.end_sec*sr)] for s in segs]
print(f' step1 VAD: {len(segs)} segments')

speaker_ids = clusterer.fit_predict(seg_audios, sr)
print(f' step0 VOICE: {len(set(speaker_ids))} speakers: {speaker_ids}')

X = np.array([smile.process_signal(s.astype(np.float32), sr).iloc[0].values for s in seg_audios], dtype=np.float32)
print(f' step2 OSMILE: {X.shape}')

pred_a = np.clip(models['arousal'].predict(X), 0, 1)
pred_e = np.clip(models['exertion'].predict(X), 0, 1)
print(f' step4 XGBOOST: a_mean={pred_a.mean():.3f} e_mean={pred_e.mean():.3f}')

at = _compute_thresholds(pred_a)
for i in range(len(pred_a)):
    br = baseline.normalize(speaker_ids[i], pred_a[i], pred_e[i])
    baseline.update(speaker_ids[i], pred_a[i], pred_e[i])
    s, c, _ = classify_state(pred_a[i], pred_e[i], at)
    print(f' step5 [{speaker_ids[i]:>10s}]: a={pred_a[i]:.3f} z={br.arousal_z:.1f} ({br.confidence}) -> {s}')

print('\n=== 3. Demo 端到端 ===')
df, summary, fig, _ = analyze_audio('output/REC-1_denoised.wav')
print(f'Demo output: {df.shape[0]} rows x {df.shape[1]} cols')
print(f'Speakers detected: {df["说话人"].nunique()}')
print(f'Columns: {list(df.columns)}')

print('\n=== 全链路通顺: PASS ===')
