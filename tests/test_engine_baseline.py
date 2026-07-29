"""step5 规则引擎 + SpeakerBaseline 单元测试"""
import sys, os, tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from step5_rules.engine import classify_state, _compute_thresholds, classify_dataframe
from step5_rules import SpeakerBaseline


class TestClassifyState:
    """classify_state 规则引擎"""

    def setup_method(self):
        self.at = {"low": 0.37, "high": 0.47, "mean": 0.42}
        self.et = {"low": 0.80, "high": 0.90, "mean": 0.85}

    def test_excited_high_arousal(self):
        state, conf, reason = classify_state(0.60, 0.70, self.at, self.et)
        assert state == "兴奋"
        assert conf == "high"

    def test_fatigue_low_arousal_high_exertion(self):
        state, conf, reason = classify_state(0.30, 0.95, self.at, self.et)
        assert state == "疲劳"
        assert conf == "high"

    def test_fatigue_low_arousal_only(self):
        state, conf, reason = classify_state(0.30, 0.75, self.at, self.et)
        assert state == "疲劳"
        assert conf == "medium"

    def test_stable_normal(self):
        state, conf, reason = classify_state(0.42, 0.85, self.at, self.et)
        assert state == "稳定"
        assert conf == "high"

    def test_default_thresholds(self):
        """无阈值时使用默认值"""
        state, _, _ = classify_state(0.90, 0.50)
        assert state == "兴奋"

    def test_reason_contains_arousal(self):
        _, _, reason = classify_state(0.60, 0.70, self.at, self.et)
        assert "arousal" in reason


class TestComputeThresholds:
    """_compute_thresholds 阈值计算"""

    def test_basic(self):
        vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        t = _compute_thresholds(vals)
        assert "low" in t
        assert "high" in t
        assert "mean" in t
        assert t["low"] < t["mean"] < t["high"]

    def test_percentile_values(self):
        vals = np.arange(1, 101, dtype=float)
        t = _compute_thresholds(vals)
        # P25 = 25.75, P75 = 75.25
        assert abs(t["low"] - 25.75) < 0.1
        assert abs(t["high"] - 75.25) < 0.1


class TestSpeakerBaseline:
    """SpeakerBaseline 个体基线"""

    def test_cold_start_uses_global(self):
        bl = SpeakerBaseline(global_mean=0.72, global_std=0.19)
        br = bl.normalize("spk_1", 0.50, 0.85)
        assert br.using_individual is False
        assert br.confidence == "low"
        assert br.speaker_n == 0

    def test_update_increments_n(self):
        bl = SpeakerBaseline(global_mean=0.72, global_std=0.19)
        for _ in range(101):  # >100 进入 stable
            bl.normalize("spk_1", 0.50, 0.85)
            bl.update("spk_1", 0.50, 0.85)
        br = bl.normalize("spk_1", 0.50, 0.85)
        assert br.confidence == "high"
        assert br.using_individual is True

    def test_persistence_roundtrip(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "bl.json")
        bl = SpeakerBaseline(global_mean=0.72, global_std=0.19)
        bl.load(path)  # 文件不存在, 优雅初始化
        for _ in range(10):
            bl.normalize("spk_1", 0.50, 0.85)
            bl.update("spk_1", 0.50, 0.85)
        bl.save(path)
        # 重新加载
        bl2 = SpeakerBaseline()
        bl2.load(path)
        assert "spk_1" in bl2._speakers
        assert bl2._speakers["spk_1"]["n"] == 10

    def test_missing_file_graceful(self):
        bl = SpeakerBaseline()
        bl.load("/nonexistent/path/bl.json")
        assert len(bl._speakers) == 0

    def test_baseline_decay(self):
        """测试滑动窗口衰减机制 (n > max_n 时触发)"""
        bl = SpeakerBaseline(global_mean=0.72, global_std=0.19)
        # 注入 250 段, 前 200 段 arousal=0.3, 后 50 段 arousal=0.8
        for _ in range(200):
            bl.normalize("spk_1", 0.3, 0.85)
            bl.update("spk_1", 0.3, 0.85)
        for _ in range(50):
            bl.normalize("spk_1", 0.8, 0.85)
            bl.update("spk_1", 0.8, 0.85)
        # 衰减后, n 应被限制在 max_n 附近 (浮点误差容忍)
        n = bl._speakers["spk_1"]["n"]
        assert n <= 201  # max_n=200 + 浮点误差

    def test_different_speakers_isolated(self):
        bl = SpeakerBaseline(global_mean=0.72, global_std=0.19)
        bl.normalize("spk_a", 0.5, 0.85)
        bl.update("spk_a", 0.5, 0.85)
        bl.normalize("spk_b", 0.9, 0.95)
        bl.update("spk_b", 0.9, 0.95)
        # 两个说话人的统计独立
        assert bl._speakers["spk_a"]["n"] == 1
        assert bl._speakers["spk_b"]["n"] == 1
