"""VoiceprintDB 单元测试"""
import sys, os, tempfile, numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from step0_voiceprint import VoiceprintDB


@pytest.fixture
def db():
    """临时空声纹库"""
    tmpdir = tempfile.mkdtemp()
    d = VoiceprintDB(os.path.join(tmpdir, "test_db.json"))
    d.load()
    return d


@pytest.fixture
def emb():
    """归一化的随机 192d embedding"""
    e = np.random.randn(192).astype(np.float32)
    return e / np.linalg.norm(e)


@pytest.fixture
def emb2():
    """与 emb 正交的第二个 embedding"""
    e = np.random.randn(192).astype(np.float32)
    # 确保与 emb 正交
    e = e - np.dot(e, emb()) * emb() if 'emb' in dir() else e
    n = np.linalg.norm(e)
    return e / n if n > 1e-6 else (e + 0.1)


class TestVoiceprintDBLifecycle:
    """生命周期: load / save"""

    def test_empty_load(self, db):
        assert len(db) == 0

    def test_save_and_reload(self, db, emb):
        sid = db.match_and_update(emb)
        db.save()
        assert os.path.exists(db.db_path)

        db2 = VoiceprintDB(db.db_path)
        db2.load()
        assert len(db2) == 1
        assert sid in db2.speakers

    def test_corrupt_json_recovery(self, db, emb):
        db.match_and_update(emb)
        db.save()
        # 写入损坏 JSON
        with open(db.db_path, "w") as f:
            f.write("{ broken")
        db2 = VoiceprintDB(db.db_path)
        db2.load()
        assert len(db2) == 0  # 重建空库


class TestVoiceprintDBMatch:
    """匹配逻辑"""

    def test_register_new(self, db, emb):
        result = db.match(emb)
        assert result.is_new
        assert result.speaker_id == ""

    def test_match_existing(self, db, emb):
        sid = db.match_and_update(emb)
        result = db.match(emb)
        assert not result.is_new
        assert result.speaker_id == sid
        assert result.similarity > 0.99

    def test_match_and_update_increments(self, db, emb):
        sid = db.match_and_update(emb)
        assert db.speakers[sid]["n_samples"] == 1
        db.match_and_update(emb)
        assert db.speakers[sid]["n_samples"] == 2

    def test_different_speakers(self, db, emb, emb2):
        # 确保两个 embedding 差异够大
        e1 = np.random.randn(192).astype(np.float32)
        e1 /= np.linalg.norm(e1)
        e2 = -e1  # 取反确保正交
        sid1 = db.match_and_update(e1)
        sid2 = db.match_and_update(e2)
        assert sid1 != sid2

    def test_zero_vector_safe(self, db):
        zero = np.zeros(192, dtype=np.float32)
        result = db.match(zero)
        assert result.is_uncertain

    def test_match_history_capped(self, db, emb):
        sid = db.match_and_update(emb)
        for _ in range(60):
            noisy = emb + np.random.randn(192).astype(np.float32) * 0.04
            noisy /= np.linalg.norm(noisy)
            db.match_and_update(noisy)
        assert len(db.speakers[sid]["match_history"]) <= 50


class TestVoiceprintDBThreshold:
    """自适应阈值"""

    @pytest.mark.parametrize("n,expected", [
        (1, 0.62),    # cold
        (5, 0.62),    # cold boundary
        (6, 0.58),    # warm
        (30, 0.58),   # warm boundary
        (31, 0.52),   # stable
        (100, 0.52),  # stable
    ])
    def test_adaptive_threshold(self, db, n, expected):
        assert abs(db._get_threshold(n) - expected) < 0.01


class TestVoiceprintDBManagement:
    """管理操作"""

    def test_rename(self, db, emb):
        sid = db.match_and_update(emb)
        db.rename(sid, "controller_zhang")
        assert db.speakers[sid]["label"] == "controller_zhang"
        assert db.speakers[sid]["registered"] is True

    def test_merge(self, db):
        e1 = np.random.randn(192).astype(np.float32)
        e1 /= np.linalg.norm(e1)
        e2 = -e1
        sid1 = db.match_and_update(e1)
        sid2 = db.match_and_update(e2)
        # 给 sid2 添加样本
        db.match_and_update(e2)
        merged = db.merge(sid1, sid2)
        assert merged == sid1
        assert sid2 not in db.speakers
        assert db.speakers[sid1]["n_samples"] == 3  # 1 + 2

    def test_delete(self, db, emb):
        sid = db.match_and_update(emb)
        assert sid in db.speakers
        db.delete(sid)
        assert sid not in db.speakers

    def test_delete_callback(self, db, emb):
        sid = db.match_and_update(emb)
        deleted = []
        db.on_delete(lambda s: deleted.append(s))
        db.delete(sid)
        assert deleted == [sid]

    def test_summary(self, db, emb):
        db.match_and_update(emb)
        s = db.summary()
        assert len(s) == 1
        assert s[0]["speaker_id"].startswith("persist_")
        assert s[0]["stage"] == "cold"
