"""VoiceprintDB - 持久化声纹库

跨请求关联同一说话人，支持个体基线持续积累。

用法:
    from step0_voiceprint import VoiceprintDB

    db = VoiceprintDB("output/voiceprint_db.json")
    db.load()
    speaker_id = db.match_and_update(embedding_192d)
    db.save()

设计依据: docs/声纹库设计方案.md v0.3
"""

import json
import os
import shutil
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class MatchResult:
    """单次匹配结果"""
    speaker_id: str
    similarity: float
    is_new: bool
    is_uncertain: bool
    label: str
    n_samples: int


class VoiceprintDB:
    """持久化声纹库

    存储: output/voiceprint_db.json
    匹配: 余弦相似度, 自适应阈值
    更新: EMA 质心 + match_history 追踪
    """

    BASE_THRESHOLD = 0.55
    MAX_HISTORY = 50

    def __init__(self, db_path: str, base_threshold: float = 0.55):
        self.db_path = db_path
        self.base_threshold = base_threshold
        self.version = 1
        self.model = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
        self.speakers: Dict[str, dict] = {}
        self._next_id = 1
        self._dirty = False
        self._on_delete = None  # 回调: 删除说话人时联动清理 (如 SpeakerBaseline)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def load(self):
        """从 JSON 加载, 文件不存在则初始化空库"""
        if not os.path.exists(self.db_path):
            self.speakers = {}
            self._next_id = 1
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.version = data.get("version", 1)
            self.model = data.get("model", self.model)
            self.base_threshold = data.get("base_threshold", self.BASE_THRESHOLD)
            self.speakers = data.get("speakers", {})
            # 恢复 centroid 为 numpy array
            for sid in self.speakers:
                c = self.speakers[sid].get("centroid")
                if c and not isinstance(c, np.ndarray):
                    self.speakers[sid]["centroid"] = np.array(c, dtype=np.float32)
            # 计算 next_id
            ids = [int(sid.split("_")[1]) for sid in self.speakers if sid.startswith("persist_")]
            self._next_id = max(ids) + 1 if ids else 1
        except (json.JSONDecodeError, KeyError) as e:
            # JSON 损坏: 备份 + 重建空库
            bak = self.db_path + ".bak"
            if os.path.exists(self.db_path):
                shutil.copy2(self.db_path, bak)
                print(f"[VoiceprintDB] JSON 损坏, 已备份到 {bak}, 重建空库")
            self.speakers = {}
            self._next_id = 1

    def save(self):
        """写入 JSON (仅当有未保存修改时)"""
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        data = {
            "version": self.version,
            "model": self.model,
            "base_threshold": self.base_threshold,
            "speakers": {},
        }
        for sid, spk in self.speakers.items():
            d = dict(spk)
            if isinstance(d.get("centroid"), np.ndarray):
                d["centroid"] = d["centroid"].tolist()
            data["speakers"][sid] = d
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._dirty = False

    # ------------------------------------------------------------------
    # 自适应阈值
    # ------------------------------------------------------------------

    def _get_threshold(self, n_samples: int) -> float:
        """根据说话人已积累样本数返回匹配阈值"""
        if n_samples <= 5:
            return self.base_threshold + 0.07  # cold: 0.62
        elif n_samples <= 30:
            return self.base_threshold + 0.03  # warm: 0.58
        else:
            return self.base_threshold - 0.03  # stable: 0.52

    def _is_stale(self, speaker: dict) -> bool:
        """说话人是否长期未活跃 (>30天)"""
        last = speaker.get("last_seen", "")
        if not last:
            return False
        try:
            dt = datetime.fromisoformat(last)
            days = (datetime.now() - dt).days
            return days > 30
        except (ValueError, TypeError):
            return False

    def _effective_threshold(self, speaker: dict) -> float:
        """实际使用的阈值 (考虑漂移)"""
        base = self._get_threshold(speaker.get("n_samples", 0))
        if self._is_stale(speaker):
            base += 0.05  # 长期未活跃, 提高阈值
        return base

    # ------------------------------------------------------------------
    # 核心匹配
    # ------------------------------------------------------------------

    def match(self, embedding: np.ndarray) -> MatchResult:
        """单条匹配, 不修改库

        Returns:
            MatchResult(speaker_id, similarity, is_new, is_uncertain, label, n_samples)
        """
        if embedding is None or np.linalg.norm(embedding) < 1e-6:
            # 零向量: 无法匹配
            return MatchResult(
                speaker_id="", similarity=0.0, is_new=True,
                is_uncertain=True, label="invalid", n_samples=0,
            )

        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        best_id = None
        best_sim = -1.0

        for sid, spk in self.speakers.items():
            centroid = spk.get("centroid")
            if centroid is None or (isinstance(centroid, np.ndarray) and np.linalg.norm(centroid) < 1e-6):
                continue
            if isinstance(centroid, list):
                centroid = np.array(centroid, dtype=np.float32)
            sim = float(np.dot(emb_norm, centroid))
            if sim > best_sim:
                best_sim = sim
                best_id = sid

        if best_id is not None:
            threshold = self._effective_threshold(self.speakers[best_id])
            gray_zone = threshold - 0.05

            if best_sim >= threshold:
                # 匹配成功
                spk = self.speakers[best_id]
                return MatchResult(
                    speaker_id=best_id, similarity=best_sim,
                    is_new=False, is_uncertain=False,
                    label=spk.get("label", best_id),
                    n_samples=spk.get("n_samples", 0),
                )
            elif best_sim >= gray_zone:
                # 灰区: 不确定
                spk = self.speakers[best_id]
                return MatchResult(
                    speaker_id=best_id, similarity=best_sim,
                    is_new=False, is_uncertain=True,
                    label=spk.get("label", best_id),
                    n_samples=spk.get("n_samples", 0),
                )

        # 新说话人
        return MatchResult(
            speaker_id="", similarity=best_sim,
            is_new=True, is_uncertain=False,
            label="", n_samples=0,
        )

    def match_and_update(self, embedding: np.ndarray) -> str:
        """匹配 + 自动注册 + 更新质心, 返回 speaker_id

        灰区匹配时: 保守策略, 视为新说话人注册 (避免误合并)
        """
        result = self.match(embedding)

        if result.is_uncertain:
            # 灰区: 注册为新说话人, 但记录潜在关联
            new_id = self._register(embedding)
            # 在新说话人记录中标注可能关联
            self.speakers[new_id]["possible_match"] = result.speaker_id
            self.speakers[new_id]["possible_sim"] = result.similarity
            return new_id

        if result.is_new:
            return self._register(embedding)

        # 匹配成功: 更新质心
        self._update_centroid(result.speaker_id, embedding)
        return result.speaker_id

    def _register(self, embedding: np.ndarray, label: Optional[str] = None) -> str:
        """注册新说话人"""
        if embedding is None or np.linalg.norm(embedding) < 1e-6:
            return "invalid"

        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        sid = f"persist_{self._next_id:03d}"
        self._next_id += 1
        now = datetime.now().isoformat(timespec="seconds")
        self.speakers[sid] = {
            "label": label or f"auto_spk_{self._next_id - 1:03d}",
            "centroid": emb_norm.astype(np.float32),
            "n_samples": 1,
            "first_seen": now,
            "last_seen": now,
            "registered": False,
            "match_history": [],
        }
        self._dirty = True
        return sid

    def _update_centroid(self, speaker_id: str, embedding: np.ndarray):
        """EMA 更新质心"""
        spk = self.speakers[speaker_id]
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        n = spk["n_samples"]
        alpha = min(0.1, 1.0 / max(n, 1))
        old = spk["centroid"]
        if isinstance(old, list):
            old = np.array(old, dtype=np.float32)

        new_centroid = (1 - alpha) * old + alpha * emb_norm
        norm = np.linalg.norm(new_centroid)
        if norm > 1e-8:
            new_centroid = new_centroid / norm

        spk["centroid"] = new_centroid.astype(np.float32)
        spk["n_samples"] = n + 1
        spk["last_seen"] = datetime.now().isoformat(timespec="seconds")

        # 追加 match_history
        sim = float(np.dot(emb_norm, new_centroid))
        history = spk.get("match_history", [])
        history.append(round(sim, 4))
        if len(history) > self.MAX_HISTORY:
            history = history[-self.MAX_HISTORY:]
        spk["match_history"] = history

        self._dirty = True

    # ------------------------------------------------------------------
    # 管理操作
    # ------------------------------------------------------------------

    def rename(self, speaker_id: str, new_label: str):
        """重命名说话人"""
        if speaker_id in self.speakers:
            self.speakers[speaker_id]["label"] = new_label
            self.speakers[speaker_id]["registered"] = True
            self._dirty = True

    def merge(self, id_a: str, id_b: str) -> str:
        """合并两个说话人, 保留 id_a, 删除 id_b

        质心取加权平均, n_samples 累加, match_history 合并
        """
        if id_a not in self.speakers or id_b not in self.speakers:
            return id_a

        a = self.speakers[id_a]
        b = self.speakers[id_b]

        ca = a["centroid"] if isinstance(a["centroid"], np.ndarray) else np.array(a["centroid"], dtype=np.float32)
        cb = b["centroid"] if isinstance(b["centroid"], np.ndarray) else np.array(b["centroid"], dtype=np.float32)

        na, nb = a["n_samples"], b["n_samples"]
        total = na + nb
        merged = (ca * na + cb * nb) / total
        norm = np.linalg.norm(merged)
        if norm > 1e-8:
            merged = merged / norm

        a["centroid"] = merged.astype(np.float32)
        a["n_samples"] = total
        a["last_seen"] = max(a.get("last_seen", ""), b.get("last_seen", ""))

        # 合并 history
        ha = a.get("match_history", [])
        hb = b.get("match_history", [])
        merged_hist = (ha + hb)[-self.MAX_HISTORY:]
        a["match_history"] = merged_hist

        del self.speakers[id_b]
        self._dirty = True
        return id_a

    def delete(self, speaker_id: str):
        """删除说话人 (合规删除), 触发 _on_delete 回调联动清理"""
        if speaker_id in self.speakers:
            del self.speakers[speaker_id]
            self._dirty = True
            if self._on_delete:
                self._on_delete(speaker_id)

    def on_delete(self, callback):
        """注册删除回调 (参数: speaker_id)"""
        self._on_delete = callback

    def summary(self) -> List[dict]:
        """所有说话人状态"""
        rows = []
        for sid, spk in self.speakers.items():
            n = spk.get("n_samples", 0)
            if n <= 5:
                stage = "cold"
            elif n <= 30:
                stage = "warm"
            elif self._is_stale(spk):
                stage = "stale"
            else:
                stage = "stable"

            history = spk.get("match_history", [])
            avg_sim = round(float(np.mean(history)), 4) if history else 0.0

            rows.append({
                "speaker_id": sid,
                "label": spk.get("label", sid),
                "n_samples": n,
                "stage": stage,
                "avg_similarity": avg_sim,
                "first_seen": spk.get("first_seen", ""),
                "last_seen": spk.get("last_seen", ""),
                "registered": spk.get("registered", False),
            })
        return sorted(rows, key=lambda r: r["n_samples"], reverse=True)

    def get_speaker_ids(self) -> List[str]:
        """所有已注册的 speaker_id"""
        return list(self.speakers.keys())

    def __len__(self):
        return len(self.speakers)
