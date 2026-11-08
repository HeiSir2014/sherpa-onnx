"""Tests for eval_asr.py helpers. Run: work/.venv/Scripts/python -m pytest -q"""
import numpy as np

import eval_asr

CLIP = {"name": "clip"}


def test_vad_cache_path_changes_with_audio():
    a = np.zeros(16000, dtype=np.float32)
    b = a.copy()
    b[100] = 0.5  # e.g. the clip was re-fetched with a different offset
    assert eval_asr.vad_cache_path(CLIP, a) != eval_asr.vad_cache_path(CLIP, b)


def test_vad_cache_path_changes_with_vad_params(monkeypatch):
    samples = np.zeros(16000, dtype=np.float32)
    before = eval_asr.vad_cache_path(CLIP, samples)
    monkeypatch.setitem(eval_asr.VAD_PARAMS, "max_speech_duration", 10)
    assert eval_asr.vad_cache_path(CLIP, samples) != before


def test_vad_cache_path_is_stable_for_same_inputs():
    samples = np.ones(16000, dtype=np.float32)
    assert (eval_asr.vad_cache_path(CLIP, samples)
            == eval_asr.vad_cache_path(CLIP, samples.copy()))
