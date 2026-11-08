"""Unit tests for metrics.py. Run: work/.venv/Scripts/python -m pytest -q"""
import pytest

from metrics import (
    Cue,
    hallucination_chars_per_min,
    median_start_offset,
    normalize,
    parse_srt,
    to_tokens,
    token_error_rate,
)


def test_normalize_converts_traditional_to_simplified():
    assert normalize("會議開始") == "会议开始"


def test_normalize_strips_punctuation_and_spaces():
    assert normalize("你好，世界！ 今天。") == "你好世界今天"


def test_normalize_spells_out_digits_like_human_subtitles():
    assert normalize("我有3个苹果") == normalize("我有三个苹果")


def test_normalize_removes_fillers():
    assert normalize("嗯我们开始吧") == normalize("我们开始吧")


def test_tokens_split_cjk_chars_and_english_words():
    expected = ["我", "用", "iphone", "十", "五", "拍", "照"]
    assert to_tokens("我用iPhone 15拍照") == expected


def test_tokens_drop_special_markers_like_sil():
    # FireRedASR2-CTC in sherpa-onnx <= 2.0.1 leaks "<sil>" into its text
    assert to_tokens("推动作用<sil><sil>") == ["推", "动", "作", "用"]


def test_token_error_rate_identical_is_zero():
    assert token_error_rate("今天天气很好", "今天天气很好") == (0, 6)


def test_token_error_rate_counts_substitution_deletion_insertion():
    # ref: 今天天气很好 (6), hyp: 今天天汽好呀 -> sub 气->汽, del 很, ins 呀
    errors, n = token_error_rate("今天天气很好", "今天天汽好呀")
    assert (errors, n) == (3, 6)


def test_token_error_rate_ignores_punctuation_and_script():
    assert token_error_rate("會議，開始。", "会议开始")[0] == 0


SRT = """1
00:00:01,000 --> 00:00:02,500
第一句

2
00:01:00,250 --> 00:01:03,000
第二句
第二行
"""


def test_parse_srt_reads_times_and_joins_lines():
    cues = parse_srt(SRT)
    assert cues == [Cue(1.0, 2.5, "第一句"), Cue(60.25, 63.0, "第二句 第二行")]


def test_parse_srt_reads_webvtt_with_cue_settings():
    vtt = ("WEBVTT\nKind: captions\n\n"
           "00:00:01.500 --> 00:00:03.000 align:start position:0%\n你好\n")
    assert parse_srt(vtt) == [Cue(1.5, 3.0, "你好")]


def test_parse_srt_strips_html_tags():
    cues = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n<i>你好</i>\n")
    assert cues[0].text == "你好"


def test_median_start_offset_uses_nearest_hyp_start():
    ref = [Cue(1.0, 2.0, "a"), Cue(10.0, 11.0, "b"), Cue(20.0, 21.0, "c")]
    hyp = [Cue(1.2, 2.0, "a"), Cue(9.0, 11.0, "b"), Cue(20.0, 21.0, "c")]
    assert median_start_offset(ref, hyp) == pytest.approx(0.2)


def test_hallucination_counts_text_outside_reference_speech():
    ref = [Cue(0.0, 10.0, "说话")]
    hyp = [Cue(1.0, 3.0, "说话"), Cue(30.0, 32.0, "多出来的字")]
    # 5 hallucinated chars over a 60 s clip -> 5 chars/min
    assert hallucination_chars_per_min(ref, hyp, duration=60.0) == 5.0


def test_hallucination_tolerates_small_boundary_slack():
    ref = [Cue(0.0, 10.0, "说话")]
    hyp = [Cue(10.3, 11.0, "尾巴")]
    assert hallucination_chars_per_min(ref, hyp, duration=60.0) == 0.0
