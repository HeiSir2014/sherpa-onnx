"""Metrics for comparing ASR output against human subtitles.

- token_error_rate: Levenshtein over tokens, where a token is one CJK char or
  one latin word (a "mixed error rate", the usual CER for zh/en text).
- median_start_offset: subtitle timing quality.
- hallucination_chars_per_min: text emitted where the reference has no speech
  (e.g. music, sound effects).

Text is normalized with FunASR's cn_tn.TextNorm (third_party/Fun-ASR), plus
traditional->simplified conversion, so that digits vs. spelled-out numbers,
punctuation and fillers do not count as errors.
"""
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import zhconv
from rapidfuzz.distance import Levenshtein

_CN_TN_DIR = Path(__file__).resolve().parents[2] / "third_party" / "Fun-ASR" / "tools"
if not (_CN_TN_DIR / "cn_tn.py").is_file():
    raise ImportError(
        f"cn_tn.py not found in {_CN_TN_DIR}. Clone "
        "https://github.com/FunAudioLLM/Fun-ASR into third_party/Fun-ASR."
    )
sys.path.insert(0, str(_CN_TN_DIR))
from cn_tn import TextNorm  # noqa: E402

_TEXT_NORM = TextNorm(
    to_banjiao=True,
    to_lower=True,
    remove_fillers=True,
    remove_erhua=True,
    remove_space=False,
)
# cn_tn only drops 呃/啊; human subtitles also omit these.
_EXTRA_FILLERS = re.compile("[嗯唔]")
_TOKEN_RE = re.compile(r"[㐀-鿿]|[a-z0-9']+")
_TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")
_TIME_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)

# A hyp segment within this many seconds of a reference cue is not
# considered hallucinated (VAD boundaries vs. subtitle timing differ).
BOUNDARY_SLACK_SECONDS = 0.5


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def normalize(text: str) -> str:
    """Normalized text without punctuation or whitespace."""
    return "".join(to_tokens(text))


def to_tokens(text: str) -> List[str]:
    # Drop markup and special tokens such as <i>, <sil>, <unk>
    text = _TAG_RE.sub("", text)
    text = zhconv.convert(text, "zh-cn")
    text = _EXTRA_FILLERS.sub("", text)
    text = _TEXT_NORM(text)
    return _TOKEN_RE.findall(text.lower())


def token_error_rate(ref: str, hyp: str) -> Tuple[int, int]:
    """Return (edit errors, number of reference tokens)."""
    ref_tokens = to_tokens(ref)
    hyp_tokens = to_tokens(hyp)
    return Levenshtein.distance(ref_tokens, hyp_tokens), len(ref_tokens)


def _seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / (10 ** len(ms))


def parse_srt(content: str) -> List[Cue]:
    cues = []
    for block in re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip()):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            m = _TIME_RE.search(line)
            if not m:
                continue
            g = m.groups()
            text = " ".join(
                _TAG_RE.sub("", t).strip() for t in lines[i + 1:] if t.strip()
            )
            cues.append(Cue(_seconds(*g[:4]), _seconds(*g[4:]), text))
            break
    return cues


def median_start_offset(ref: List[Cue], hyp: List[Cue]) -> float:
    """Median |start(ref cue) - start(nearest hyp segment)| in seconds."""
    if not ref or not hyp:
        return float("nan")
    hyp_starts = [c.start for c in hyp]
    return statistics.median(
        min(abs(r.start - h) for h in hyp_starts) for r in ref
    )


def _overlaps_reference(seg: Cue, ref: List[Cue]) -> bool:
    lo = seg.start - BOUNDARY_SLACK_SECONDS
    hi = seg.end + BOUNDARY_SLACK_SECONDS
    return any(r.start < hi and r.end > lo for r in ref)


def hallucination_chars_per_min(
    ref: List[Cue], hyp: List[Cue], duration: float
) -> float:
    """Tokens emitted in hyp segments that overlap no reference cue."""
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    extra = sum(
        len(to_tokens(seg.text)) for seg in hyp
        if not _overlaps_reference(seg, ref)
    )
    return extra * 60.0 / duration
