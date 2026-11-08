#!/usr/bin/env python3
"""Download evaluation clips listed in manifest.json.

For each clip, writes:
  work/data/<scenario>/<name>.wav      16 kHz mono, only [start, start+duration)
  work/data/<scenario>/<name>.ref.srt  human subtitles, shifted to clip time

Only uploader-provided subtitles are used (yt-dlp `subtitles`). YouTube's
`automatic_captions` are ASR output and must never be used as a reference.

Usage:
  python fetch_material.py            # download missing clips
  python fetch_material.py --verify   # check every clip is present and sane
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import yt_dlp

from metrics import Cue, parse_srt

HERE = Path(__file__).resolve().parent
DATA = HERE / "work" / "data"
SAMPLE_RATE = 16000


def clip_paths(clip):
    d = DATA / clip["scenario"]
    return d / f"{clip['name']}.wav", d / f"{clip['name']}.ref.srt"


def format_ts(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{format_ts(c.start)} --> {format_ts(c.end)}\n")
            f.write(f"{c.text}\n\n")


def shift_cues(cues, start, duration):
    """Keep cues inside [start, start+duration), re-based to 0."""
    end = start + duration
    out = []
    for c in cues:
        if c.end <= start or c.start >= end:
            continue
        out.append(Cue(max(c.start, start) - start, min(c.end, end) - start,
                       c.text))
    return out


def fetch(clip):
    wav_path, ref_path = clip_paths(clip)
    if wav_path.exists() and ref_path.exists():
        print(f"SKIP {clip['name']}")
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={clip['id']}"
    start, duration, lang = clip["start"], clip["duration"], clip["sub_lang"]

    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": f"{tmp}/a.%(ext)s",
            "writesubtitles": True,
            "writeautomaticsub": False,
            "subtitleslangs": [lang],
            "subtitlesformat": "srt/vtt/best",
            "download_ranges": yt_dlp.utils.download_range_func(
                None, [(start, start + duration)]),
            "force_keyframes_at_cuts": False,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        if lang not in (info.get("subtitles") or {}):
            raise RuntimeError(f"{clip['name']}: no manual '{lang}' subtitles")

        audio = [p for p in Path(tmp).glob("a.*") if p.suffix not in
                 (".srt", ".vtt", ".json3", ".srv3", ".ttml")]
        subs = list(Path(tmp).glob(f"a.{lang}.*"))
        if len(audio) != 1 or len(subs) != 1:
            raise RuntimeError(f"{clip['name']}: unexpected files "
                               f"{[p.name for p in Path(tmp).iterdir()]}")

        # The section download may start slightly before `start` (keyframe),
        # so trim precisely by duration from the end of the downloaded file.
        got = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(audio[0])]).strip())
        offset = max(0.0, got - duration)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-ss", f"{offset:.3f}",
            "-i", str(audio[0]), "-t", str(duration),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
            str(wav_path)], check=True)

        # parse_srt also reads WebVTT: same "start --> end" lines, '.' or ','
        cues = parse_srt(subs[0].read_text(encoding="utf-8"))
        write_srt(shift_cues(cues, start, duration), ref_path)
    print(f"OK   {clip['name']}  ({lang}, audio offset {offset:.2f}s)")


def verify(clip) -> bool:
    wav_path, ref_path = clip_paths(clip)
    if not wav_path.exists() or not ref_path.exists():
        print(f"MISSING {clip['name']}")
        return False
    info = sf.info(str(wav_path))
    cues = parse_srt(ref_path.read_text(encoding="utf-8"))
    ok = (info.samplerate == SAMPLE_RATE and info.channels == 1
          and abs(info.duration - clip["duration"]) < 2.0 and len(cues) > 20)
    chars = sum(len(c.text) for c in cues)
    print(f"{'OK  ' if ok else 'BAD '} {clip['name']:<28} "
          f"{info.duration:7.1f}s  {len(cues):4d} cues  {chars:5d} chars")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--manifest", default=str(HERE / "manifest.json"))
    args = p.parse_args()
    clips = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.verify:
        sys.exit(0 if all([verify(c) for c in clips]) else 1)
    for clip in clips:
        fetch(clip)


if __name__ == "__main__":
    main()
