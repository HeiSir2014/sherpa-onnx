#!/usr/bin/env python3
"""Evaluate sherpa-onnx ASR models on the clips in manifest.json.

Pipeline per clip (same as python-api-examples/generate-subtitles.py):
  silero VAD -> one OfflineStream per speech segment -> hyp SRT.
VAD runs once per clip and is cached, so every model decodes exactly the same
segments.

Usage:
  python eval_asr.py                          # all models, all clips
  python eval_asr.py --models sense-voice fire-red-ctc --scenario film
  python eval_asr.py --report                 # print tables from results
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

from fetch_material import clip_paths, write_srt
from metrics import (
    Cue,
    hallucination_chars_per_min,
    median_start_offset,
    parse_srt,
    token_error_rate,
)

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
MODELS = WORK / "models"
REPO_ROOT = HERE.parents[1]
SAMPLE_RATE = 16000
NUM_THREADS = int(os.environ.get("ASR_EVAL_THREADS", os.cpu_count() or 4))
BATCH_SIZE = int(os.environ.get("ASR_EVAL_BATCH", 8))


def _find(directory: Path, pattern: str) -> str:
    hits = sorted(directory.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No '{pattern}' in {directory}")
    return str(hits[0])


def _model_dir(name: str) -> Path:
    for base in (MODELS, REPO_ROOT):
        if (base / name).is_dir():
            return base / name
    raise FileNotFoundError(f"Model dir '{name}' not found in {MODELS} "
                            f"or {REPO_ROOT}. See README.md for downloads.")


def _sense_voice_dir(name: str):
    d = _model_dir(name)
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=_find(d, "model*.onnx"), tokens=_find(d, "tokens.txt"),
        num_threads=NUM_THREADS, use_itn=True)


def _sense_voice():
    # 2025-09-09 = SenseVoiceSmall fine-tuned on 21.8k h of Cantonese (WSYue)
    return _sense_voice_dir(
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09")


def _sense_voice_2024():
    # Original iic/SenseVoiceSmall
    return _sense_voice_dir(
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")


def _sense_voice_funasr_nano():
    # Fun-ASR-Nano-2512 audio encoder + its CTC head, no LLM decoder
    return _sense_voice_dir("sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17")


def _fire_red_ctc():
    d = _model_dir("sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25")
    return sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
        model=_find(d, "model*.onnx"), tokens=_find(d, "tokens.txt"),
        num_threads=NUM_THREADS)


def _fire_red_aed():
    d = _model_dir("sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26")
    return sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
        encoder=_find(d, "encoder*.onnx"), decoder=_find(d, "decoder*.onnx"),
        tokens=_find(d, "tokens.txt"), num_threads=NUM_THREADS)


def _qwen3_asr():
    d = _model_dir("sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25")
    return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=_find(d, "conv_frontend*.onnx"),
        encoder=_find(d, "encoder*.onnx"), decoder=_find(d, "decoder*.onnx"),
        tokenizer=_find(d, "tokenizer*"), num_threads=NUM_THREADS)


def _funasr_nano(llm: str, **kwargs):
    d = _model_dir("sherpa-onnx-funasr-nano-int8-2025-12-30")
    return sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        encoder_adaptor=_find(d, "encoder_adaptor*.onnx"),
        llm=llm, embedding=_find(d, "embedding*.onnx"),
        tokenizer=_find(d, "Qwen3-0.6B"), num_threads=NUM_THREADS, **kwargs)


def _funasr_nano_int8_pen():
    # The sherpa-onnx int8 package with the fork's former defaults
    # (penalty 1.2, 3-gram blocking). The int8 LLM loops without them.
    d = _model_dir("sherpa-onnx-funasr-nano-int8-2025-12-30")
    return _funasr_nano(_find(d, "llm*.onnx"), repetition_penalty=1.2,
                        no_repeat_ngram_size=3)


def _funasr_nano_fp16():
    # int8 encoder/embedding with an fp16 LLM from
    # modelscope.cn/models/zengshuishui/FunASR-nano-onnx (llm_fp16/)
    return _funasr_nano(_find(MODELS / "funasr-nano-llm-fp16", "llm.fp16.onnx"))


def _paraformer():
    d = _model_dir("sherpa-onnx-paraformer-zh-int8-2025-10-07")
    return sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=_find(d, "model*.onnx"), tokens=_find(d, "tokens.txt"),
        num_threads=NUM_THREADS)


def _whisper_turbo():
    d = _model_dir("sherpa-onnx-whisper-turbo")
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=_find(d, "*encoder.int8.onnx"),
        decoder=_find(d, "*decoder.int8.onnx"),
        tokens=_find(d, "*tokens.txt"), language="zh",
        num_threads=NUM_THREADS)


MODEL_FACTORIES = {
    "sense-voice": _sense_voice,
    "sense-voice-2024": _sense_voice_2024,
    "sv-funasr-nano-ctc": _sense_voice_funasr_nano,
    "fire-red-ctc": _fire_red_ctc,
    "fire-red-aed": _fire_red_aed,
    "qwen3-asr-0.6b": _qwen3_asr,
    "funasr-nano-int8-pen": _funasr_nano_int8_pen,
    "funasr-nano-fp16": _funasr_nano_fp16,
    "paraformer": _paraformer,
    "whisper-turbo": _whisper_turbo,
}


def vad_segments(clip, samples: np.ndarray):
    """Speech segments as (start_sample, end_sample), cached per clip."""
    cache = WORK / "vad" / f"{clip['name']}.json"
    if cache.exists():
        return [tuple(s) for s in json.loads(cache.read_text())]

    config = sherpa_onnx.VadModelConfig()
    # Same settings as python-api-examples/generate-subtitles.py
    config.silero_vad.model = str(MODELS / "silero_vad.onnx")
    config.silero_vad.threshold = 0.2
    config.silero_vad.min_silence_duration = 0.25
    config.silero_vad.min_speech_duration = 0.25
    config.silero_vad.max_speech_duration = 5
    config.sample_rate = SAMPLE_RATE
    window = config.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)

    segments = []

    def drain():
        while not vad.empty():
            start = vad.front.start
            segments.append((start, start + len(vad.front.samples)))
            vad.pop()

    for i in range(0, len(samples) - window + 1, window):
        vad.accept_waveform(samples[i:i + window])
        drain()
    vad.flush()
    drain()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(segments))
    return segments


def decode_clip(recognizer, samples, segments):
    """Decode VAD segments in small batches to bound peak memory.

    Decoding all ~100 segments of a clip in one decode_streams() call made
    the larger models exhaust system RAM.
    """
    hyp = []
    elapsed = 0.0
    for i in range(0, len(segments), BATCH_SIZE):
        batch = segments[i:i + BATCH_SIZE]
        streams = []
        for begin, end in batch:
            s = recognizer.create_stream()
            s.accept_waveform(SAMPLE_RATE, samples[begin:end])
            streams.append(s)
        t0 = time.perf_counter()
        recognizer.decode_streams(streams)
        elapsed += time.perf_counter() - t0
        hyp += [Cue(b / SAMPLE_RATE, e / SAMPLE_RATE, s.result.text.strip())
                for (b, e), s in zip(batch, streams)]
    return [c for c in hyp if c.text], elapsed


def text_metrics(ref, hyp, duration):
    errors, n_ref = token_error_rate(" ".join(c.text for c in ref),
                                     " ".join(c.text for c in hyp))
    return {
        "errors": errors,
        "ref_tokens": n_ref,
        "cer": errors / max(n_ref, 1),
        "start_offset": median_start_offset(ref, hyp),
        "halluc_per_min": hallucination_chars_per_min(ref, hyp, duration),
    }


def rescore():
    """Recompute text metrics from saved hyp SRTs; keeps measured RTF."""
    for path in sorted((WORK / "results").glob("*/*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        hyp_path = path.with_suffix(".srt")
        clip = {"scenario": result["scenario"], "name": result["clip"]}
        ref_path = clip_paths(clip)[1]
        ref = parse_srt(ref_path.read_text(encoding="utf-8"))
        hyp = parse_srt(hyp_path.read_text(encoding="utf-8"))
        result.update(text_metrics(ref, hyp, result["duration"]))
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("rescored", flush=True)


def evaluate(model_name, clips, force=False):
    out_dir = WORK / "results" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        clips = [c for c in clips
                 if not (out_dir / f"{c['name']}.json").exists()]
    if not clips:
        print(f"\n=== {model_name}: all clips done, skipping (use --force)")
        return
    print(f"\n=== {model_name} (threads={NUM_THREADS}, batch={BATCH_SIZE}, "
          f"{len(clips)} clips)", flush=True)
    t0 = time.perf_counter()
    recognizer = MODEL_FACTORIES[model_name]()
    load_seconds = time.perf_counter() - t0

    for clip in clips:
        wav_path, ref_path = clip_paths(clip)
        samples, sr = sf.read(str(wav_path), dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{wav_path}: expected {SAMPLE_RATE} Hz, got {sr}")
        duration = len(samples) / SAMPLE_RATE
        ref = parse_srt(ref_path.read_text(encoding="utf-8"))
        segments = vad_segments(clip, samples)

        hyp, elapsed = decode_clip(recognizer, samples, segments)
        write_srt(hyp, out_dir / f"{clip['name']}.srt")

        result = {
            "model": model_name,
            "clip": clip["name"],
            "scenario": clip["scenario"],
            "duration": duration,
            "rtf": elapsed / duration,
            "load_seconds": load_seconds,
            **text_metrics(ref, hyp, duration),
        }
        (out_dir / f"{clip['name']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1))
        print(f"  {clip['name']:<28} CER {result['cer']:6.2%}  "
              f"RTF {result['rtf']:.3f}  offset {result['start_offset']:.2f}s"
              f"  halluc {result['halluc_per_min']:.1f}/min", flush=True)


def report(scenario_filter=None):
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in (WORK / "results").glob("*/*.json")]
    for scenario in sorted({r["scenario"] for r in rows}):
        if scenario_filter and scenario != scenario_filter:
            continue
        sub = [r for r in rows if r["scenario"] == scenario]
        clips = sorted({r["clip"] for r in sub})
        print(f"\n## {scenario}  ({len(clips)} clips)")
        print(f"{'model':<16} {'CER':>7} {'RTF':>6} {'offset':>7} "
              f"{'halluc/min':>10}  per-clip CER")
        table = []
        for model in sorted({r["model"] for r in sub}):
            mr = {r["clip"]: r for r in sub if r["model"] == model}
            if set(mr) != set(clips):
                continue  # only rank models that ran on every clip
            errors = sum(r["errors"] for r in mr.values())
            n_ref = sum(r["ref_tokens"] for r in mr.values())
            dur = sum(r["duration"] for r in mr.values())
            rtf = sum(r["rtf"] * r["duration"] for r in mr.values()) / dur
            off = float(np.median([r["start_offset"] for r in mr.values()]))
            hal = sum(r["halluc_per_min"] * r["duration"]
                      for r in mr.values()) / dur
            per = " ".join(f"{mr[c]['cer']:.1%}" for c in clips)
            table.append((errors / n_ref, model, rtf, off, hal, per))
        for cer, model, rtf, off, hal, per in sorted(table):
            print(f"{model:<16} {cer:7.2%} {rtf:6.3f} {off:6.2f}s "
                  f"{hal:10.1f}  {per}")
        print(f"clips: {' | '.join(clips)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="*", default=list(MODEL_FACTORIES),
                   choices=list(MODEL_FACTORIES))
    p.add_argument("--scenario", choices=["meeting", "film"])
    p.add_argument("--clips", nargs="*", help="clip names (default: all)")
    p.add_argument("--report", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="re-run (model, clip) pairs that already have results")
    p.add_argument("--rescore", action="store_true",
                   help="recompute text metrics from saved SRTs, then report")
    args = p.parse_args()
    if args.rescore:
        rescore()
        report(args.scenario)
        return
    if args.report:
        report(args.scenario)
        return
    clips = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    clips = [c for c in clips
             if (not args.scenario or c["scenario"] == args.scenario)
             and (not args.clips or c["name"] in args.clips)]
    if not clips:
        raise SystemExit("No clips selected")
    for model in args.models:
        evaluate(model, clips, force=args.force)
    report(args.scenario)


if __name__ == "__main__":
    main()
