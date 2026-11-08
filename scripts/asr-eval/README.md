# ASR evaluation: meetings vs. film/TV subtitles

**Results and recommendations: [REPORT.md](REPORT.md)**

Compares sherpa-onnx offline ASR models on two scenarios. The test clips come
from YouTube videos that carry **human-made** Chinese subtitles, which serve as
the reference text.

| Scenario | Clips (`manifest.json`) | What it stresses |
|---|---|---|
| `meeting` | round-tables, podcasts, interview, talk | overlapping speech, accents, disfluency |
| `film` | TV dramas (modern/costume/sci-fi), action & comedy movies | music, effects, emotional speech |

## Setup

```bash
cd scripts/asr-eval
uv venv work/.venv --python 3.12
VIRTUAL_ENV=work/.venv uv pip install sherpa-onnx==2.0.1 yt-dlp soundfile numpy zhconv rapidfuzz pytest
```

Requires `ffmpeg` on PATH and `third_party/Fun-ASR` (for `tools/cn_tn.py`).

Models go in `work/models/`. FunASR-Nano is also looked up in the repo root.
Download from <https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>:
`silero_vad.onnx` plus the model directories named in `eval_asr.py`.

## Run

```bash
work/.venv/Scripts/python -m pytest -q          # metric unit tests
work/.venv/Scripts/python fetch_material.py     # download clips + refs
work/.venv/Scripts/python fetch_material.py --verify
work/.venv/Scripts/python eval_asr.py           # all models x all clips
work/.venv/Scripts/python eval_asr.py --report  # tables only
work/.venv/Scripts/python make_report.py        # regenerate REPORT.md
```

`REPORT.md` is generated: parameter counts, precision mix and sizes are
read from the ONNX files, and metrics (including the
substitution/deletion/insertion breakdown) from `work/results`. It needs the
`onnx` package (`uv pip install onnx`).

Outputs: `work/results/<model>/<clip>.srt` (hypothesis subtitles) and
`<clip>.json` (metrics).

## Metrics

- **CER**: Levenshtein distance over tokens (CJK chars plus latin words),
  computed after traditional-to-simplified conversion and `cn_tn` normalization
  (digits spelled out, punctuation and fillers removed).
- **RTF**: decode time divided by audio duration, on CPU with
  `ASR_EVAL_THREADS` threads (default: all cores). Model load time is excluded
  but recorded.
- **halluc/min**: characters per minute emitted where the reference has no
  subtitle (music, effects, silence).
- **offset**: median start-time error against the reference cues. All models
  share one cached VAD segmentation, so this measures VAD, not the model.

### Caveats

- Subtitles are not strict verbatim transcripts: they drop repetitions
  ("就就就"), fillers and some overlapping speech. Absolute CER is therefore
  inflated, but every model is affected equally, so the ranking still holds.
- 8 minutes per clip and 5 clips per scenario is a screening benchmark, not a
  definitive one.
