# ASR evaluation report (2026-11-08)

10 clips × 8 min from official YouTube channels with human Chinese subtitles,
5 meeting-like and 5 film/TV. sherpa-onnx fork at `8cdb8be0`, CPU only
(6 threads, 6-core desktop CPU), silero VAD segments shared by all models.
Run `python eval_asr.py --report` to regenerate these tables from `work/results`.

## Meeting (round-tables, podcast, interview, talk)

| Model | CER | RTF | 1 h audio takes |
|---|---|---|---|
| **FireRedASR2-AED** int8 | **9.08%** | 0.66 | 40 min |
| Qwen3-ASR-0.6B int8 | 9.62% | 0.41 | 25 min |
| **SenseVoice-FunASR-Nano-CTC** int8 | 10.52% | **0.033** | **2 min** |
| SenseVoice 2025-09-09 (Cantonese fine-tune) | 10.95% | 0.10 | 6 min |
| SenseVoice 2024-07-17 (original) | 10.99% | 0.059 | 3.5 min |
| FireRedASR2-CTC int8 | 11.08% | 0.48 | 29 min |
| FunASR-Nano, fp16 LLM (fixed) | 12.02% | 1.02 | 61 min |
| Paraformer-zh int8 | 13.24% | 0.10 | 6 min |
| Whisper-turbo int8 | 19.84% | 1.47* | 88 min |
| FunASR-Nano, int8 LLM + penalties (previous fork default) | 66.58% | 0.31 | — |

## Film / TV (modern drama, costume drama, sci-fi, action, comedy)

| Model | CER | RTF | Hallucination (chars/min) |
|---|---|---|---|
| **FireRedASR2-AED** int8 | **14.46%** | 0.38 | 1.3 |
| Qwen3-ASR-0.6B int8 | 16.25% | 0.23 | 1.2 |
| FireRedASR2-CTC int8 | 17.90% | 0.26 | 0.9 |
| **SenseVoice 2024-07-17 (original)** | 18.06% | **0.032** | 1.6 |
| SenseVoice-FunASR-Nano-CTC int8 | 18.75% | **0.025** | 1.2 |
| SenseVoice 2025-09-09 (Cantonese fine-tune) | 19.06% | 0.050 | 1.2 |
| Paraformer-zh int8 | 21.97% | 0.046 | 1.4 |
| FunASR-Nano, fp16 LLM (fixed) | 24.36% | 0.54 | 3.3 |
| Whisper-turbo int8 | 36.74% | 0.57 | 2.2 |
| FunASR-Nano, int8 LLM + penalties (previous fork default) | 70.50% | 0.18 | 2.7 |

\* Whisper's meeting RTF is inflated: 4 of its 5 meeting clips ran while
the fork was being rebuilt.

## Recommendations

| Use case | Choice | Why |
|---|---|---|
| Live meeting captions | SenseVoice-FunASR-Nano-CTC + VAD | 1.4 pt behind the best model at 1/20 of the cost; RTF 0.033 leaves the CPU almost idle |
| Post-meeting minutes | FireRedASR2-AED | Best on all 5 meeting clips. Use Qwen3-ASR if you need hotwords (FireRed has none in sherpa-onnx) |
| Film/TV subtitles, quality | FireRedASR2-AED | Best overall, and best on costume and sci-fi |
| Film/TV subtitles, cost | SenseVoice 2024-07-17 | RTF 0.032. CTC gives real token timestamps for subtitle timing |
| Avoid | FunASR-Nano (LLM), Whisper-turbo, SenseVoice 2025-09-09 for Mandarin | Slowest or least accurate; the 2025-09-09 package is a Cantonese fine-tune |

## Bugs found and fixed during this evaluation

- `be8cc495` FireRedASR2-CTC leaked `<sil>` tokens into the text.
- `869f35ce` FunASR-Nano repetition penalty and n-gram blocking were masking
  a degraded int8 LLM and cost accuracy on a healthy one. Both are now opt-in.
- `8cdb8be0` FunASR-Nano could loop until `max_new_tokens`. Generation is now
  capped by audio duration. This took one clip from 33.6% to 15.4% CER.
- `f3460762` The Python FunASR-Nano default prompt used an ASCII colon.
- `99bb5ef6` Python `OfflineTtsModelConfig` binding broken by the upstream
  merge.

Root cause of the FunASR-Nano numbers: the int8 LLM in the sherpa-onnx
package `sherpa-onnx-funasr-nano-int8-2025-12-30`. Swapping only the LLM for
fp16 (`modelscope.cn/models/zengshuishui/FunASR-nano-onnx`, `llm_fp16/`) took
a 101 s window from 134.8% to 24.9% CER.

## Caveats

- References are human subtitles, not verbatim transcripts: they drop
  disfluencies ("就就就") and some overlapping speech. Absolute CER is
  inflated, but every model sees the same references, so the ranking holds.
- The action clip has little dialogue (45 cues), so its per-clip CER is noisy.
- `offset` measures the shared VAD segmentation, not the models. Subtitle
  timing per model needs token timestamps, which only the CTC models provide.
- 80 minutes of audio is a screening benchmark. Confirm on your own material
  before deploying.
