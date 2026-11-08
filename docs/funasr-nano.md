# FunASR-Nano Model

FunASR-Nano is a lightweight speech recognition model optimized for edge deployment, integrated into sherpa-onnx for cross-platform inference.

## Model Overview

FunASR-Nano is based on the **Qwen3-0.6B** language model architecture, designed for:
- Low-latency speech recognition
- Edge device deployment
- Multi-language support (Chinese, English, Japanese, Korean, Cantonese, dialects)

## Model Files

The FunASR-Nano model package contains:

```
sherpa-onnx-funasr-nano-int8-2025-12-30/
├── encoder_adaptor.int8.onnx   # Audio encoder (238MB)
├── embedding.int8.onnx         # Embedding layer (155MB)
├── llm.int8.onnx               # Language model (600MB)
├── Qwen3-0.6B/                 # Tokenizer files
│   ├── tokenizer.json
│   ├── vocab.json
│   └── merges.txt
├── test_wavs/                  # Test audio files
│   ├── lyrics.wav
│   ├── ja.wav
│   ├── vietnamese.wav
│   └── ...
└── README.md
```

## Download

```bash
# From GitHub releases
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
```

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌─────────────┐
│ Audio Input │ ──► │ Encoder Adaptor  │ ──► │  Embedding  │ ──► │     LLM     │
│   (WAV)     │     │ (encoder_adaptor │     │ (embedding  │     │ (llm.int8   │
│             │     │  .int8.onnx)     │     │  .int8.onnx)│     │  .onnx)     │
└─────────────┘     └──────────────────┘     └─────────────┘     └──────────────┘
                                                                        │
                                                                        ▼
                                                              ┌─────────────────┐
                                                              │   Tokenizer     │
                                                              │ (Qwen3 BPE)     │
                                                              │                 │
                                                              │ Decode to Text  │
                                                              └─────────────────┘
                                                                        │
                                                                        ▼
                                                                  Transcription
```

## Configuration Parameters

### OfflineFunASRNanoModelConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_adaptor` | string | "" | Path to encoder_adaptor.onnx |
| `llm` | string | "" | Path to llm.onnx (KV cache model) |
| `embedding` | string | "" | Path to embedding.onnx |
| `tokenizer` | string | "" | Path to tokenizer directory (Qwen3-0.6B) |
| `system_prompt` | string | "You are a helpful assistant." | System prompt for the LLM |
| `user_prompt` | string | "语音转写：" | User prompt template |
| `max_new_tokens` | int32 | 512 | Maximum tokens to generate |
| `temperature` | float | 1e-6 | Sampling temperature (near 0 = greedy) |
| `top_p` | float | 0.8 | Top-p (nucleus) sampling threshold |
| `repetition_penalty` | float | 1.0 | Penalty for repeated tokens (1.0 = none) |
| `no_repeat_ngram_size` | int32 | 0 | Block repeated n-grams (0 = disabled) |
| `max_tokens_per_second` | float | 15 | Cap generated tokens at ~this rate × audio seconds + 16, so loops stop early; logs a warning when hit (0 = disabled). C++/Python/CLI only; the C API keeps its upstream struct layout and uses the defaults |
| `seed` | int32 | 42 | Random seed for reproducibility |

## Repetition Control

FunASR-Nano has two opt-in mechanisms against repetitive (looping) output.
Both are **disabled by default**.

> **Re-download the model package rather than turning these on.** Heavy
> looping came from a broken int8 LLM (`llm.int8.onnx`) in packages downloaded
> before 2026-04-12 (issue #3066). Upstream replaced it with a working int8
> LLM (`llm_int8_compat`, #3493); it is also available at
> <https://modelscope.cn/models/zengshuishui/FunASR-nano-onnx>.
> On the 80 minutes of `scripts/asr-eval` material (5 meeting + 5 film clips,
> 8 minutes each):
>
> | LLM | Penalties | Meeting CER | Film CER | Meeting RTF |
> |---|---|---|---|---|
> | old int8 | 1.2 + 3-gram | 66.6% (many dropped words) | 70.5% | 0.31 |
> | fp16 | off | 12.0% | 24.4% | 1.02 |
> | int8_compat | off | **10.5%** | **21.0%** | **0.44** |
>
> Chinese text naturally repeats common characters, so penalties also suppress
> correct tokens. Both good LLMs still loop occasionally (a 5 s segment
> becoming "完了完了…" or "<>>>>…"; int8_compat did so less often than fp16).
> `max_tokens_per_second` (on by default, 15) bounds every such loop to the
> segment's duration without touching normal output: on 811 Chinese VAD
> segments, real text needed at most 7.5 tokens/s.

### Repetition Penalty

The `repetition_penalty` parameter penalizes tokens that have already been generated:
- **Value = 1.0**: No penalty (disabled)
- **Value > 1.0**: Penalize repeated tokens (recommended: 1.1-1.5)
- **Effect**: For positive logits, divide by penalty; for negative logits, multiply by penalty

```cpp
// Example: Enable repetition penalty
config.model_config.funasr_nano.repetition_penalty = 1.2f;
```

### N-gram Blocking

The `no_repeat_ngram_size` parameter prevents exact phrase repetition:
- **Value = 0**: Disabled
- **Value = 2**: No bigram can repeat (two consecutive tokens)
- **Value = 3**: No trigram can repeat (three consecutive tokens)

```cpp
// Example: Block repeated trigrams
config.model_config.funasr_nano.no_repeat_ngram_size = 3;
```

### Algorithm Details

**N-gram Blocking Algorithm:**
```
For each candidate token:
  1. Build candidate n-gram: [last (n-1) tokens] + [candidate]
  2. Search history for exact match of this n-gram
  3. If found, set logit to -∞ (block this token)
```

**Repetition Penalty Algorithm:**
```
For each token in vocabulary:
  if token appears in generated history:
    if logit > 0: logit = logit / penalty
    else: logit = logit * penalty
```

## Streaming Transcription Support

FunASR-Nano supports streaming text output during transcription. While the audio must be processed in full (it's an LLM-based model), the generated text can be streamed token-by-token as it's produced.

### C++ Streaming Example

```cpp
#include "sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.h"

// Create recognizer with repetition control
sherpa_onnx::OfflineRecognizerConfig config;
config.model_config.funasr_nano.encoder_adaptor = "encoder_adaptor.int8.onnx";
config.model_config.funasr_nano.llm = "llm.int8.onnx";
config.model_config.funasr_nano.embedding = "embedding.int8.onnx";
config.model_config.funasr_nano.tokenizer = "Qwen3-0.6B";
// Optional, off by default; only if the LLM loops (see Repetition Control)
// config.model_config.funasr_nano.repetition_penalty = 1.2f;
// config.model_config.funasr_nano.no_repeat_ngram_size = 3;

OfflineRecognizerFunASRNanoImpl recognizer(config);

// Define streaming callback
auto streaming_callback = [](const std::string &text, int64_t token_id, bool is_final) -> bool {
    std::cout << text << std::flush;  // Print each token immediately
    if (is_final) {
        std::cout << std::endl;
    }
    return true;  // Return false to stop generation early
};

// Create stream and add audio
auto stream = recognizer.CreateStream();
stream->AcceptWaveform(sample_rate, samples.data(), samples.size());

// Decode with streaming callback
recognizer.DecodeStreamWithCallback(stream.get(), streaming_callback);

// Result is also stored in the stream
std::cout << "Final: " << stream->GetResult().text << std::endl;
```

### Callback Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `const std::string&` | Decoded text chunk from the current token |
| `token_id` | `int64_t` | The token ID that produced this text (-1 for flush) |
| `is_final` | `bool` | True if this is the last token/chunk |

**Return Value:** Return `true` to continue generation, `false` to stop early.

## Sampling Methods

### Token Sampling Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Logits from LLM                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              1. Apply Repetition Penalty                            │
│    For tokens in history: logit = logit / penalty (if > 0)         │
│                           logit = logit * penalty (if < 0)         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              2. Apply N-gram Blocking                               │
│    For tokens that would create repeated n-gram: logit = -∞        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              3. Temperature Scaling                                 │
│    logits = logits / temperature                                   │
│    (temperature ≤ 1e-6 → greedy decoding)                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              4. Top-p (Nucleus) Sampling                            │
│    Sort by probability, keep tokens until cumsum >= top_p          │
│    Renormalize and sample from this subset                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                           Selected Token
```

### SampleTokenWithPenalty Function

The core sampling function signature:

```cpp
int64_t SampleTokenWithPenalty(
    const void *logits,                    // Raw logits from LLM
    bool is_fp16,                          // FP16 or FP32 format
    int32_t vocab_size,                    // Vocabulary size
    float temperature,                     // Sampling temperature
    float top_p,                           // Nucleus sampling threshold
    float repetition_penalty,              // Repetition penalty (≥1.0)
    int32_t no_repeat_ngram_size,          // N-gram blocking size (0=off)
    const std::vector<int64_t> &generated_ids  // Previously generated tokens
) const;
```

## Usage Examples

### Dart API

```dart
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

final config = sherpa_onnx.OfflineFunAsrNanoModelConfig(
  encoderAdaptor: './sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx',
  llm: './sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx',
  embedding: './sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx',
  tokenizer: './sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B',
);

final modelConfig = sherpa_onnx.OfflineModelConfig(funAsrNano: config);
final recognizer = sherpa_onnx.OfflineRecognizer(
  sherpa_onnx.OfflineRecognizerConfig(model: modelConfig),
);

// Decode audio
final result = recognizer.decode(audioSamples, sampleRate);
print(result.text);
```

### Python API

```python
import sherpa_onnx

config = sherpa_onnx.OfflineRecognizerConfig(
    model_config=sherpa_onnx.OfflineModelConfig(
        funasr_nano=sherpa_onnx.OfflineFunAsrNanoModelConfig(
            encoder_adaptor="./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx",
            llm="./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx",
            embedding="./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx",
            tokenizer="./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B",
        )
    )
)

recognizer = sherpa_onnx.OfflineRecognizer(config)
stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, samples)
recognizer.decode(stream)
print(stream.result.text)
```

### Command Line (using example scripts)

```bash
# Dart example
cd dart-api-examples/non-streaming-asr
./run-funasr-nano.sh

# Java example
cd java-api-examples
./run-non-streaming-decode-file-funasr-nano.sh

# Go example
cd go-api-examples/non-streaming-funasr-nano-decode-files
./run.sh
```

## Supported Languages

| Language | Quality | Test Audio | Notes |
|----------|---------|------------|-------|
| Chinese (Mandarin) | Excellent | lyrics.wav | Best performance |
| English | Good | lyrics_en_*.wav | |
| Japanese | Good | ja.wav | |
| Korean | Good | - | |
| Cantonese (粤语) | Good | dia_yue.wav | Use repetition penalty |
| Shanghainese (上海话) | Good | dia_sh.wav | Use repetition penalty |
| Minnan (闽南语) | Good | dia_minnan.wav | Use repetition penalty |
| Hunan dialect (湖南话) | Good | dia_hunan.wav | Use repetition penalty |
| Vietnamese | Experimental | vietnamese.wav | |

**Note**: If dialect audio loops, first make sure the package has the fixed
int8 LLM (downloaded on or after 2026-04-12, see Repetition Control). Use
`repetition_penalty=1.2` and `no_repeat_ngram_size=3` only as a fallback.

## Test Audio Files

The model package includes test audio files in `test_wavs/`:

| File | Description |
|------|-------------|
| `lyrics.wav` | Chinese song lyrics |
| `lyrics_en_*.wav` | English song lyrics |
| `ja.wav` | Japanese speech |
| `ja_en_codeswitch.wav` | Japanese-English code-switching |
| `vietnamese.wav` | Vietnamese speech |
| `dia_*.wav` | Chinese dialect samples |
| `rag_*.wav` | Domain-specific (medical, chemistry, physics) |
| `far_*.wav` | Far-field audio samples |
| `noise_en.wav` | Noisy English speech |

## Performance Characteristics

- **Model Size**: ~1GB total (INT8 quantized)
- **Inference**: Non-streaming (offline)
- **Latency**: Depends on audio length and hardware
- **Memory**: Moderate (suitable for mobile devices)

## Integration with VAD

For long audio files, combine with Voice Activity Detection:

```python
# Using silero-vad with FunASR-Nano
vad_config = sherpa_onnx.VadModelConfig(
    silero_vad=sherpa_onnx.SileroVadModelConfig(model="silero_vad.onnx")
)

# Process VAD segments with FunASR-Nano recognizer
```

See `dotnet-examples/vad-non-streaming-funasr-nano/` for complete examples.

## Related Components

- **Tokenizer**: [FunASR-Nano Tokenizer Documentation](funasr-nano-tokenizer.md)
- **Model Source**: [FunASR Project](https://github.com/modelscope/FunASR)
- **Base Model**: Qwen3-0.6B

## Troubleshooting

### Common Issues

1. **"Cannot find tokenizer.json"**
   - Ensure tokenizer path points to the `Qwen3-0.6B` directory, not the model root

2. **UnicodeDecodeError**
   - Fixed in v1.12.23 (see PR #3058)
   - Update to latest sherpa-onnx version

3. **Out of memory on mobile**
   - Use INT8 quantized models (default)
   - Consider using smaller models for constrained devices

4. **Repetitive output (e.g., "那那那那...")**
   - Packages downloaded before 2026-04-12 ship a broken int8 LLM that loops
     heavily: re-download the package, or replace `llm.int8.onnx` with
     `llm_int8_compat` (see Repetition Control)
   - Occasional short loops remain with any LLM precision;
     `max_tokens_per_second` (default 15) keeps each one bounded
   - As a last resort, `repetition_penalty = 1.2` and
     `no_repeat_ngram_size = 3` stop the loops but drop correct words

5. **Slow inference**
   - Reduce `max_new_tokens` if you expect shorter transcriptions
   - Use INT8 models (default) instead of FP32

## Version History

- **2025-12-30**: Initial INT8 quantized release
- **v1.12.23**: Fixed UnicodeDecodeError in tokenizer (PR #3058)
- **v1.12.22**: Added Dart API support (PR #3055)
- **v1.12.24**: Added repetition penalty and n-gram blocking for dialect support
- **v2.0.1 (fork)**: Repetition penalty and n-gram blocking are now off by
  default (they reduced accuracy on every LLM tested). Both are exposed in
  Python. `max_tokens_per_second` (default 15) bounds output length.
