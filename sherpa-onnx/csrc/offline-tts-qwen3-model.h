// sherpa-onnx/csrc/offline-tts-qwen3-model.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

// KV-cache state for the main talker autoregressive loop
struct Qwen3TalkerState {
  std::vector<Ort::Value> kv_cache;
};

// Configuration values read from config.json
struct Qwen3TtsConfig {
  int32_t num_code_groups = 16;
  int32_t hidden_size = 1024;
  int32_t talker_vocab_size = 3072;
  int32_t code_predictor_vocab_size = 2048;

  int64_t tts_bos_token_id = 151672;
  int64_t tts_eos_token_id = 151673;
  int64_t tts_pad_token_id = 151671;

  int64_t codec_bos_id = 2149;
  int64_t codec_eos_token_id = 2150;
  int64_t codec_pad_id = 2148;
  int64_t codec_nothink_id = 2155;
  int64_t codec_think_id = 2154;
  int64_t codec_think_bos_id = 2156;
  int64_t codec_think_eos_id = 2157;
};

// Updates `config` from the text of a config.json. Talker fields are read
// from the "talker_config" object (Hugging Face format) or, if it is absent,
// from the top level (format written by scripts/qwen3-tts/export-onnx.py).
// Fields that are missing keep their current values. Returns false, leaving
// `config` unchanged, if the text is not valid JSON.
bool ParseQwen3TtsConfigJson(const std::string &text, Qwen3TtsConfig *config);

class OfflineTtsQwen3Model {
 public:
  explicit OfflineTtsQwen3Model(const OfflineTtsModelConfig &config);

#if __ANDROID_API__ >= 9
  OfflineTtsQwen3Model(AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
  OfflineTtsQwen3Model(NativeResourceManager *mgr,
                       const OfflineTtsModelConfig &config);
#endif

  ~OfflineTtsQwen3Model();

  // Text processing: token IDs -> text embeddings (via text_embed + projection)
  // Input:  input_ids [1, T] int64
  // Output: text_embed [1, T, D] float32
  Ort::Value RunTextProject(Ort::Value input_ids) const;

  // Codec token embedding (primary codebook, vocab_size=3072)
  // Input:  input_ids [1, T] int64
  // Output: embed [1, T, D] float32
  Ort::Value RunCodecEmbed(Ort::Value input_ids) const;

  // Code predictor embedding (residual codebooks, vocab_size=2048 each)
  // Input:  input_ids [1, 1] int64, generation_step [1] int64
  // Output: embed [1, 1, D] float32
  Ort::Value RunCodePredictorEmbed(Ort::Value input_ids,
                                   Ort::Value generation_step) const;

  // Code predictor: predict residual codebook token
  // No KV cache: re-runs full attention each step (max 17 tokens)
  // Input:  inputs_embeds [1, T, D] float32, generation_step [1] int64
  // Output: logits [1, V_cp] float32  (V_cp = 2048)
  Ort::Value RunCodePredictor(Ort::Value inputs_embeds,
                              Ort::Value generation_step) const;

  // Talker prefill: process full context, produce KV-cache
  // Input:  inputs_embeds [1, T, D] float32, attention_mask [1, T] int64
  // Output: logits [1, T, V] float32, last_hidden [1, T, D], KV-cache
  struct TalkerPrefillResult {
    Ort::Value logits;
    Ort::Value last_hidden;
    Qwen3TalkerState state;
  };
  TalkerPrefillResult RunTalkerPrefill(Ort::Value inputs_embeds,
                                       Ort::Value attention_mask) const;

  // Talker decode: generate one token with KV-cache
  // Input:  inputs_embeds [1, 1, D], attention_mask [1, total_len], KV-cache
  // Output: logits [1, 1, V], last_hidden [1, 1, D], updated KV-cache
  struct TalkerDecodeResult {
    Ort::Value logits;
    Ort::Value last_hidden;
    Qwen3TalkerState state;
  };
  TalkerDecodeResult RunTalkerDecode(Ort::Value inputs_embeds,
                                     Ort::Value attention_mask,
                                     Qwen3TalkerState state) const;

  // Speaker encoder: mel spectrogram -> speaker embedding
  // Input:  mels [1, 128, T] float32
  // Output: speaker_embedding [1, D] float32
  Ort::Value RunSpeakerEncoder(Ort::Value mels) const;

  // Tokenizer 12Hz decoder: codec tokens -> audio waveform
  // Input:  audio_codes [1, T, num_codebooks] int64
  // Output: audio_values [1, max_samples] float32, lengths [1] int64
  struct Tokenizer12hzDecodeResult {
    Ort::Value audio_values;
    Ort::Value lengths;
  };
  // Batch decoder: traces with large T, suited for full-sequence decode.
  Tokenizer12hzDecodeResult RunTokenizer12hzDecode(
      Ort::Value audio_codes) const;
  // Streaming decoder: uses the small-N stream model when available,
  // otherwise falls back to the batch decoder.
  Tokenizer12hzDecodeResult RunTokenizer12hzDecodeStream(
      Ort::Value audio_codes) const;

  // Returns true if the streaming decoder model is loaded.
  bool HasTokenizer12hzDecodeStream() const;

  // Access model configuration
  const Qwen3TtsConfig &GetConfig() const;

  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_MODEL_H_
