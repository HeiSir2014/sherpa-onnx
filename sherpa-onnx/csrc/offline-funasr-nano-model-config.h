// sherpa-onnx/csrc/offline-funasr-nano-model-config.h
//
// Copyright (c)  2025  zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_CONFIG_H_

#include <functional>
#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// Callback for streaming token generation
// Parameters:
//   - text: Decoded text chunk from this token
//   - token_id: The token ID that was generated, or -1 for the final call
//   - is_final: True exactly once, after generation ends. `text` then holds
//     any remaining incomplete UTF-8 bytes and is usually empty.
// Returns: true to continue generation, false to stop early
using FunASRNanoStreamingCallback = std::function<bool(
    const std::string &text, int64_t token_id, bool is_final)>;

struct OfflineFunASRNanoModelConfig {
  // Path to encoder_adaptor.onnx
  std::string encoder_adaptor;

  // Path to llm.onnx (KV cache model)
  std::string llm;

  // Path to embedding.onnx
  std::string embedding;

  // Path to tokenizer directory (e.g., Qwen3-0.6B)
  std::string tokenizer;

  // System prompt (must match FunASR-nano training format)
  std::string system_prompt = "You are a helpful assistant.";

  // User prompt template (must match FunASR-nano training format)
  std::string user_prompt = "语音转写：";

  // Maximum number of new tokens to generate
  int32_t max_new_tokens = 512;

  // Sampling temperature
  float temperature = 1e-6f;

  // Top-p (nucleus) sampling threshold
  float top_p = 0.8f;

  // Repetition penalty (1.0 = no penalty, >1.0 = penalize repeated tokens).
  // Off by default: Chinese text naturally repeats common characters, and a
  // penalty hurts accuracy on a healthy (fp16/fp32) LLM. Only enable it to
  // stop looping output, e.g. with a degraded int8 LLM.
  float repetition_penalty = 1.0f;

  // N-gram blocking size (0 = disabled, >0 = block repeated n-grams).
  // Off by default for the same reason as repetition_penalty.
  int32_t no_repeat_ngram_size = 0;

  // Random seed for reproducibility
  int32_t seed = 42;

  // Language for transcription (empty string means None)
  std::string language;

  // Whether to apply inverse text normalization (ITN)
  bool itn = true;

  // Hotwords
  std::string hotwords;

  OfflineFunASRNanoModelConfig() = default;

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_CONFIG_H_

