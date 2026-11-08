// sherpa-onnx/csrc/offline-tts-qwen3-model-config.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// Configuration for Qwen3-TTS-12Hz models.
// The model is split into 9 ONNX sub-models for inference.
struct OfflineTtsQwen3ModelConfig {
  // Path to the text_project ONNX model (text token IDs -> text embeddings)
  std::string text_project;

  // Path to the codec_embed ONNX model (codec token -> embedding)
  std::string codec_embed;

  // Path to the code_predictor_embed ONNX model (residual codebook embeddings)
  std::string code_predictor_embed;

  // Path to the code_predictor ONNX model (sub-talker for residual codebooks)
  std::string code_predictor;

  // Path to the talker_prefill ONNX model (prefill phase of autoregressive LM)
  std::string talker_prefill;

  // Path to the talker_decode ONNX model (decode phase, one token at a time)
  std::string talker_decode;

  // Path to the speaker_encoder ONNX model (reference audio -> speaker embed)
  std::string speaker_encoder;

  // Path to the tokenizer12hz_encode ONNX model (audio -> codec tokens)
  std::string tokenizer12hz_encode;

  // Path to the tokenizer12hz_decode ONNX model (codec tokens -> audio, batch)
  std::string tokenizer12hz_decode;

  // Path to the streaming-optimized tokenizer12hz_decode model.
  // Exported with a small fixed max_codes_length (e.g. 25 frames = 2 s).
  // When set, used for per-chunk decode in streaming mode so that each
  // decode call costs ~chunk_frames/1024 × batch_decode_time instead of
  // the full batch cost, enabling real-time first-chunk delivery.
  // Optional: if empty, batch decoder is used for all modes.
  std::string tokenizer12hz_decode_stream;

  // Directory containing tokenizer files (vocab.json, merges.txt, etc.)
  std::string tokenizer_dir;

  OfflineTtsQwen3ModelConfig() = default;

  OfflineTtsQwen3ModelConfig(
      const std::string &text_project, const std::string &codec_embed,
      const std::string &code_predictor_embed,
      const std::string &code_predictor, const std::string &talker_prefill,
      const std::string &talker_decode, const std::string &speaker_encoder,
      const std::string &tokenizer12hz_encode,
      const std::string &tokenizer12hz_decode, const std::string &tokenizer_dir)
      : text_project(text_project),
        codec_embed(codec_embed),
        code_predictor_embed(code_predictor_embed),
        code_predictor(code_predictor),
        talker_prefill(talker_prefill),
        talker_decode(talker_decode),
        speaker_encoder(speaker_encoder),
        tokenizer12hz_encode(tokenizer12hz_encode),
        tokenizer12hz_decode(tokenizer12hz_decode),
        tokenizer_dir(tokenizer_dir) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_MODEL_CONFIG_H_
