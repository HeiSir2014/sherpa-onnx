// sherpa-onnx/csrc/offline-tts-qwen3-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_IMPL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/funasr-nano-tokenizer.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-qwen3-model.h"

namespace sherpa_onnx {

class OfflineTtsQwen3Impl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsQwen3Impl(const OfflineTtsConfig &config);

#if __ANDROID_API__ >= 9
  OfflineTtsQwen3Impl(AAssetManager *mgr, const OfflineTtsConfig &config);
#endif

#if __OHOS__
  OfflineTtsQwen3Impl(NativeResourceManager *mgr,
                      const OfflineTtsConfig &config);
#endif

  int32_t SampleRate() const override { return 24000; }

  int32_t NumSpeakers() const override { return 1; }

  GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override;

  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &gen_config,
      GeneratedAudioCallback callback = nullptr) const override;

 private:
  Ort::Value RunTextProjectHelper(const std::vector<int64_t> &ids) const;
  Ort::Value RunCodecEmbedHelper(const std::vector<int64_t> &ids) const;

  // Decode a batch of codec frames to audio samples.
  // codes: each element is one frame of num_code_groups token IDs.
  // use_stream: if true, use the streaming decoder session (small-N trace);
  //             falls back to batch decoder when stream model is not loaded.
  // Returns valid audio samples (length determined by the decoder's lengths
  // output).
  std::vector<float> DecodeFrames(
      const std::vector<std::vector<int64_t>> &codes,
      bool use_stream = false) const;

  // Sample a token from logits with temperature, top-k, top-p, rep penalty.
  // Suppress tokens in [suppress_start, suppress_end) except
  // suppress_exception. If suppress_eos is true, also suppress the eos token
  // (for min_new_tokens).
  int64_t SampleFromLogits(const Ort::Value &logits_tensor, int32_t vocab_size,
                           float temperature, int32_t top_k, float top_p,
                           float repetition_penalty,
                           const std::vector<int64_t> &generated_ids,
                           int32_t suppress_start = -1,
                           int32_t suppress_end = -1,
                           int64_t suppress_exception = -1,
                           bool suppress_eos = false) const;

  OfflineTtsConfig config_;
  OfflineTtsQwen3Model model_;
  mutable FunASRNanoTokenizer tokenizer_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_IMPL_H_
