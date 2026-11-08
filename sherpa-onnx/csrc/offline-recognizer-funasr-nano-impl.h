// sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.h
//
// Copyright (c)  2025  zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FUNASR_NANO_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FUNASR_NANO_IMPL_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/funasr-nano-tokenizer.h"
#include "sherpa-onnx/csrc/offline-funasr-nano-model-config.h"
#include "sherpa-onnx/csrc/offline-funasr-nano-model.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/pad-sequence.h"

namespace sherpa_onnx {

// Returns true when the (LFR-stacked) fbank frames are all identical, which
// for this model family means the clip is digital silence: dither is disabled
// and the fbank is deterministic, so constant samples produce constant frames
// while any real signal varies from frame to frame.
//
// Exposed here (rather than kept file-local) so it can be unit tested.
bool FunASRNanoAudioIsSilent(const float *features, int32_t n);

// Build the prompt with per-stream hotwords, language, itn, and user_prompt
// options. Missing options use config defaults; empty string options override
// them. For itn, 0 disables, nonzero integers enable, and invalid/empty values
// use the config default. Neither the stream nor the config is modified.
std::string FunASRNanoBuildUserPrompt(
    const OfflineFunASRNanoModelConfig &config, const OfflineStream &stream);

// Token budget for one utterance: the configured max_new_tokens, capped at
// tokens_per_second * audio_seconds plus a small margin. Real speech needs
// fewer tokens than a generous rate, so the cap bounds a degenerate loop
// ("star star star ...") instead of letting it run to max_new_tokens.
// A non-positive duration or rate disables the cap.
int32_t FunASRNanoMaxNewTokens(int32_t configured, float audio_seconds,
                               float tokens_per_second);

class OfflineRecognizerFunASRNanoImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerFunASRNanoImpl(
      const OfflineRecognizerConfig &config);

  template <typename Manager>
  OfflineRecognizerFunASRNanoImpl(Manager *mgr,
                                  const OfflineRecognizerConfig &config);

  std::unique_ptr<OfflineStream> CreateStream() const override;

  void DecodeStreams(OfflineStream **ss, int32_t n) const override;

  // Decode streams with streaming callback for real-time text output
  // The callback is called for each generated token with the decoded text.
  // The streamed pieces are raw model output: they are not post-processed
  // (e.g. normalized) and their concatenation can differ from the final
  // text. Always use OfflineStream::GetResult().text, set when this returns,
  // as the authoritative result.
  void DecodeStreamsWithCallback(
      OfflineStream **ss, int32_t n,
      const FunASRNanoStreamingCallback &callback) const;

  // Decode a single stream with streaming callback
  void DecodeStreamWithCallback(
      OfflineStream *s, const FunASRNanoStreamingCallback &callback) const;

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void InitFeatConfig();
  std::vector<float> ApplyLFR(const std::vector<float> &in) const;

  std::vector<int64_t> BuildSourceIds(const std::string &system_prompt,
                                      const std::string &user_prompt,
                                      int32_t audio_token_len,
                                      int32_t &fbank_beg_idx,
                                      int32_t &fake_token_len) const;

  int64_t SampleTokenFromLogitsFp16OrFp32(const void *logits,
                                         bool is_fp16,
                                         int32_t vocab_size) const;

  int64_t SampleTokenWithTemperatureAndTopP(const void *logits,
                                            bool is_fp16,
                                            int32_t vocab_size,
                                            float temperature,
                                            float top_p) const;

  // Sample with repetition penalty and n-gram blocking
  int64_t SampleTokenWithPenalty(
      const void *logits, bool is_fp16, int32_t vocab_size, float temperature,
      float top_p, float repetition_penalty, int32_t no_repeat_ngram_size,
      const std::vector<int64_t> &generated_ids) const;

  // Check if adding token_id would create a repeated n-gram
  bool WouldCreateRepeatedNgram(
      int64_t token_id, int32_t ngram_size,
      const std::vector<int64_t> &generated_ids) const;

  // If `callback` is set, it is invoked with the decoded text of each
  // generated token, then once more with is_final=true when generation ends.
  OfflineRecognitionResult GenerateText(
      Ort::Value encoder_out, const std::string &system_prompt,
      const std::string &user_prompt,
      const FunASRNanoStreamingCallback &callback = nullptr) const;

  OfflineRecognizerConfig config_;
  std::unique_ptr<OfflineFunASRNanoModel> model_;
  std::unique_ptr<FunASRNanoTokenizer> tokenizer_;
  mutable std::mt19937 rng_;
  // Protects rng_, which is shared and drawn from by
  // SampleTokenWithTemperatureAndTopP(). DecodeStreams() may be called
  // concurrently from multiple threads on the same recognizer instance (see
  // sherpa-onnx-offline-parallel.cc), so draws from rng_ must be serialized to
  // avoid a data race.
  mutable std::mutex rng_mutex_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FUNASR_NANO_IMPL_H_

