// sherpa-onnx/csrc/offline-tts-qwen3-stream-stitcher.h
//
// Copyright (c)  2026  sherpa-onnx contributors

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_STREAM_STITCHER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_STREAM_STITCHER_H_

#include <cstdint>
#include <vector>

namespace sherpa_onnx {

// Joins audio from chunk-wise decoding into one continuous stream.
//
// Each chunk is decoded with some left context: the decoder output is
// [context audio | new audio], where the context re-decodes the end of the
// previous chunk. The stitcher holds back the last `crossfade_samples` of
// what it has produced. When the next chunk arrives, those held samples and
// the last `crossfade_samples` of the new chunk's context describe the same
// time span, so they are crossfaded (overlap-add). The result has exactly
// one sample per decoded sample of new audio: nothing is dropped or
// duplicated.
//
// If a chunk has less context than crossfade_samples (e.g. context disabled),
// there is no shared span to blend; the held samples are emitted unchanged.
class Qwen3StreamStitcher {
 public:
  explicit Qwen3StreamStitcher(int32_t crossfade_samples);

  // `decoded` is the decoder output for [context frames | new frames];
  // the first `context_samples` samples belong to the context.
  // Returns the samples that are final and can be delivered now.
  std::vector<float> Push(const std::vector<float> &decoded,
                          int32_t context_samples);

  // Returns the samples still held back. Call once after the last Push().
  std::vector<float> Flush();

 private:
  int32_t crossfade_samples_;
  std::vector<float> held_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_STREAM_STITCHER_H_
