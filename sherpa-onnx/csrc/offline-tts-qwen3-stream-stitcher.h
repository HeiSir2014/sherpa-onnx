// sherpa-onnx/csrc/offline-tts-qwen3-stream-stitcher.h
//
// Copyright (c)  2026  sherpa-onnx contributors

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_STREAM_STITCHER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_STREAM_STITCHER_H_

#include <cstdint>
#include <functional>
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

// Chunk and left-context sizes (in codec frames) for one decoder call.
struct Qwen3StreamWindow {
  int32_t chunk_frames;
  int32_t context_frames;

  bool operator==(const Qwen3StreamWindow &o) const {
    return chunk_frames == o.chunk_frames &&
           context_frames == o.context_frames;
  }
};

// Shrinks a window so that chunk + context fits a decoder that can produce
// at most `max_frames` frames per call; frames beyond that are cut off by
// the decoder. The context is reduced first; the chunk only if it alone is
// too large. max_frames <= 0 means unknown capacity: nothing is changed.
Qwen3StreamWindow Qwen3FitStreamWindow(int32_t chunk_frames,
                                       int32_t context_frames,
                                       int32_t max_frames);

// Decodes frames [0, num_frames) in windows of `window_frames` new frames,
// each with up to `context_frames` frames of left context, and stitches the
// results with Qwen3StreamStitcher. `decode(begin, end)` must return the
// audio for frames [begin, end). Gives the same result as the streaming
// path with the same window, so batch and streaming output agree.
std::vector<float> Qwen3DecodeWindowed(
    int32_t num_frames, int32_t window_frames, int32_t context_frames,
    int32_t crossfade_samples, int32_t samples_per_frame,
    const std::function<std::vector<float>(int32_t, int32_t)> &decode);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_STREAM_STITCHER_H_
