// sherpa-onnx/csrc/offline-tts-qwen3-stream-stitcher-test.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/offline-tts-qwen3-stream-stitcher.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

constexpr int32_t kSamplesPerFrame = 4;

// Fake decoder: every sample of frame f has the value f. With a perfect
// (context-aware) decoder, overlapping regions of two chunks are identical,
// so correct stitching must reproduce the batch decode exactly.
std::vector<float> Decode(int32_t first_frame, int32_t num_frames) {
  std::vector<float> out;
  for (int32_t f = first_frame; f < first_frame + num_frames; ++f) {
    out.insert(out.end(), kSamplesPerFrame, static_cast<float>(f));
  }
  return out;
}

// Streams `total` frames in chunks of `chunk` frames, each decoded with up to
// `context` frames of left context, then flushes.
std::vector<float> Stream(int32_t total, int32_t chunk, int32_t context,
                          int32_t overlap_frames) {
  Qwen3StreamStitcher stitcher(overlap_frames * kSamplesPerFrame);
  std::vector<float> out;
  for (int32_t start = 0; start < total; start += chunk) {
    const int32_t n = std::min(chunk, total - start);
    const int32_t ctx = std::min(context, start);
    auto piece = stitcher.Push(Decode(start - ctx, ctx + n),
                               ctx * kSamplesPerFrame);
    out.insert(out.end(), piece.begin(), piece.end());
  }
  auto tail = stitcher.Flush();
  out.insert(out.end(), tail.begin(), tail.end());
  return out;
}

TEST(Qwen3StreamStitcher, WithContextMatchesBatchExactly) {
  // 10 frames in chunks of 3 (last chunk partial), 2 frames context,
  // 2 frames overlap.
  EXPECT_EQ(Stream(10, 3, 2, 2), Decode(0, 10));
}

TEST(Qwen3StreamStitcher, NeverDropsOrDuplicatesSamples) {
  for (int32_t chunk : {1, 2, 5, 25}) {
    for (int32_t context : {0, 1, 2, 50}) {
      EXPECT_EQ(Stream(57, chunk, context, 2).size(),
                Decode(0, 57).size())
          << "chunk=" << chunk << " context=" << context;
    }
  }
}

TEST(Qwen3StreamStitcher, WithoutContextKeepsAudioUnblended) {
  // With no left context there is no shared time span to crossfade, so the
  // output must be the plain concatenation.
  EXPECT_EQ(Stream(10, 3, 0, 2), Decode(0, 10));
}

TEST(Qwen3StreamStitcher, CrossfadesTheSharedSpan) {
  Qwen3StreamStitcher stitcher(/*crossfade_samples=*/2);
  // Chunk 1: samples [0 0 0 0 1 1], nothing held yet; holds the last 2.
  auto a = stitcher.Push({0, 0, 0, 0, 1, 1}, /*context_samples=*/0);
  EXPECT_EQ(a, (std::vector<float>{0, 0, 0, 0}));
  // Chunk 2 re-decodes that same span as context with different values
  // (3 3), followed by new audio (5 5 5). The held [1 1] and context [3 3]
  // describe the same time span and are blended with weights 1/3, 2/3.
  auto b = stitcher.Push({3, 3, 5, 5, 5}, /*context_samples=*/2);
  ASSERT_EQ(b.size(), 3u);
  EXPECT_FLOAT_EQ(b[0], 1.0f * 2 / 3 + 3.0f * 1 / 3);
  EXPECT_FLOAT_EQ(b[1], 1.0f * 1 / 3 + 3.0f * 2 / 3);
  EXPECT_FLOAT_EQ(b[2], 5.0f);
  EXPECT_EQ(stitcher.Flush(), (std::vector<float>{5, 5}));
}

TEST(Qwen3StreamStitcher, FlushOnEmptyIsEmpty) {
  Qwen3StreamStitcher stitcher(8);
  EXPECT_TRUE(stitcher.Flush().empty());
}

}  // namespace
}  // namespace sherpa_onnx
