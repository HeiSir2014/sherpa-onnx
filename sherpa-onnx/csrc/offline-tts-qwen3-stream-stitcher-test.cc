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

TEST(Qwen3DecodeWindowed, MatchesSingleDecodeExactly) {
  auto decode = [](int32_t begin, int32_t end) {
    return Decode(begin, end - begin);
  };
  const auto actual = Qwen3DecodeWindowed(
      57, /*window_frames=*/5, /*context_frames=*/3,
      /*crossfade_samples=*/2 * kSamplesPerFrame, kSamplesPerFrame, decode);
  const auto expected = Decode(0, 57);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    // Blending equal values can be off by one ulp.
    EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "sample " << i;
  }
}

TEST(Qwen3DecodeWindowed, NeverExceedsWindowPlusContext) {
  int32_t max_frames = 0;
  int32_t covered_until = 0;
  auto decode = [&](int32_t begin, int32_t end) {
    max_frames = std::max(max_frames, end - begin);
    EXPECT_LE(begin, covered_until);  // no gap between windows
    covered_until = end;
    return Decode(begin, end - begin);
  };
  Qwen3DecodeWindowed(100, /*window_frames=*/25, /*context_frames=*/49,
                      /*crossfade_samples=*/0, kSamplesPerFrame, decode);
  EXPECT_EQ(max_frames, 74);
  EXPECT_EQ(covered_until, 100);
}

TEST(Qwen3DecodeWindowed, EmptyInputDecodesNothing) {
  int32_t calls = 0;
  auto decode = [&](int32_t, int32_t) {
    ++calls;
    return std::vector<float>{};
  };
  EXPECT_TRUE(Qwen3DecodeWindowed(0, 25, 49, 8, kSamplesPerFrame, decode)
                  .empty());
  EXPECT_EQ(calls, 0);
}

TEST(Qwen3StreamWindow, FitsDecoderCapacity) {
  // Defaults (chunk 25, context 50) overflow a 74-frame decoder: shrink the
  // context, keep the chunk.
  EXPECT_EQ(Qwen3FitStreamWindow(25, 50, 74), (Qwen3StreamWindow{25, 49}));
  // Already fits: unchanged.
  EXPECT_EQ(Qwen3FitStreamWindow(25, 10, 74), (Qwen3StreamWindow{25, 10}));
  // Chunk alone too large: clamp it and drop the context.
  EXPECT_EQ(Qwen3FitStreamWindow(100, 50, 74), (Qwen3StreamWindow{74, 0}));
  // Unknown capacity (0): unchanged.
  EXPECT_EQ(Qwen3FitStreamWindow(25, 50, 0), (Qwen3StreamWindow{25, 50}));
}

}  // namespace
}  // namespace sherpa_onnx
