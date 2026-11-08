// sherpa-onnx/csrc/offline-stream-test.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/offline-stream.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

std::vector<float> Sine(int32_t n) {
  std::vector<float> samples(n);
  for (int32_t i = 0; i < n; ++i) {
    samples[i] = 0.1f * std::sin(0.05f * static_cast<float>(i));
  }
  return samples;
}

std::vector<float> Features(float dither) {
  FeatureExtractorConfig config;
  config.dither = dither;
  OfflineStream stream(config);
  const auto samples = Sine(16000);
  stream.AcceptWaveform(16000, samples.data(),
                        static_cast<int32_t>(samples.size()));
  return stream.GetFrames();
}

TEST(OfflineStream, WithoutDitherLeavesGlobalRandAlone) {
  std::srand(12345);
  const int expected = std::rand();

  std::srand(12345);
  Features(/*dither=*/0.0f);
  EXPECT_EQ(std::rand(), expected);
}

TEST(OfflineStream, DitheredFeaturesAreReproducible) {
  const auto a = Features(/*dither=*/1.0f);
  std::rand();  // Disturb the global rand() state between the two runs.
  const auto b = Features(/*dither=*/1.0f);
  EXPECT_EQ(a, b);
}

}  // namespace
}  // namespace sherpa_onnx
