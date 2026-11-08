// sherpa-onnx/csrc/offline-tts-qwen3-stream-stitcher.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/offline-tts-qwen3-stream-stitcher.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace sherpa_onnx {

Qwen3StreamStitcher::Qwen3StreamStitcher(int32_t crossfade_samples)
    : crossfade_samples_(std::max(crossfade_samples, 0)) {}

std::vector<float> Qwen3StreamStitcher::Push(const std::vector<float> &decoded,
                                             int32_t context_samples) {
  context_samples = std::clamp(context_samples, 0,
                               static_cast<int32_t>(decoded.size()));
  const auto new_begin = decoded.begin() + context_samples;

  std::vector<float> out;
  out.reserve(held_.size() + (decoded.end() - new_begin));

  const int32_t held = static_cast<int32_t>(held_.size());
  if (held > 0 && context_samples >= held) {
    // The held samples and the last `held` samples of this chunk's context
    // cover the same time span: crossfade them (overlap-add).
    const auto shared = new_begin - held;
    for (int32_t i = 0; i < held; ++i) {
      const float t = static_cast<float>(i + 1) / (held + 1);
      out.push_back(held_[i] * (1.0f - t) + shared[i] * t);
    }
  } else {
    // No shared span to blend (e.g. no left context): keep the held audio.
    out.insert(out.end(), held_.begin(), held_.end());
  }
  out.insert(out.end(), new_begin, decoded.end());

  // Hold back the tail so the next chunk can blend into it.
  const int32_t keep = std::min(crossfade_samples_,
                                static_cast<int32_t>(out.size()));
  held_.assign(out.end() - keep, out.end());
  out.resize(out.size() - keep);
  return out;
}

std::vector<float> Qwen3StreamStitcher::Flush() {
  std::vector<float> out = std::move(held_);
  held_.clear();
  return out;
}

}  // namespace sherpa_onnx
