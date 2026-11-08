// cxx-api-examples/qwen3-tts-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

// This file shows how to use the sherpa-onnx C++ API for TTS with Qwen3-TTS.
//
// Qwen3-TTS is a fork-only model and is not exposed through the C API (and
// therefore not through the cxx-api wrapper): adding its config to
// SherpaOnnxOfflineTtsModelConfig would shift the fields that follow it in
// SherpaOnnxOfflineTtsConfig for every binding that mirrors the C layout.
// This example uses the core C++ classes directly instead.
//
// clang-format off
/*
Usage

Please download the Qwen3-TTS ONNX models first:

  https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

./qwen3-tts-cxx-api

 */
// clang-format on

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/csrc/wave-writer.h"

static void Save(const std::string &filename, const std::vector<float> &samples,
                 int32_t sample_rate) {
  sherpa_onnx::WriteWave(filename, sample_rate, samples.data(),
                         static_cast<int32_t>(samples.size()));
  fprintf(stderr, "Saved to %s  (%zu samples)\n", filename.c_str(),
          samples.size());
}

int32_t main(int32_t argc, char *argv[]) {
  sherpa_onnx::OfflineTtsConfig config;
  std::string model_dir = "./qwen3-tts-0.6b-int8";

  auto &qwen3 = config.model.qwen3;
  qwen3.text_project = model_dir + "/text_project_q.onnx";
  qwen3.codec_embed = model_dir + "/codec_embed_q.onnx";
  qwen3.code_predictor_embed = model_dir + "/code_predictor_embed_q.onnx";
  qwen3.code_predictor = model_dir + "/code_predictor_q.onnx";
  qwen3.talker_prefill = model_dir + "/talker_prefill_q.onnx";
  qwen3.talker_decode = model_dir + "/talker_decode_q.onnx";
  qwen3.tokenizer12hz_decode = model_dir + "/tokenizer12hz_decode_q.onnx";
  // Optional: streaming-optimized decoder exported with a small trace_T
  // (e.g. python3 export-onnx.py --streaming-chunk-frames=25).
  // When set, streaming mode decodes each chunk as it is generated
  // (true real-time first-chunk delivery).  Without it, streaming mode
  // fires progress-only callbacks during AR generation and delivers all
  // audio at once after a single batch decode (same quality as batch).
  qwen3.tokenizer12hz_decode_stream =
      model_dir + "/tokenizer12hz_decode_stream.onnx";
  qwen3.tokenizer_dir = model_dir + "/tokenizer";
  config.model.num_threads = 4;
  config.model.provider = "cpu";
  config.model.debug = false;

  if (!config.Validate()) {
    fprintf(stderr, "Invalid config. Please check the model paths.\n");
    return -1;
  }

  std::string text = "今天天气真好，阳光明媚，适合出去走走。";
  sherpa_onnx::OfflineTts tts(config);

  // ------------------------------------------------------------------
  // Test 1: Batch mode  (chunk_frames not set -> decode all at once)
  // ------------------------------------------------------------------
  fprintf(stderr, "\n=== Batch mode ===\n");
  {
    sherpa_onnx::GenerationConfig cfg;
    cfg.extra["max_new_tokens"] = "2048";
    auto audio = tts.Generate(text, cfg);
    Save("./qwen3-batch.wav", audio.samples, audio.sample_rate);
  }

  // ------------------------------------------------------------------
  // Test 2: Streaming mode  (chunk_frames=25 -> ~2s per chunk)
  // ------------------------------------------------------------------
  fprintf(stderr, "\n=== Streaming mode (chunk_frames=25, overlap=2) ===\n");
  {
    sherpa_onnx::GenerationConfig cfg;
    cfg.extra["max_new_tokens"] = "2048";
    cfg.extra["chunk_frames"] = "25";
    cfg.extra["overlap_frames"] = "2";

    // Receives audio chunks as they are decoded. Return 1 to continue,
    // 0 to stop.
    std::vector<float> callback_samples;
    auto on_chunk = [&callback_samples](const float *samples,
                                        int32_t num_samples,
                                        float progress) -> int32_t {
      fprintf(stderr, "chunk: %d samples  progress: %.1f%%\n", num_samples,
              progress * 100);
      if (samples && num_samples > 0) {
        callback_samples.insert(callback_samples.end(), samples,
                                samples + num_samples);
      }
      return 1;
    };

    auto audio = tts.Generate(text, cfg, on_chunk);
    Save("./qwen3-streaming.wav", audio.samples, audio.sample_rate);
    // The raw callback audio (streaming decoder output, stitched chunks)
    Save("./qwen3-streaming-callback.wav", callback_samples,
         audio.sample_rate);
  }

  fprintf(stderr, "\nInput text: %s\n", text.c_str());
  return 0;
}
