// cxx-api-examples/funasr-nano-cxx-api.cc
//
// Copyright (c)  2025  zengyw
//
// This file demonstrates how to use FunASR-nano with streaming ASR output.
// The streaming API provides real-time token-by-token output via callback.
// It also tests consecutive recognitions to verify reproducibility.
//
// clang-format off
//
// Usage:
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
// tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
// rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
//
// ./funasr-nano-cxx-api [wave_file]
//
// clang-format on

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int32_t main(int32_t argc, char *argv[]) {
  // Configure the recognizer
  sherpa_onnx::OfflineRecognizerConfig config;
  config.model_config.num_threads = 2;
  config.model_config.debug = false;
  config.model_config.provider = "cpu";

  // FunASR-nano model paths
  const std::string model_dir = "./sherpa-onnx-funasr-nano-int8-2025-12-30";
  auto &funasr_nano = config.model_config.funasr_nano;
  funasr_nano.encoder_adaptor = model_dir + "/encoder_adaptor.int8.onnx";
  funasr_nano.llm = model_dir + "/llm.int8.onnx";
  funasr_nano.embedding = model_dir + "/embedding.int8.onnx";
  funasr_nano.tokenizer = model_dir + "/Qwen3-0.6B";

  // repetition_penalty and no_repeat_ngram_size are off by default. The int8
  // LLM in this package can loop on long or noisy audio. Prefer an fp16 LLM
  // for real workloads; enabling the penalties trades looping for dropped
  // words, e.g.:
  //   funasr_nano.repetition_penalty = 1.2f;
  //   funasr_nano.no_repeat_ngram_size = 3;

  // Validate configuration
  if (!config.Validate()) {
    std::cerr << "Invalid config. Please check your model paths.\n";
    return -1;
  }

  std::cout << "Loading model...\n";
  sherpa_onnx::OfflineRecognizerFunASRNanoImpl recognizer(config);
  std::cout << "Model loaded.\n\n";

  // Read audio file - use command line arg or default
  std::string wave_filename =
      "./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav";

  if (argc > 1) {
    wave_filename = argv[1];
  }

  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(wave_filename, &sample_rate, &is_ok);

  if (!is_ok) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  float duration = samples.size() / static_cast<float>(sample_rate);
  std::cout << "Audio: " << wave_filename << "\n";
  std::cout << "Duration: " << std::fixed << std::setprecision(2) << duration
            << "s\n";
  std::cout << std::string(60, '=') << "\n\n";

  // ============================================================
  // Test 1: Non-streaming API (DecodeStreams)
  // ============================================================
  std::cout << "Test 1: Non-streaming API\n";
  std::cout << std::string(60, '-') << "\n";

  auto stream1 = recognizer.CreateStream();
  stream1->AcceptWaveform(sample_rate, samples.data(), samples.size());

  const auto begin1 = std::chrono::steady_clock::now();
  sherpa_onnx::OfflineStream *streams1[] = {stream1.get()};
  recognizer.DecodeStreams(streams1, 1);
  const auto end1 = std::chrono::steady_clock::now();

  const float elapsed1 =
      std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1)
          .count() /
      1000.0f;

  auto result1 = stream1->GetResult();
  std::cout << "Result: " << result1.text << "\n";
  std::cout << "Time:   " << std::fixed << std::setprecision(3) << elapsed1
            << "s\n";
  std::cout << "RTF:    " << std::fixed << std::setprecision(4)
            << elapsed1 / duration << "\n\n";

  // ============================================================
  // Test 2: Streaming API with callback (DecodeStreamWithCallback)
  // ============================================================
  std::cout << "Test 2: Streaming API with callback\n";
  std::cout << std::string(60, '-') << "\n";

  auto stream2 = recognizer.CreateStream();
  stream2->AcceptWaveform(sample_rate, samples.data(), samples.size());

  const auto begin2 = std::chrono::steady_clock::now();
  int32_t token_count = 0;
  std::string streaming_text;

  auto streaming_callback = [&token_count, &streaming_text](
                                const std::string &text, int64_t token_id,
                                bool is_final) -> bool {
    std::cout << text << std::flush;
    streaming_text += text;
    if (token_id >= 0) {
      ++token_count;
    }
    if (is_final) {
      std::cout << std::endl;
    }
    return true;  // Continue generation
  };

  std::cout << ">>> ";
  recognizer.DecodeStreamWithCallback(stream2.get(), streaming_callback);
  const auto end2 = std::chrono::steady_clock::now();

  const float elapsed2 =
      std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2)
          .count() /
      1000.0f;

  auto result2 = stream2->GetResult();
  std::cout << "Result: " << result2.text << "\n";
  std::cout << "Tokens: " << token_count << "\n";
  std::cout << "Time:   " << std::fixed << std::setprecision(3) << elapsed2
            << "s\n";
  std::cout << "RTF:    " << std::fixed << std::setprecision(4)
            << elapsed2 / duration << "\n\n";

  // ============================================================
  // Test 3: Second non-streaming call (test reproducibility)
  // ============================================================
  std::cout << "Test 3: Second non-streaming call (reproducibility test)\n";
  std::cout << std::string(60, '-') << "\n";

  auto stream3 = recognizer.CreateStream();
  stream3->AcceptWaveform(sample_rate, samples.data(), samples.size());

  const auto begin3 = std::chrono::steady_clock::now();
  sherpa_onnx::OfflineStream *streams3[] = {stream3.get()};
  recognizer.DecodeStreams(streams3, 1);
  const auto end3 = std::chrono::steady_clock::now();

  const float elapsed3 =
      std::chrono::duration_cast<std::chrono::milliseconds>(end3 - begin3)
          .count() /
      1000.0f;

  auto result3 = stream3->GetResult();
  std::cout << "Result: " << result3.text << "\n";
  std::cout << "Time:   " << std::fixed << std::setprecision(3) << elapsed3
            << "s\n";
  std::cout << "RTF:    " << std::fixed << std::setprecision(4)
            << elapsed3 / duration << "\n\n";

  // ============================================================
  // Compare results
  // ============================================================
  std::cout << std::string(60, '=') << "\n";
  std::cout << "Comparison:\n";
  std::cout << std::string(60, '-') << "\n";

  bool match_1_3 = (result1.text == result3.text);
  std::cout << "Test 1 vs Test 3 (reproducibility): "
            << (match_1_3 ? "PASS" : "FAIL") << "\n";

  if (!match_1_3) {
    std::cout << "  Test 1: " << result1.text << "\n";
    std::cout << "  Test 3: " << result3.text << "\n";
  }

  bool match_1_2 = (result1.text == result2.text);
  std::cout << "Test 1 vs Test 2 (streaming == non-streaming): "
            << (match_1_2 ? "PASS" : "FAIL") << "\n";

  if (!match_1_2) {
    std::cout << "  Test 1: " << result1.text << "\n";
    std::cout << "  Test 2: " << result2.text << "\n";
  }

  std::cout << std::string(60, '=') << "\n";

  return (match_1_3 && match_1_2) ? 0 : 1;
}
