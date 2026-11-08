// sherpa-onnx/csrc/sherpa-onnx-vad-microphone-funasr-nano.cc
//
// Copyright (c)  2025  Xiaomi Corporation
// Copyright (c)  2025  zengyw
//
// This file demonstrates how to use FunASR-nano with VAD and microphone input
// for streaming speech recognition with real-time text output.

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "portaudio.h"  // NOLINT
#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/microphone.h"
#include "sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"

bool stop = false;
std::mutex mutex;
sherpa_onnx::CircularBuffer buffer(16000 * 60);

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void * /*user_data*/) {
  std::lock_guard<std::mutex> lock(mutex);
  buffer.Push(reinterpret_cast<const float *>(input_buffer), frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t /*sig*/) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
FunASR-Nano with VAD and Microphone - Streaming Speech Recognition

This program demonstrates real-time speech recognition using:
- FunASR-nano model for speech-to-text
- Silero VAD for voice activity detection
- Microphone input via PortAudio
- Streaming callback for real-time text output

Usage:

Step 1: Download silero_vad.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

Step 2: Download FunASR-nano model
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2

Step 3: Run
  ./bin/sherpa-onnx-vad-microphone-funasr-nano \
    --silero-vad-model=./silero_vad.onnx \
    --funasr-nano-encoder-adaptor=./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --funasr-nano-llm=./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --funasr-nano-embedding=./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    --funasr-nano-tokenizer=./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --num-threads=2

Environment variables:
  SHERPA_ONNX_MIC_DEVICE      - Microphone device index (default: system default)
  SHERPA_ONNX_MIC_SAMPLE_RATE - Microphone sample rate (default: 16000)
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::VadModelConfig vad_config;
  sherpa_onnx::OfflineRecognizerConfig asr_config;

  // Enable streaming output by default
  bool enable_streaming = true;
  po.Register("enable-streaming", &enable_streaming,
              "Enable streaming text output (token by token)");

  vad_config.Register(&po);
  asr_config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 0) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "VAD Config:\n%s\n", vad_config.ToString().c_str());
  fprintf(stderr, "ASR Config:\n%s\n", asr_config.ToString().c_str());

  if (!vad_config.Validate()) {
    fprintf(stderr, "Errors in vad_config!\n");
    return -1;
  }

  if (!asr_config.Validate()) {
    fprintf(stderr, "Errors in asr_config!\n");
    return -1;
  }

  // Verify FunASR-nano model is configured
  if (asr_config.model_config.funasr_nano.encoder_adaptor.empty()) {
    fprintf(stderr,
            "Error: FunASR-nano model not configured!\n"
            "Please specify --funasr-nano-encoder-adaptor, --funasr-nano-llm, "
            "--funasr-nano-embedding, and --funasr-nano-tokenizer\n");
    return -1;
  }

  fprintf(stderr, "Creating FunASR-nano recognizer...\n");
  sherpa_onnx::OfflineRecognizerFunASRNanoImpl recognizer(asr_config);
  fprintf(stderr, "FunASR-nano recognizer created!\n");

  sherpa_onnx::Microphone mic;

  int32_t device_index = Pa_GetDefaultInputDevice();
  if (device_index == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    fprintf(stderr,
            "If you are using Linux, please check your ALSA configuration\n");
    exit(EXIT_FAILURE);
  }

  const char *pDeviceIndex = std::getenv("SHERPA_ONNX_MIC_DEVICE");
  if (pDeviceIndex) {
    fprintf(stderr, "Using specified device: %s\n", pDeviceIndex);
    device_index = atoi(pDeviceIndex);
  }
  mic.PrintDevices(device_index);

  float mic_sample_rate = 16000;
  const char *pSampleRateStr = std::getenv("SHERPA_ONNX_MIC_SAMPLE_RATE");
  if (pSampleRateStr) {
    mic_sample_rate = atof(pSampleRateStr);
    fprintf(stderr, "Using sample rate %.0f for mic\n", mic_sample_rate);
  }

  if (!mic.OpenDevice(device_index, mic_sample_rate, 1, RecordCallback,
                      nullptr)) {
    fprintf(stderr, "Failed to open device %d\n", device_index);
    exit(EXIT_FAILURE);
  }

  float sample_rate = 16000;
  std::unique_ptr<sherpa_onnx::LinearResample> resampler;
  if (mic_sample_rate != sample_rate) {
    float min_freq = std::min(mic_sample_rate, sample_rate);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler = std::make_unique<sherpa_onnx::LinearResample>(
        mic_sample_rate, sample_rate, lowpass_cutoff, lowpass_filter_width);
  }

  auto vad = std::make_unique<sherpa_onnx::VoiceActivityDetector>(vad_config);

  fprintf(stderr, "\n");
  fprintf(stderr, "========================================\n");
  fprintf(stderr, "  FunASR-Nano Microphone ASR Started\n");
  fprintf(stderr, "========================================\n");
  fprintf(stderr, "Please speak. Press Ctrl+C to exit.\n\n");

  int32_t window_size = vad_config.silero_vad.window_size;
  int32_t segment_index = 0;

  // Streaming callback for real-time text output. The pieces are raw model
  // output; the final, post-processed text is in stream->GetResult().
  std::string streamed_text;
  auto streaming_callback = [&streamed_text](const std::string &text,
                                             int64_t token_id,
                                             bool is_final) -> bool {
    // Print each token as it's generated
    fprintf(stdout, "%s", text.c_str());
    fflush(stdout);
    streamed_text += text;
    return true;  // Continue generation
  };

  while (!stop) {
    {
      std::lock_guard<std::mutex> lock(mutex);

      while (buffer.Size() >= window_size) {
        std::vector<float> samples = buffer.Get(buffer.Head(), window_size);
        buffer.Pop(window_size);

        if (resampler) {
          std::vector<float> tmp;
          resampler->Resample(samples.data(), samples.size(), true, &tmp);
          samples = std::move(tmp);
        }

        vad->AcceptWaveform(samples.data(), samples.size());
      }
    }

    while (!vad->Empty()) {
      const auto &segment = vad->Front();
      auto stream = recognizer.CreateStream();
      stream->AcceptWaveform(sample_rate, segment.samples.data(),
                             segment.samples.size());

      float duration = segment.samples.size() / static_cast<float>(sample_rate);
      fprintf(stderr, "[Segment %d: %.2fs] >>> ", segment_index, duration);

      if (enable_streaming) {
        // Use streaming callback for real-time output
        streamed_text.clear();
        recognizer.DecodeStreamWithCallback(stream.get(), streaming_callback);
        const std::string &final_text = stream->GetResult().text;
        if (final_text != streamed_text) {
          // Post-processing changed the text: show the authoritative result.
          fprintf(stdout, "\n  => %s", final_text.c_str());
        }
      } else {
        // Use standard decoding
        sherpa_onnx::OfflineStream *streams[] = {stream.get()};
        recognizer.DecodeStreams(streams, 1);
        const auto &result = stream->GetResult();
        fprintf(stdout, "%s", result.text.c_str());
      }

      fprintf(stdout, "\n");
      fflush(stdout);

      ++segment_index;
      vad->Pop();
    }

    Pa_Sleep(100);  // Sleep for 100ms
  }

  fprintf(stderr, "\nTotal segments processed: %d\n", segment_index);
  return 0;
}
