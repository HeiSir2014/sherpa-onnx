// sherpa-onnx/csrc/offline-tts-qwen3-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-qwen3-model.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

using json = nlohmann::json;

namespace {

template <typename T>
void ReadIfPresent(const json &j, const char *key, T *value) {
  auto it = j.find(key);
  if (it != j.end() && it->is_number_integer()) *value = it->get<T>();
}

}  // namespace

bool ParseQwen3TtsConfigJson(const std::string &text,
                             Qwen3TtsConfig *config) {
  const json j = json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object()) return false;

  Qwen3TtsConfig c = *config;
  auto talker = j.find("talker_config");
  const json &tc =
      (talker != j.end() && talker->is_object()) ? *talker : j;
  ReadIfPresent(tc, "codec_bos_id", &c.codec_bos_id);
  ReadIfPresent(tc, "codec_eos_token_id", &c.codec_eos_token_id);
  ReadIfPresent(tc, "codec_pad_id", &c.codec_pad_id);
  ReadIfPresent(tc, "codec_nothink_id", &c.codec_nothink_id);
  ReadIfPresent(tc, "codec_think_id", &c.codec_think_id);
  ReadIfPresent(tc, "codec_think_bos_id", &c.codec_think_bos_id);
  ReadIfPresent(tc, "codec_think_eos_id", &c.codec_think_eos_id);
  ReadIfPresent(tc, "num_code_groups", &c.num_code_groups);
  ReadIfPresent(tc, "hidden_size", &c.hidden_size);
  ReadIfPresent(tc, "vocab_size", &c.talker_vocab_size);

  ReadIfPresent(j, "tts_bos_token_id", &c.tts_bos_token_id);
  ReadIfPresent(j, "tts_eos_token_id", &c.tts_eos_token_id);
  ReadIfPresent(j, "tts_pad_token_id", &c.tts_pad_token_id);

  *config = c;
  return true;
}

class OfflineTtsQwen3Model::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    Init();
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    Init(mgr);
  }
#endif

#if __OHOS__
  Impl(NativeResourceManager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    Init(mgr);
  }
#endif

  Ort::Value RunTextProject(Ort::Value input_ids) const {
    auto out = text_project_sess_->Run({}, text_project_input_names_ptr_.data(),
                                       &input_ids, 1,
                                       text_project_output_names_ptr_.data(),
                                       text_project_output_names_ptr_.size());
    return std::move(out[0]);
  }

  Ort::Value RunCodecEmbed(Ort::Value input_ids) const {
    auto out = codec_embed_sess_->Run({}, codec_embed_input_names_ptr_.data(),
                                      &input_ids, 1,
                                      codec_embed_output_names_ptr_.data(),
                                      codec_embed_output_names_ptr_.size());
    return std::move(out[0]);
  }

  Ort::Value RunCodePredictorEmbed(Ort::Value input_ids,
                                   Ort::Value generation_step) const {
    std::array<Ort::Value, 2> inputs = {std::move(input_ids),
                                        std::move(generation_step)};
    auto out = code_predictor_embed_sess_->Run(
        {}, code_predictor_embed_input_names_ptr_.data(), inputs.data(),
        inputs.size(), code_predictor_embed_output_names_ptr_.data(),
        code_predictor_embed_output_names_ptr_.size());
    return std::move(out[0]);
  }

  Ort::Value RunCodePredictor(Ort::Value inputs_embeds,
                              Ort::Value generation_step) const {
    std::array<Ort::Value, 2> inputs = {std::move(inputs_embeds),
                                        std::move(generation_step)};
    auto out = code_predictor_sess_->Run(
        {}, code_predictor_input_names_ptr_.data(), inputs.data(),
        inputs.size(), code_predictor_output_names_ptr_.data(),
        code_predictor_output_names_ptr_.size());
    return std::move(out[0]);
  }

  Ort::Value RunSpeakerEncoder(Ort::Value mels) const {
    auto out = speaker_encoder_sess_->Run(
        {}, speaker_encoder_input_names_ptr_.data(), &mels, 1,
        speaker_encoder_output_names_ptr_.data(),
        speaker_encoder_output_names_ptr_.size());
    return std::move(out[0]);
  }

  Tokenizer12hzDecodeResult RunTokenizer12hzDecode(
      Ort::Value audio_codes) const {
    auto out = tokenizer12hz_decode_sess_->Run(
        {}, tokenizer12hz_decode_input_names_ptr_.data(), &audio_codes, 1,
        tokenizer12hz_decode_output_names_ptr_.data(),
        tokenizer12hz_decode_output_names_ptr_.size());
    Tokenizer12hzDecodeResult result;
    result.audio_values = std::move(out[0]);
    if (out.size() > 1) {
      result.lengths = std::move(out[1]);
    }
    return result;
  }

  // Uses the streaming-optimized session if loaded, otherwise falls back
  // to the batch decoder.  Call this for per-chunk decode in streaming mode.
  Tokenizer12hzDecodeResult RunTokenizer12hzDecodeStream(
      Ort::Value audio_codes) const {
    if (tokenizer12hz_decode_stream_sess_) {
      auto out = tokenizer12hz_decode_stream_sess_->Run(
          {}, tokenizer12hz_decode_stream_input_names_ptr_.data(), &audio_codes,
          1, tokenizer12hz_decode_stream_output_names_ptr_.data(),
          tokenizer12hz_decode_stream_output_names_ptr_.size());
      Tokenizer12hzDecodeResult result;
      result.audio_values = std::move(out[0]);
      if (out.size() > 1) {
        result.lengths = std::move(out[1]);
      }
      return result;
    }
    // Fallback: use batch decoder
    return RunTokenizer12hzDecode(std::move(audio_codes));
  }

  bool HasTokenizer12hzDecodeStream() const {
    return tokenizer12hz_decode_stream_sess_ != nullptr;
  }

  int32_t Tokenizer12hzDecodeStreamMaxFrames() const {
    return tokenizer12hz_decode_stream_max_frames_;
  }

  TalkerPrefillResult RunTalkerPrefill(Ort::Value inputs_embeds,
                                       Ort::Value attention_mask) const {
    std::array<Ort::Value, 2> inputs = {std::move(inputs_embeds),
                                        std::move(attention_mask)};
    auto out = talker_prefill_sess_->Run(
        {}, talker_prefill_input_names_ptr_.data(), inputs.data(),
        inputs.size(), talker_prefill_output_names_ptr_.data(),
        talker_prefill_output_names_ptr_.size());

    TalkerPrefillResult result;
    result.logits = std::move(out[0]);
    result.last_hidden = std::move(out[1]);

    // Remaining outputs are KV-cache tensors
    for (size_t i = 2; i < out.size(); ++i) {
      result.state.kv_cache.push_back(std::move(out[i]));
    }

    return result;
  }

  TalkerDecodeResult RunTalkerDecode(Ort::Value inputs_embeds,
                                     Ort::Value attention_mask,
                                     Qwen3TalkerState state) const {
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(inputs_embeds));
    inputs.push_back(std::move(attention_mask));

    for (auto &kv : state.kv_cache) {
      inputs.push_back(std::move(kv));
    }

    auto out = talker_decode_sess_->Run(
        {}, talker_decode_input_names_ptr_.data(), inputs.data(), inputs.size(),
        talker_decode_output_names_ptr_.data(),
        talker_decode_output_names_ptr_.size());

    TalkerDecodeResult result;
    result.logits = std::move(out[0]);
    result.last_hidden = std::move(out[1]);

    for (size_t i = 2; i < out.size(); ++i) {
      result.state.kv_cache.push_back(std::move(out[i]));
    }

    return result;
  }

  const Qwen3TtsConfig &GetConfig() const { return tts_config_; }

  OrtAllocator *Allocator() const { return allocator_; }

 private:
  void Init() {
    InitTextProject(config_.qwen3.text_project);
    InitCodecEmbed(config_.qwen3.codec_embed);
    InitCodePredictorEmbed(config_.qwen3.code_predictor_embed);
    InitCodePredictor(config_.qwen3.code_predictor);
    InitTalkerPrefill(config_.qwen3.talker_prefill);
    InitTalkerDecode(config_.qwen3.talker_decode);

    if (!config_.qwen3.speaker_encoder.empty()) {
      InitSpeakerEncoder(config_.qwen3.speaker_encoder);
    }

    InitTokenizer12hzDecode(config_.qwen3.tokenizer12hz_decode);

    if (!config_.qwen3.tokenizer12hz_decode_stream.empty()) {
      InitTokenizer12hzDecodeStream(config_.qwen3.tokenizer12hz_decode_stream);
    }

    // Try to load config.json from the model directory
    LoadConfigJson();

    if (tokenizer12hz_decode_stream_sess_) {
      tokenizer12hz_decode_stream_max_frames_ = ProbeStreamDecoderMaxFrames();
    }
  }

  // The stream decoder is traced with a fixed output buffer: its output
  // shape is dynamic in the metadata, but every call returns the same number
  // of samples and audio beyond that is cut off. Decode one frame to learn
  // the buffer size and the samples per frame. Returns 0 if unknown.
  int32_t ProbeStreamDecoderMaxFrames() const {
    std::array<int64_t, 3> shape = {1, 1, tts_config_.num_code_groups};
    Ort::Value codes = Ort::Value::CreateTensor(
        allocator_, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::fill_n(codes.GetTensorMutableData<int64_t>(),
                tts_config_.num_code_groups, 0);

    auto result = RunTokenizer12hzDecodeStream(std::move(codes));
    if (!result.lengths.IsTensor()) return 0;

    const int64_t samples_per_frame =
        result.lengths.GetTensorData<int64_t>()[0];
    const int64_t buffer =
        result.audio_values.GetTensorTypeAndShapeInfo().GetElementCount();
    if (samples_per_frame <= 0) return 0;

    const int32_t max_frames = static_cast<int32_t>(buffer / samples_per_frame);
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Stream decoder holds at most %d frames per call",
                       max_frames);
    }
    return max_frames;
  }

#if __ANDROID_API__ >= 9
  void Init(AAssetManager *mgr) {
    Init();  // Fallback for now
  }
#endif

#if __OHOS__
  void Init(NativeResourceManager *mgr) {
    Init();  // Fallback for now
  }
#endif

  void LoadConfigJson() {
    // Try to find config.json in the same directory as talker_decode
    std::string model_path = config_.qwen3.talker_decode;
    auto pos = model_path.find_last_of("/\\");
    std::string dir =
        (pos != std::string::npos) ? model_path.substr(0, pos) : ".";
    std::string config_path = dir + "/config.json";

    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
      SHERPA_ONNX_LOGE(
          "config.json not found at %s, using default token IDs for "
          "Qwen3-TTS-12Hz-0.6B-Base",
          config_path.c_str());
      return;
    }

    std::ostringstream text;
    text << ifs.rdbuf();
    if (!ParseQwen3TtsConfigJson(text.str(), &tts_config_)) {
      SHERPA_ONNX_LOGE("Failed to parse %s, using default token IDs",
                       config_path.c_str());
      return;
    }

    SHERPA_ONNX_LOGE(
        "Loaded config.json: num_code_groups=%d, "
        "codec_eos=%d, hidden_size=%d",
        tts_config_.num_code_groups,
        static_cast<int>(tts_config_.codec_eos_token_id),
        tts_config_.hidden_size);
  }

  void InitSession(const std::string &model_path,
                   std::unique_ptr<Ort::Session> &sess,
                   std::vector<std::string> &input_names,
                   std::vector<const char *> &input_names_ptr,
                   std::vector<std::string> &output_names,
                   std::vector<const char *> &output_names_ptr) {
    sess = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(model_path), sess_opts_);

    GetInputNames(sess.get(), &input_names, &input_names_ptr);
    GetOutputNames(sess.get(), &output_names, &output_names_ptr);

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Loaded %s: inputs=[", model_path.c_str());
      for (const auto &n : input_names) {
        SHERPA_ONNX_LOGE("  %s", n.c_str());
      }
      SHERPA_ONNX_LOGE("], outputs=[");
      for (const auto &n : output_names) {
        SHERPA_ONNX_LOGE("  %s", n.c_str());
      }
      SHERPA_ONNX_LOGE("]");
    }
  }

  void InitTextProject(const std::string &path) {
    InitSession(path, text_project_sess_, text_project_input_names_,
                text_project_input_names_ptr_, text_project_output_names_,
                text_project_output_names_ptr_);
  }

  void InitCodecEmbed(const std::string &path) {
    InitSession(path, codec_embed_sess_, codec_embed_input_names_,
                codec_embed_input_names_ptr_, codec_embed_output_names_,
                codec_embed_output_names_ptr_);
  }

  void InitCodePredictorEmbed(const std::string &path) {
    InitSession(path, code_predictor_embed_sess_,
                code_predictor_embed_input_names_,
                code_predictor_embed_input_names_ptr_,
                code_predictor_embed_output_names_,
                code_predictor_embed_output_names_ptr_);
  }

  void InitCodePredictor(const std::string &path) {
    InitSession(path, code_predictor_sess_, code_predictor_input_names_,
                code_predictor_input_names_ptr_, code_predictor_output_names_,
                code_predictor_output_names_ptr_);
  }

  void InitTalkerPrefill(const std::string &path) {
    InitSession(path, talker_prefill_sess_, talker_prefill_input_names_,
                talker_prefill_input_names_ptr_, talker_prefill_output_names_,
                talker_prefill_output_names_ptr_);
  }

  void InitTalkerDecode(const std::string &path) {
    InitSession(path, talker_decode_sess_, talker_decode_input_names_,
                talker_decode_input_names_ptr_, talker_decode_output_names_,
                talker_decode_output_names_ptr_);
  }

  void InitSpeakerEncoder(const std::string &path) {
    InitSession(path, speaker_encoder_sess_, speaker_encoder_input_names_,
                speaker_encoder_input_names_ptr_, speaker_encoder_output_names_,
                speaker_encoder_output_names_ptr_);
  }

  void InitTokenizer12hzDecode(const std::string &path) {
    InitSession(path, tokenizer12hz_decode_sess_,
                tokenizer12hz_decode_input_names_,
                tokenizer12hz_decode_input_names_ptr_,
                tokenizer12hz_decode_output_names_,
                tokenizer12hz_decode_output_names_ptr_);
  }

  void InitTokenizer12hzDecodeStream(const std::string &path) {
    InitSession(path, tokenizer12hz_decode_stream_sess_,
                tokenizer12hz_decode_stream_input_names_,
                tokenizer12hz_decode_stream_input_names_ptr_,
                tokenizer12hz_decode_stream_output_names_,
                tokenizer12hz_decode_stream_output_names_ptr_);
  }

 private:
  OfflineTtsModelConfig config_;
  Qwen3TtsConfig tts_config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  // ONNX sessions (speaker_encoder is optional)
  std::unique_ptr<Ort::Session> text_project_sess_;
  std::unique_ptr<Ort::Session> codec_embed_sess_;
  std::unique_ptr<Ort::Session> code_predictor_embed_sess_;
  std::unique_ptr<Ort::Session> code_predictor_sess_;
  std::unique_ptr<Ort::Session> talker_prefill_sess_;
  std::unique_ptr<Ort::Session> talker_decode_sess_;
  std::unique_ptr<Ort::Session> speaker_encoder_sess_;
  std::unique_ptr<Ort::Session> tokenizer12hz_decode_sess_;
  // Optional streaming decoder (small-N trace for low-latency chunk decode)
  std::unique_ptr<Ort::Session> tokenizer12hz_decode_stream_sess_;
  // Frames the stream decoder can return per call; 0 = unknown.
  int32_t tokenizer12hz_decode_stream_max_frames_ = 0;

  // Input/output names for each session
  std::vector<std::string> text_project_input_names_;
  std::vector<const char *> text_project_input_names_ptr_;
  std::vector<std::string> text_project_output_names_;
  std::vector<const char *> text_project_output_names_ptr_;

  std::vector<std::string> codec_embed_input_names_;
  std::vector<const char *> codec_embed_input_names_ptr_;
  std::vector<std::string> codec_embed_output_names_;
  std::vector<const char *> codec_embed_output_names_ptr_;

  std::vector<std::string> code_predictor_embed_input_names_;
  std::vector<const char *> code_predictor_embed_input_names_ptr_;
  std::vector<std::string> code_predictor_embed_output_names_;
  std::vector<const char *> code_predictor_embed_output_names_ptr_;

  std::vector<std::string> code_predictor_input_names_;
  std::vector<const char *> code_predictor_input_names_ptr_;
  std::vector<std::string> code_predictor_output_names_;
  std::vector<const char *> code_predictor_output_names_ptr_;

  std::vector<std::string> talker_prefill_input_names_;
  std::vector<const char *> talker_prefill_input_names_ptr_;
  std::vector<std::string> talker_prefill_output_names_;
  std::vector<const char *> talker_prefill_output_names_ptr_;

  std::vector<std::string> talker_decode_input_names_;
  std::vector<const char *> talker_decode_input_names_ptr_;
  std::vector<std::string> talker_decode_output_names_;
  std::vector<const char *> talker_decode_output_names_ptr_;

  std::vector<std::string> speaker_encoder_input_names_;
  std::vector<const char *> speaker_encoder_input_names_ptr_;
  std::vector<std::string> speaker_encoder_output_names_;
  std::vector<const char *> speaker_encoder_output_names_ptr_;

  std::vector<std::string> tokenizer12hz_decode_input_names_;
  std::vector<const char *> tokenizer12hz_decode_input_names_ptr_;
  std::vector<std::string> tokenizer12hz_decode_output_names_;
  std::vector<const char *> tokenizer12hz_decode_output_names_ptr_;

  std::vector<std::string> tokenizer12hz_decode_stream_input_names_;
  std::vector<const char *> tokenizer12hz_decode_stream_input_names_ptr_;
  std::vector<std::string> tokenizer12hz_decode_stream_output_names_;
  std::vector<const char *> tokenizer12hz_decode_stream_output_names_ptr_;
};

OfflineTtsQwen3Model::OfflineTtsQwen3Model(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineTtsQwen3Model::OfflineTtsQwen3Model(AAssetManager *mgr,
                                           const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

#if __OHOS__
OfflineTtsQwen3Model::OfflineTtsQwen3Model(NativeResourceManager *mgr,
                                           const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineTtsQwen3Model::~OfflineTtsQwen3Model() = default;

Ort::Value OfflineTtsQwen3Model::RunTextProject(Ort::Value input_ids) const {
  return impl_->RunTextProject(std::move(input_ids));
}

Ort::Value OfflineTtsQwen3Model::RunCodecEmbed(Ort::Value input_ids) const {
  return impl_->RunCodecEmbed(std::move(input_ids));
}

Ort::Value OfflineTtsQwen3Model::RunCodePredictorEmbed(
    Ort::Value input_ids, Ort::Value generation_step) const {
  return impl_->RunCodePredictorEmbed(std::move(input_ids),
                                      std::move(generation_step));
}

Ort::Value OfflineTtsQwen3Model::RunCodePredictor(
    Ort::Value inputs_embeds, Ort::Value generation_step) const {
  return impl_->RunCodePredictor(std::move(inputs_embeds),
                                 std::move(generation_step));
}

OfflineTtsQwen3Model::TalkerPrefillResult
OfflineTtsQwen3Model::RunTalkerPrefill(Ort::Value inputs_embeds,
                                       Ort::Value attention_mask) const {
  return impl_->RunTalkerPrefill(std::move(inputs_embeds),
                                 std::move(attention_mask));
}

OfflineTtsQwen3Model::TalkerDecodeResult OfflineTtsQwen3Model::RunTalkerDecode(
    Ort::Value inputs_embeds, Ort::Value attention_mask,
    Qwen3TalkerState state) const {
  return impl_->RunTalkerDecode(std::move(inputs_embeds),
                                std::move(attention_mask), std::move(state));
}

Ort::Value OfflineTtsQwen3Model::RunSpeakerEncoder(Ort::Value mels) const {
  return impl_->RunSpeakerEncoder(std::move(mels));
}

OfflineTtsQwen3Model::Tokenizer12hzDecodeResult
OfflineTtsQwen3Model::RunTokenizer12hzDecode(Ort::Value audio_codes) const {
  return impl_->RunTokenizer12hzDecode(std::move(audio_codes));
}

OfflineTtsQwen3Model::Tokenizer12hzDecodeResult
OfflineTtsQwen3Model::RunTokenizer12hzDecodeStream(
    Ort::Value audio_codes) const {
  return impl_->RunTokenizer12hzDecodeStream(std::move(audio_codes));
}

int32_t OfflineTtsQwen3Model::Tokenizer12hzDecodeStreamMaxFrames() const {
  return impl_->Tokenizer12hzDecodeStreamMaxFrames();
}

bool OfflineTtsQwen3Model::HasTokenizer12hzDecodeStream() const {
  return impl_->HasTokenizer12hzDecodeStream();
}

const Qwen3TtsConfig &OfflineTtsQwen3Model::GetConfig() const {
  return impl_->GetConfig();
}

OrtAllocator *OfflineTtsQwen3Model::Allocator() const {
  return impl_->Allocator();
}

}  // namespace sherpa_onnx
