// sherpa-onnx/csrc/offline-tts-qwen3-model-test.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/offline-tts-qwen3-model.h"

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

// Hugging Face config.json: talker fields nested under "talker_config".
TEST(ParseQwen3TtsConfigJson, ReadsNestedHuggingFaceFormat) {
  Qwen3TtsConfig config;
  ASSERT_TRUE(ParseQwen3TtsConfigJson(R"({
    "tts_bos_token_id": 11, "tts_eos_token_id": 12, "tts_pad_token_id": 13,
    "talker_config": {
      "num_code_groups": 8, "hidden_size": 512, "vocab_size": 4096,
      "codec_bos_id": 1, "codec_eos_token_id": 2, "codec_pad_id": 3,
      "codec_nothink_id": 4, "codec_think_id": 5,
      "codec_think_bos_id": 6, "codec_think_eos_id": 7
    }
  })", &config));

  EXPECT_EQ(config.num_code_groups, 8);
  EXPECT_EQ(config.hidden_size, 512);
  EXPECT_EQ(config.talker_vocab_size, 4096);
  EXPECT_EQ(config.codec_bos_id, 1);
  EXPECT_EQ(config.codec_eos_token_id, 2);
  EXPECT_EQ(config.codec_pad_id, 3);
  EXPECT_EQ(config.codec_nothink_id, 4);
  EXPECT_EQ(config.codec_think_id, 5);
  EXPECT_EQ(config.codec_think_bos_id, 6);
  EXPECT_EQ(config.codec_think_eos_id, 7);
  EXPECT_EQ(config.tts_bos_token_id, 11);
  EXPECT_EQ(config.tts_eos_token_id, 12);
  EXPECT_EQ(config.tts_pad_token_id, 13);
}

// scripts/qwen3-tts/export-onnx.py writes every field at the top level.
TEST(ParseQwen3TtsConfigJson, ReadsFlatExporterFormat) {
  Qwen3TtsConfig config;
  ASSERT_TRUE(ParseQwen3TtsConfigJson(R"({
    "model_type": "qwen3-tts-12hz",
    "num_code_groups": 8, "hidden_size": 512, "vocab_size": 4096,
    "codec_bos_id": 1, "codec_eos_token_id": 2, "codec_pad_id": 3,
    "codec_nothink_id": 4, "codec_think_id": 5,
    "codec_think_bos_id": 6, "codec_think_eos_id": 7,
    "tts_bos_token_id": 11, "tts_eos_token_id": 12, "tts_pad_token_id": 13
  })", &config));

  EXPECT_EQ(config.num_code_groups, 8);
  EXPECT_EQ(config.hidden_size, 512);
  EXPECT_EQ(config.talker_vocab_size, 4096);
  EXPECT_EQ(config.codec_bos_id, 1);
  EXPECT_EQ(config.codec_eos_token_id, 2);
  EXPECT_EQ(config.codec_think_eos_id, 7);
  EXPECT_EQ(config.tts_pad_token_id, 13);
}

TEST(ParseQwen3TtsConfigJson, KeepsDefaultsForMissingFields) {
  const Qwen3TtsConfig defaults;
  Qwen3TtsConfig config;
  ASSERT_TRUE(ParseQwen3TtsConfigJson(R"({"codec_eos_token_id": 99})",
                                      &config));
  EXPECT_EQ(config.codec_eos_token_id, 99);
  EXPECT_EQ(config.num_code_groups, defaults.num_code_groups);
  EXPECT_EQ(config.tts_bos_token_id, defaults.tts_bos_token_id);
}

TEST(ParseQwen3TtsConfigJson, RejectsInvalidJsonAndKeepsDefaults) {
  const Qwen3TtsConfig defaults;
  Qwen3TtsConfig config;
  EXPECT_FALSE(ParseQwen3TtsConfigJson("{not json", &config));
  EXPECT_EQ(config.codec_eos_token_id, defaults.codec_eos_token_id);
}

}  // namespace
}  // namespace sherpa_onnx
