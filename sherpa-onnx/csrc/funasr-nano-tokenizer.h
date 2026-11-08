// sherpa-onnx/csrc/funasr-nano-tokenizer.h
//
// Copyright (c)  2025  zengyw
//
// A self-contained Qwen3 ByteLevel-BPE tokenizer implementation.
// - No dependency on tokenizers-cpp / HF tokenizers
// - Loads vocab.json + merges.txt + tokenizer.json(added_tokens)
// - Supports AddedTokens via Trie longest-match
// - ByteLevel bytes_to_unicode encode/decode
// - Streaming decode support for real-time ASR output

#ifndef SHERPA_ONNX_CSRC_FUNASR_NANO_TOKENIZER_H_
#define SHERPA_ONNX_CSRC_FUNASR_NANO_TOKENIZER_H_

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sherpa_onnx {

// Callback type for streaming token output
// Parameters:
//   - text: The decoded text chunk (may be partial for multi-byte characters)
//   - token_id: The token ID that produced this text
//   - is_final: True if this is the last token in the sequence
// Return: false to stop generation early, true to continue
using StreamingTokenCallback =
    std::function<bool(const std::string &text, int64_t token_id, bool is_final)>;

// State class for streaming decode operations
// Maintains pending bytes across token boundaries for proper UTF-8 handling
class StreamingDecodeState {
 public:
  StreamingDecodeState() = default;

  // Get mutable reference to pending bytes buffer
  std::string &PendingBytes() { return pending_bytes_; }
  const std::string &PendingBytes() const { return pending_bytes_; }

  // Get/set token count
  int32_t TokenCount() const { return token_count_; }
  void IncrementTokenCount() { ++token_count_; }

  // Reset state for new sequence
  void Reset() {
    pending_bytes_.clear();
    token_count_ = 0;
  }

  // Check if there are pending bytes
  bool HasPendingBytes() const { return !pending_bytes_.empty(); }

  // Flush any remaining pending bytes (with replacement chars for invalid)
  std::string FlushPendingBytes();

 private:
  std::string pending_bytes_;
  int32_t token_count_ = 0;
};

class FunASRNanoTokenizer {
 public:
  explicit FunASRNanoTokenizer(const std::string &tokenizer_dir);

  template <typename Manager>
  FunASRNanoTokenizer(Manager *mgr, const std::string &tokenizer_dir);

  std::vector<int64_t> Encode(const std::string &text);
  std::string Decode(const std::vector<int64_t> &token_ids);
  std::string GetTokenStringStreaming(int64_t token_id,
                                      std::string *pending_bytes) const;

  // Streaming decode with state management
  // Decodes a single token using the provided state, returns the text chunk
  std::string DecodeTokenStreaming(int64_t token_id,
                                   StreamingDecodeState *state) const;

  // Streaming decode with callback
  // Decodes tokens one by one, calling the callback for each text chunk
  // Returns the complete decoded text
  // If callback returns false, stops early and returns accumulated text
  std::string DecodeWithCallback(const std::vector<int64_t> &token_ids,
                                 const StreamingTokenCallback &callback) const;

  // Check if a token ID is a special token (EOS, PAD, etc.)
  bool IsSpecialToken(int64_t token_id) const;

  // Get vocabulary size
  int32_t GetVocabSize() const {
    return static_cast<int32_t>(id2token_.size());
  }

  // Get token string by ID (without streaming, returns raw token)
  std::string IdToToken(int64_t token_id) const;

  int64_t GetEosTokenId() const { return eos_token_id_; }
  int64_t GetPadTokenId() const { return pad_token_id_; }
  int64_t GetImEndTokenId() const { return im_end_token_id_; }
  int64_t GetImStartTokenId() const { return im_start_token_id_; }

  // Public structures for helper functions
  struct AddedToken {
    std::string content;
    int32_t id = -1;
    bool single_word = false;
    bool lstrip = false;
    bool rstrip = false;
    bool normalized = false;
    bool special = false;
  };

  struct TrieNode {
    std::unordered_map<uint8_t, int32_t> next;
    int32_t token_index = -1;  // index in added_tokens_ if terminal
  };

 private:
  void Init(const std::string &tokenizer_dir);

  template <typename Manager>
  void Init(Manager *mgr, const std::string &tokenizer_dir);

  void FinalizeSpecialIds();

 private:
  // Special ids
  int64_t eos_token_id_ = -1;
  int64_t pad_token_id_ = -1;
  int64_t im_end_token_id_ = -1;
  int64_t im_start_token_id_ = -1;

  std::unordered_set<int32_t> special_ids_;

  // Vocab: token <-> id
  std::unordered_map<std::string, int32_t> token2id_;
  std::vector<std::string> id2token_;

  // merges ranks: "left\tright" -> rank
  std::unordered_map<std::string, int32_t> merges_rank_;

  // BPE cache: bytelevel_word -> list of merged tokens
  std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

  // bytes_to_unicode mapping (ByteLevel)
  std::string byte_to_unicode_[256];
  std::unordered_map<std::string, uint8_t> unicode_to_byte_;

  // AddedTokens
  std::vector<AddedToken> added_tokens_;
  std::vector<TrieNode> trie_;
  std::unordered_set<std::string> added_token_contents_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FUNASR_NANO_TOKENIZER_H_
