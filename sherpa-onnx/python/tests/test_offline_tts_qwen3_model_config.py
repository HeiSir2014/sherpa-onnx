# sherpa-onnx/python/tests/test_offline_tts_qwen3_model_config.py
#
# To run this single test, use
#
#  ctest --verbose -R  test_offline_tts_qwen3_model_config_py

import unittest

# Import only the package: importing the raw _sherpa_onnx extension as well
# would load it twice and pybind11 rejects the duplicate type registration.
import sherpa_onnx


class TestOfflineTtsQwen3ModelConfig(unittest.TestCase):
    def test_is_exported_from_the_package(self):
        self.assertTrue(hasattr(sherpa_onnx, "OfflineTtsQwen3ModelConfig"))

    def test_stream_decoder_is_empty_by_default(self):
        config = sherpa_onnx.OfflineTtsQwen3ModelConfig()
        self.assertEqual(config.tokenizer12hz_decode_stream, "")

    def test_stream_decoder_is_writable(self):
        config = sherpa_onnx.OfflineTtsQwen3ModelConfig()
        config.tokenizer12hz_decode_stream = "decode_stream.onnx"
        self.assertEqual(config.tokenizer12hz_decode_stream, "decode_stream.onnx")

    def test_stream_decoder_is_a_constructor_argument(self):
        config = sherpa_onnx.OfflineTtsQwen3ModelConfig(
            tokenizer12hz_decode="decode.onnx",
            tokenizer12hz_decode_stream="decode_stream.onnx",
        )
        self.assertEqual(
            (config.tokenizer12hz_decode, config.tokenizer12hz_decode_stream),
            ("decode.onnx", "decode_stream.onnx"),
        )

    def test_stream_decoder_appears_in_str(self):
        config = sherpa_onnx.OfflineTtsQwen3ModelConfig()
        config.tokenizer12hz_decode_stream = "decode_stream.onnx"
        self.assertIn("decode_stream.onnx", str(config))


if __name__ == "__main__":
    unittest.main()
