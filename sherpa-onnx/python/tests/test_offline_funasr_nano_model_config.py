# sherpa-onnx/python/tests/test_offline_funasr_nano_model_config.py
#
# To run this single test, use
#
#  ctest --verbose -R  test_offline_funasr_nano_model_config_py

import inspect
import unittest

# Import only the package: importing the raw _sherpa_onnx extension as well
# would load it twice and pybind11 rejects the duplicate type registration.
import sherpa_onnx


class TestOfflineFunASRNanoModelConfig(unittest.TestCase):
    def test_repetition_penalty_is_disabled_by_default(self):
        config = sherpa_onnx.OfflineFunASRNanoModelConfig()
        self.assertAlmostEqual(config.repetition_penalty, 1.0)

    def test_ngram_blocking_is_disabled_by_default(self):
        config = sherpa_onnx.OfflineFunASRNanoModelConfig()
        self.assertEqual(config.no_repeat_ngram_size, 0)

    def test_penalties_are_writable(self):
        config = sherpa_onnx.OfflineFunASRNanoModelConfig()
        config.repetition_penalty = 1.2
        config.no_repeat_ngram_size = 3
        self.assertEqual(
            (round(config.repetition_penalty, 4), config.no_repeat_ngram_size),
            (1.2, 3),
        )

    def test_max_tokens_per_second_is_writable_and_zero_allowed(self):
        config = sherpa_onnx.OfflineFunASRNanoModelConfig()
        config.max_tokens_per_second = 0
        self.assertEqual(config.max_tokens_per_second, 0)

    def test_from_funasr_nano_defaults_match_cpp(self):
        # Compare against the C++ struct defaults, not literals, so the two
        # cannot silently diverge.
        cpp = sherpa_onnx.OfflineFunASRNanoModelConfig()
        params = inspect.signature(
            sherpa_onnx.OfflineRecognizer.from_funasr_nano
        ).parameters
        for name in (
            "repetition_penalty",
            "no_repeat_ngram_size",
            "max_tokens_per_second",
            "user_prompt",
            "system_prompt",
            "max_new_tokens",
            "seed",
        ):
            with self.subTest(name=name):
                expected = getattr(cpp, name)
                actual = params[name].default
                if isinstance(expected, float):
                    self.assertAlmostEqual(actual, expected, places=5)
                else:
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
