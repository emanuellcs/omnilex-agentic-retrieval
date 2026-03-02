import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock transformers before importing QueryTranslator
mock_transformers = MagicMock()
mock_torch = MagicMock()

with patch.dict(sys.modules, {"transformers": mock_transformers, "torch": mock_torch}):
    from omnilex.retrieval.translator import QueryTranslator


class TestQueryTranslator(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        mock_transformers.reset_mock()
        mock_torch.reset_mock()

        # Manually reset side_effect and return_value as reset_mock doesn't do it
        mock_transformers.MarianTokenizer.from_pretrained.side_effect = None
        mock_transformers.MarianMTModel.from_pretrained.side_effect = None

        # Configure the mock tokenizer and model
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()

        mock_transformers.MarianTokenizer.from_pretrained.return_value = (
            self.mock_tokenizer
        )
        mock_transformers.MarianMTModel.from_pretrained.return_value.to.return_value = (
            self.mock_model
        )

    def test_init_with_mocks(self):
        translator = QueryTranslator()
        self.assertIsNotNone(translator.tokenizer)
        self.assertIsNotNone(translator.model)
        mock_transformers.MarianTokenizer.from_pretrained.assert_called_once()
        mock_transformers.MarianMTModel.from_pretrained.assert_called_once()

    def test_translate_caching(self):
        translator = QueryTranslator()

        # Mock translation output
        self.mock_tokenizer.return_value.to.return_value = {"input_ids": [1, 2, 3]}
        self.mock_model.generate.return_value = [[1, 2, 3]]
        self.mock_tokenizer.decode.return_value = "German translation"

        # First call
        res1 = translator.translate("English text")
        self.assertEqual(res1, "German translation")
        self.assertEqual(self.mock_model.generate.call_count, 1)

        # Second call (should hit cache)
        res2 = translator.translate("English text")
        self.assertEqual(res2, "German translation")
        self.assertEqual(self.mock_model.generate.call_count, 1)

    def test_translate_empty_input(self):
        translator = QueryTranslator()
        self.assertEqual(translator.translate(""), "")
        self.assertEqual(translator.translate(None), None)

    def test_fallback_when_model_fails(self):
        # Mock a failure in loading
        mock_transformers.MarianTokenizer.from_pretrained.side_effect = Exception(
            "Load failed"
        )
        translator = QueryTranslator()

        self.assertIsNone(translator.model)
        self.assertEqual(translator.translate("English text"), "English text")


if __name__ == "__main__":
    unittest.main()
