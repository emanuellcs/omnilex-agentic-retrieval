import unittest
from unittest.mock import MagicMock, patch

from omnilex.retrieval import translator
from omnilex.retrieval.translator import QueryTranslator

mock_transformers = MagicMock()
mock_torch = MagicMock()


class TestQueryTranslator(unittest.TestCase):
    def setUp(self):
        self.tokenizer_patch = patch.object(
            translator,
            "MarianTokenizer",
            mock_transformers.MarianTokenizer,
        )
        self.model_patch = patch.object(
            translator,
            "MarianMTModel",
            mock_transformers.MarianMTModel,
        )
        self.torch_patch = patch.object(translator, "torch", mock_torch)
        self.tokenizer_patch.start()
        self.model_patch.start()
        self.torch_patch.start()

        # Reset mocks
        mock_transformers.reset_mock()
        mock_torch.reset_mock()
        mock_torch.cuda.is_available.return_value = False

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

    def tearDown(self):
        self.torch_patch.stop()
        self.model_patch.stop()
        self.tokenizer_patch.stop()

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
