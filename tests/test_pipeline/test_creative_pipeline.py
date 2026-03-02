import unittest
from unittest.mock import MagicMock, patch
import omnilex.pipeline.creative_pipeline


class TestCreativePipeline(unittest.TestCase):
    def setUp(self):
        self.mock_hybrid = MagicMock()
        self.mock_reranker = MagicMock()
        self.corpus_set = {"A", "B", "C"}

        # Create mocks for the genai SDK
        self.mock_genai = MagicMock()
        self.mock_types = MagicMock()
        self.mock_errors = MagicMock()
        self.mock_client = MagicMock()
        self.mock_genai.Client.return_value = self.mock_client

        # Patch the genai references in the module
        self.patcher_genai = patch(
            "omnilex.pipeline.creative_pipeline.genai", self.mock_genai
        )
        self.patcher_types = patch(
            "omnilex.pipeline.creative_pipeline.types", self.mock_types
        )
        self.patcher_errors = patch(
            "omnilex.pipeline.creative_pipeline.errors", self.mock_errors
        )

        self.patcher_genai.start()
        self.patcher_types.start()
        self.patcher_errors.start()

        # Initialize pipeline
        self.pipeline = omnilex.pipeline.creative_pipeline.CreativePipeline(
            self.mock_hybrid,
            self.mock_reranker,
            gemini_api_key="fake_key",
            corpus_citation_set=self.corpus_set,
        )

    def tearDown(self):
        self.patcher_genai.stop()
        self.patcher_types.stop()
        self.patcher_errors.stop()

    def test_close_client(self):
        self.pipeline.close()
        self.mock_client.close.assert_called_once()
        self.assertIsNone(self.pipeline.client)

    def test_context_manager(self):
        # We need to ensure CreativePipeline uses our mock when initialized
        with omnilex.pipeline.creative_pipeline.CreativePipeline(
            self.mock_hybrid, self.mock_reranker, gemini_api_key="key"
        ) as cp:
            self.assertEqual(cp.client, self.mock_client)

        self.mock_client.close.assert_called_once()

    def test_optional_api_key(self):
        # Should call Client(api_key=None) which uses env vars
        omnilex.pipeline.creative_pipeline.CreativePipeline(
            self.mock_hybrid, self.mock_reranker
        )
        self.mock_genai.Client.assert_called_with(api_key=None)

    def test_parse_arbiter_response(self):
        response = """
Based on the debate:
INCLUDE: A
EXCLUDE: B
INCLUDE: C
        """
        candidates = [{"citation": "A"}, {"citation": "B"}, {"citation": "C"}]
        selected = self.pipeline._parse_arbiter_response(response, candidates)

        self.assertIn("A", selected)
        self.assertIn("C", selected)
        self.assertNotIn("B", selected)

    def test_parse_arbiter_response_fallback(self):
        # Test fallback when format is slightly off
        response = "The Arbiter decides to INCLUDE: A and EXCLUDE: B."
        candidates = [{"citation": "A"}, {"citation": "B"}]
        selected = self.pipeline._parse_arbiter_response(response, candidates)
        self.assertIn("A", selected)


if __name__ == "__main__":
    unittest.main()
