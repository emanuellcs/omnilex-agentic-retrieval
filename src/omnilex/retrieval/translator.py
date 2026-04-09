from __future__ import annotations

import logging
from pathlib import Path

try:
    from transformers import MarianMTModel, MarianTokenizer
    import torch
except ImportError:
    MarianMTModel = None
    MarianTokenizer = None
    torch = None

logger = logging.getLogger(__name__)


class QueryTranslator:
    """Translate English queries to German using Helsinki-NLP/opus-mt-en-de.

    Uses a simple dictionary cache to avoid re-translating same queries.
    """

    def __init__(
        self, model_name: str = "Helsinki-NLP/opus-mt-en-de", device: str | None = None
    ):
        """Initialize QueryTranslator.

        Args:
            model_name: HuggingFace model name or local path
            device: Device to use (cuda or cpu). Auto-detects if None.
        """
        self.model_name = self._resolve_model_path(model_name)
        self._cache: dict[str, str] = {}

        if MarianMTModel is None or MarianTokenizer is None:
            logger.warning(
                "transformers not installed. QueryTranslator will return original text."
            )
            self.model = None
            self.tokenizer = None
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        try:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)
            logger.info(f"Loaded translation model {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load translation model {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model name to a local path if available."""
        # 1. Check if it's already a valid local path
        if Path(model_name).exists():
            return model_name

        # 2. Check Kaggle input directory
        kaggle_path = Path("/kaggle/input/omnilex-models") / model_name.split("/")[-1]
        if kaggle_path.exists():
            return str(kaggle_path)

        # 3. Fallback to original name (HF Hub)
        return model_name

    def translate(self, text: str) -> str:
        """Translate English text to German.

        Args:
            text: English text

        Returns:
            German translation or original text if translation fails
        """
        if not text or not text.strip():
            return text

        if text in self._cache:
            return self._cache[text]

        if self.model is None or self.tokenizer is None:
            return text

        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                translated_tokens = self.model.generate(**inputs)

            translation = self.tokenizer.decode(
                translated_tokens[0], skip_special_tokens=True
            )
            self._cache[text] = translation
            return translation
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a batch of English texts to German.

        Args:
            texts: List of English texts

        Returns:
            List of German translations
        """
        return [self.translate(t) for t in texts]

    def clear_cache(self) -> None:
        """Clear the translation cache."""
        self._cache.clear()
