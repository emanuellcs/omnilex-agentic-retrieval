from __future__ import annotations

import logging
import re
import time
from typing import Any

try:
    from google import genai
    from google.genai import errors, types
except ImportError:
    genai = None
    types = None
    errors = None

from omnilex.pipeline.full_pipeline import FullPipeline
from omnilex.pipeline.hybrid_retriever import HybridRetriever
from omnilex.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class CreativePipeline(FullPipeline):
    """Creative prize pipeline using an adversarial three-agent debate with Gemini."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        gemini_api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
        corpus_citation_set: set[str] | None = None,
    ):
        """Initialize CreativePipeline.

        Args:
            hybrid_retriever: Orchestrator for retrieval
            reranker: Cross-encoder reranker
            gemini_api_key: API key for Gemini. If None, uses GEMINI_API_KEY or GOOGLE_API_KEY env vars.
            model: Gemini model name
            corpus_citation_set: Grounding set
        """
        super().__init__(hybrid_retriever, reranker, corpus_citation_set=corpus_citation_set)
        self.gemini_api_key = gemini_api_key
        self.model_name = model

        if genai is None:
            logger.warning(
                "google-genai not installed. CreativePipeline will fall back to thresholding."
            )
            self.client = None
        else:
            try:
                # If gemini_api_key is None, genai.Client() will look for environment variables
                self.client = genai.Client(api_key=gemini_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.client = None

    def __enter__(self) -> CreativePipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the Gemini client to release resources."""
        if self.client:
            self.client.close()
            self.client = None

    def _build_advocate_prompt(self, query: str, candidates: list[dict[str, Any]]) -> str:
        candidates_text = "\n".join(
            [f"- {c['citation']}: {str(c.get('text', ''))[:300]}..." for c in candidates[:30]]
        )
        return f"""You are a Swiss legal Advocate. Your goal is to argue FOR the relevance of each citation provided below in the context of the legal query.
Be inclusive: if a citation might be relevant, argue for its inclusion.

Query: {query}

Candidates:
{candidates_text}

For each citation, provide a brief argument why it is relevant."""

    def _build_devils_advocate_prompt(
        self, query: str, candidates: list[dict[str, Any]], advocate_response: str
    ) -> str:
        candidates_text = "\n".join([f"- {c['citation']}" for c in candidates[:30]])
        return f"""You are a Swiss legal Devil's Advocate. Your goal is to CHALLENGE the relevance of each citation, even if the Advocate argued for it.
Be exclusive: if a citation is not strictly necessary or only tangentially related, argue for its exclusion.

Query: {query}

Advocate's Arguments:
{advocate_response}

Candidates list:
{candidates_text}

Challenge the Advocate's arguments for each citation."""

    def _build_arbiter_prompt(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        advocate_response: str,
        devils_response: str,
    ) -> str:
        candidates_text = "\n".join([f"- {c['citation']}" for c in candidates[:30]])
        return f"""You are the Arbiter in a Swiss legal research task. Your goal is to maximize the F1 score: include a citation IF AND ONLY IF it is directly relevant and its omission would be a mistake.

Query: {query}

Full Debate Transcript:
ADVOCATE:
{advocate_response}

DEVIL'S ADVOCATE:
{devils_response}

Candidates list:
{candidates_text}

Make a final decision for each citation. Use this EXACT format for each citation:
INCLUDE: [Citation String]
OR
EXCLUDE: [Citation String]

Ensure you copy the citation string exactly from the candidates list."""

    def _parse_arbiter_response(self, response: str, candidates: list[dict[str, Any]]) -> list[str]:
        # Extract all INCLUDE lines
        include_pattern = r"^INCLUDE\s*:\s*(.+)$"
        selected_citations = []

        for line in response.split("\n"):
            match = re.search(include_pattern, line, re.IGNORECASE)
            if match:
                cit = match.group(1).strip()
                # Grounding
                if cit in self.corpus_citation_set:
                    selected_citations.append(cit)

        # Fallback: if parsing failed to find any INCLUDE, use the candidate citations themselves
        # as the model might have formatted it slightly differently
        if not selected_citations:
            for cand in candidates[:30]:
                if f"INCLUDE: {cand['citation']}" in response:
                    selected_citations.append(cand["citation"])

        return list(set(selected_citations))

    def run_query_with_debate(self, query: str) -> list[str]:
        """Run the full pipeline using the three-agent debate mechanism."""
        if not self.client:
            logger.warning("Gemini client not available, falling back to thresholding.")
            return self.run_query(query)

        try:
            # 1. Get top candidates (first 2 stages of parent)
            candidates = self.hybrid_retriever.retrieve(query)
            expanded = self.hybrid_retriever.expand_with_graph(candidates)
            reranked = self.reranker.rerank(query, expanded, top_k=30)

            if not reranked:
                return []

            config = types.GenerateContentConfig(temperature=0.2)

            # 2. Advocate call
            advocate_resp = self.client.models.generate_content(
                model=self.model_name,
                contents=self._build_advocate_prompt(query, reranked),
                config=config,
            ).text

            # 3. Devil's Advocate call
            devils_resp = self.client.models.generate_content(
                model=self.model_name,
                contents=self._build_devils_advocate_prompt(query, reranked, advocate_resp),
                config=config,
            ).text

            # 4. Arbiter call
            arbiter_resp = self.client.models.generate_content(
                model=self.model_name,
                contents=self._build_arbiter_prompt(query, reranked, advocate_resp, devils_resp),
                config=config,
            ).text

            # 5. Parse and return
            return self._parse_arbiter_response(arbiter_resp, reranked)

        except errors.APIError as e:
            logger.error(
                f"Gemini API error (Code: {e.code}): {e.message}. Falling back to thresholding."
            )
            return self.run_query(query)
        except Exception as e:
            logger.error(f"Gemini debate failed: {e}. Falling back to thresholding.")
            return self.run_query(query)

    def run_batch_with_debate(
        self, queries: list[str], delay_seconds: float = 1.0
    ) -> list[list[str]]:
        results = []
        from tqdm import tqdm

        for q in tqdm(queries, desc="Creative debate inference"):
            results.append(self.run_query_with_debate(q))
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        return results
