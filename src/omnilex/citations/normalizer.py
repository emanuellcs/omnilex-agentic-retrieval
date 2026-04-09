"""Citation parsing and normalization for Swiss legal citations."""

import re

from .abbreviations import (
    get_all_abbreviations_mapping,
    get_german_abbreviations,
)
from .types import Citation, CitationType


class CitationNormalizer:
    """Normalize Swiss legal citations to canonical form.

    Handles two main citation types:
    1. Federal law citations - Canonical format: Art. X [Abs. Y] BOOK
       Examples: "Art. 1 ZGB", "Art. 11 Abs. 2 OR", "Art. 1 CC"
       BOOK is the law abbreviation (loaded from data/abbrev-translations.json)
       Note: Subparagraph elements (lit., Ziff., Nr., etc.) are normalized away.

    2. BGE (Bundesgerichtsentscheide) - Federal Court decision citations
       Examples: "BGE 116 Ia 56", "BGE 116 Ia 56 E. 2b"
    """

    # BGE pattern: "BGE 116 Ia 56", "BGE 116 Ia 56 E. 2b", "BGE 141 III 513 E. 5.3.1"
    # Volume: 1-3 digits, Section: Roman numerals + optional letter, Page: 1+ digits
    # Consideration formats: simple (2b), decimal (5.3.1), slash (1b/gg), range (2-4)
    BGE_PATTERN = (
        r"BGE\s+(\d{1,3})\s+"  # Volume
        r"([IVX]+[a-z]?)\s+"  # Section (Ia, II, III, etc.)
        r"(\d+)"  # Page
        r"(?:\s+(?:E\.|cons\.?|Erw\.?)\s*"  # Consideration prefix
        r"((?:[IVX]+\.)?"  # Optional Roman numeral prefix (e.g., I.)
        r"(?:\d+[a-z]?)"  # Base: digit(s) + optional letter
        r"(?:\.\d+[a-z]?)*"  # Decimal extensions (e.g., .3.1)
        r"(?:(?:/[a-z]+)|(?:-\d+[a-z]?))?))?"  # Suffix: slash+letters OR range
    )

    # Article pattern: "Art. 1", "Art 1", "Art. 1a", "Artikel 1"
    # Note: Handles no-space like "Art.117a". We use negative lookahead (?! [a-z])
    # to avoid capturing 'Art. 1' in 'Art. 1abc' but allow 'Art. 1ZGB'
    ARTICLE_PATTERN = r"(?:Art\.?|Artikel)\s*(\d+[a-z]?)(?![a-z])"

    # Paragraph pattern: "Abs. 2", "Absatz 2", "al. 2" (French), "cpv. 2" (Italian)
    PARAGRAPH_PATTERN = r"(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*(\d+[a-z]?)"

    def __init__(self):
        """Initialize normalizer with abbreviations from JSON file."""
        # Load all abbreviations and their mapping to German canonical form
        self._abbreviation_map = get_all_abbreviations_mapping()
        # Sorted by length (longest first) for proper regex matching
        self._all_abbrevs = sorted(self._abbreviation_map.keys(), key=len, reverse=True)

    def normalize(self, raw_citation: str) -> Citation | None:
        """Parse and normalize a raw citation string.

        Args:
            raw_citation: Raw citation text

        Returns:
            Normalized Citation object or None if unparseable
        """
        if not raw_citation or not raw_citation.strip():
            return None

        raw_citation = raw_citation.strip()

        # Try BGE pattern first (more specific)
        bge_match = re.search(self.BGE_PATTERN, raw_citation, re.IGNORECASE)
        if bge_match:
            return self._parse_bge(raw_citation, bge_match)

        # Try law abbreviation patterns (e.g., "Art. 1 ZGB" or "Art. 1 CC")
        # Check against all known abbreviations (de, fr, it)
        for abbrev in self._all_abbrevs:
            if abbrev in raw_citation:
                # Use the German canonical abbreviation
                de_abbrev = self._abbreviation_map[abbrev]
                return self._parse_law_abbrev(raw_citation, de_abbrev)

        return None

    def _parse_bge(self, raw_citation: str, match: re.Match) -> Citation:
        """Parse a BGE (court decision) citation."""
        volume, section, page, consideration = match.groups()

        # Build canonical ID
        canonical = f"BGE {volume} {section} {page}"
        if consideration:
            canonical += f" E. {consideration}"

        return Citation(
            raw_text=raw_citation,
            citation_type=CitationType.COURT_DECISION,
            canonical_id=canonical,
            volume=int(volume),
            section=section,
            page=int(page),
            consideration=consideration,
        )

    def _parse_law_abbrev(self, raw_citation: str, abbrev: str) -> Citation:
        """Parse a citation using law abbreviation (e.g., 'Art. 1 ZGB').

        Outputs canonical format: Art. X [Abs. Y] BOOK
        where BOOK is the law abbreviation (ZGB, OR, StGB, etc.).
        Subparagraph elements (lit., Ziff., etc.) are normalized away.
        """
        book = abbrev  # Use the abbreviation as book identifier

        # Extract article if present
        article = None
        art_match = re.search(self.ARTICLE_PATTERN, raw_citation, re.IGNORECASE)
        if art_match:
            article = f"Art. {art_match.group(1).strip()}"

        # Extract paragraph if present
        paragraph = None
        para_match = re.search(self.PARAGRAPH_PATTERN, raw_citation, re.IGNORECASE)
        if para_match:
            paragraph = f"Abs. {para_match.group(1)}"

        # Build canonical ID: Art. X [Abs. Y] BOOK
        parts = []
        if article:
            parts.append(article)
        if paragraph:
            parts.append(paragraph)
        parts.append(book)
        canonical = " ".join(parts)

        return Citation(
            raw_text=raw_citation,
            citation_type=CitationType.FEDERAL_LAW,
            canonical_id=canonical,
            book=book,
            article=article,
            paragraph=paragraph,
        )

    def canonicalize(self, raw_citation: str) -> str | None:
        """Return just the canonical ID for a citation.

        Args:
            raw_citation: Raw citation text

        Returns:
            Canonical citation ID string or None if unparseable
        """
        citation = self.normalize(raw_citation)
        return citation.canonical_id if citation else None

    def canonicalize_list(self, citations: list[str]) -> list[str]:
        """Normalize a list of citations to canonical IDs.

        Args:
            citations: List of raw citation strings

        Returns:
            List of unique canonical citation IDs (deduplicated)
        """
        result = []
        seen = set()

        for c in citations:
            canonical = self.canonicalize(c)
            if canonical and canonical not in seen:
                result.append(canonical)
                seen.add(canonical)

        return result

    def are_equivalent(self, citation1: str, citation2: str) -> bool:
        """Check if two citations refer to the same source.

        Args:
            citation1: First citation string
            citation2: Second citation string

        Returns:
            True if citations are equivalent after normalization
        """
        c1 = self.canonicalize(citation1)
        c2 = self.canonicalize(citation2)

        if c1 is None or c2 is None:
            return False

        return c1 == c2
