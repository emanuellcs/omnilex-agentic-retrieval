"""Data types for Swiss legal citations."""

from dataclasses import dataclass, field
from enum import Enum


class CitationType(Enum):
    """Types of Swiss legal citations."""

    FEDERAL_LAW = "federal_law"  # SR citations (e.g., SR 210)
    COURT_DECISION = (
        "court_decision"  # Court citations (e.g., BGE 116 Ia 56, 5A_800/2019 E. 2)
    )
    ORDINANCE = "ordinance"  # Verordnung
    TREATY = "treaty"  # International treaties
    UNKNOWN = "unknown"


@dataclass
class Citation:
    """Normalized citation representation.

    Attributes:
        raw_text: Original citation text as found in source
        citation_type: Type of legal citation (federal law, BGE, etc.)
        canonical_id: Normalized identifier for comparison

        For federal laws:
            book: Law abbreviation (e.g., "ZGB", "OR", "StGB")
            article: Article reference (e.g., "Art. 1", "Art. 123a")
            paragraph: Paragraph reference (e.g., "Abs. 2")
            sr_number: (deprecated) SR collection number - not used

        For court decisions:
            volume: Volume number for BGE decisions (e.g., 116)
            section: Section for BGE decisions (e.g., "Ia", "II", "III")
            page: Starting page number (e.g., 56)
            consideration: Consideration reference (e.g., "E. 2b")

    Note: Subparagraph elements (lit., Ziff., Nr., etc.) are normalized away.
    Citations are only stored down to paragraph level.
    """

    raw_text: str
    citation_type: CitationType
    canonical_id: str

    # Federal law fields
    book: str | None = None  # Law abbreviation (ZGB, OR, StGB, etc.)
    sr_number: str | None = None  # Deprecated - not used
    article: str | None = None
    paragraph: str | None = None

    # Court decision (BGE) fields
    volume: int | None = None
    section: str | None = None
    page: int | None = None
    consideration: str | None = None

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.canonical_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Citation):
            return False
        return self.canonical_id == other.canonical_id


@dataclass
class Query:
    """Input query for retrieval.

    Attributes:
        query_id: Unique identifier for the query
        text: Query text (typically in English)
        language: Language code of the query (default: "en")
        metadata: Additional query metadata
    """

    query_id: str
    text: str
    language: str = "en"
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalSample:
    """Training/evaluation sample with query and gold citations.

    Attributes:
        query: The input query
        gold_citations: List of ground truth citations
        metadata: Additional sample metadata (area, jurisdiction, etc.)
    """

    query: Query
    gold_citations: list[Citation]
    metadata: dict = field(default_factory=dict)
