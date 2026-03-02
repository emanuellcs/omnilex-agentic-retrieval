"""Tests for citation normalizer."""

from omnilex.citations.normalizer import CitationNormalizer
from omnilex.citations.types import CitationType


class TestCitationNormalizer:
    """Test suite for CitationNormalizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = CitationNormalizer()

    # BGE Citation Tests

    def test_parse_bge_basic(self):
        """Test parsing basic BGE citation with consideration."""
        result = self.normalizer.normalize("BGE 116 Ia 56 E. 2b")
        assert result is not None
        assert result.citation_type == CitationType.COURT_DECISION
        assert result.canonical_id == "BGE 116 Ia 56 E. 2b"
        assert result.volume == 116
        assert result.section == "Ia"
        assert result.page == 56
        assert result.consideration == "2b"

    def test_parse_bge_with_consideration(self):
        """Test parsing BGE citation with consideration."""
        result = self.normalizer.normalize("BGE 116 Ia 56 E. 2b")
        assert result is not None
        assert result.canonical_id == "BGE 116 Ia 56 E. 2b"
        assert result.consideration == "2b"

    def test_parse_bge_section_ii(self):
        """Test parsing BGE with section II."""
        result = self.normalizer.normalize("BGE 119 II 449 E. 3.4")
        assert result is not None
        assert result.section == "II"
        assert result.page == 449
        assert result.consideration == "3.4"

    def test_parse_bge_section_iii(self):
        """Test parsing BGE with section III."""
        result = self.normalizer.normalize("BGE 121 III 38 E. 2b")
        assert result is not None
        assert result.section == "III"
        assert result.consideration == "2b"

    def test_parse_bge_with_cons(self):
        """Test parsing BGE with 'cons.' format."""
        result = self.normalizer.normalize("BGE 116 Ia 56 cons. 2b")
        assert result is not None
        assert "E. 2b" in result.canonical_id

    def test_parse_bge_decimal_consideration(self):
        """Test parsing BGE with decimal consideration number (e.g., 5.3.1)."""
        result = self.normalizer.normalize("BGE 141 III 513 E. 5.3.1")
        assert result is not None
        assert result.consideration == "5.3.1"
        assert result.canonical_id == "BGE 141 III 513 E. 5.3.1"

    def test_parse_bge_deep_decimal_consideration(self):
        """Test parsing BGE with deep decimal consideration (e.g., 2.6.2)."""
        result = self.normalizer.normalize("BGE 136 IV 1 E. 2.6.2")
        assert result is not None
        assert result.consideration == "2.6.2"
        assert result.canonical_id == "BGE 136 IV 1 E. 2.6.2"

    def test_parse_bge_simple_decimal(self):
        """Test parsing BGE with simple decimal consideration (e.g., 3.2)."""
        result = self.normalizer.normalize("BGE 132 III 668 E. 3.2")
        assert result is not None
        assert result.consideration == "3.2"

    def test_parse_bge_slash_letters_consideration(self):
        """Test parsing BGE with slash and letters in consideration (e.g., 1b/gg)."""
        result = self.normalizer.normalize("BGE 117 II 432 E. 1b/gg")
        assert result is not None
        assert result.consideration == "1b/gg"
        assert result.canonical_id == "BGE 117 II 432 E. 1b/gg"

    def test_parse_bge_range_consideration(self):
        """Test parsing BGE with range consideration (e.g., 2-4)."""
        result = self.normalizer.normalize("BGE 137 IV 13 E. 2-4")
        assert result is not None
        assert result.consideration == "2-4"
        assert result.canonical_id == "BGE 137 IV 13 E. 2-4"

    # Law Abbreviation Tests

    def test_parse_zgb_abbreviation(self):
        """Test parsing ZGB (Civil Code) abbreviation."""
        result = self.normalizer.normalize("Art. 1 ZGB")
        assert result is not None
        assert result.book == "ZGB"
        assert result.article == "Art. 1"
        assert result.canonical_id == "Art. 1 ZGB"

    def test_parse_or_abbreviation(self):
        """Test parsing OR (Code of Obligations) abbreviation."""
        result = self.normalizer.normalize("Art. 41 OR")
        assert result is not None
        assert result.book == "OR"
        assert result.article == "Art. 41"
        assert result.canonical_id == "Art. 41 OR"

    def test_parse_stgb_abbreviation(self):
        """Test parsing StGB (Criminal Code) abbreviation."""
        result = self.normalizer.normalize("Art. 117 StGB")
        assert result is not None
        assert result.book == "StGB"
        assert result.canonical_id == "Art. 117 StGB"

    def test_parse_with_paragraph(self):
        """Test parsing citation with article and paragraph."""
        result = self.normalizer.normalize("Art. 11 Abs. 2 OR")
        assert result is not None
        assert result.canonical_id == "Art. 11 Abs. 2 OR"
        assert result.book == "OR"
        assert result.article == "Art. 11"
        assert result.paragraph == "Abs. 2"

    def test_parse_with_littera_normalized_away(self):
        """Test that littera is normalized away (only paragraph level kept)."""
        result = self.normalizer.normalize("Art. 117 lit. a StGB")
        assert result is not None
        # Littera is stripped, only article and book remain
        assert result.canonical_id == "Art. 117 StGB"

    # Edge Cases

    def test_invalid_citation_returns_none(self):
        """Test that invalid citation returns None."""
        result = self.normalizer.normalize("This is not a citation")
        assert result is None

    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = self.normalizer.normalize("")
        assert result is None

    def test_none_input_returns_none(self):
        """Test that None input returns None."""
        result = self.normalizer.normalize(None)  # type: ignore
        assert result is None

    def test_whitespace_handling(self):
        """Test handling of extra whitespace in law citations."""
        result = self.normalizer.normalize("  Art.  1  ZGB  ")
        assert result is not None
        assert result.book == "ZGB"

    # Canonicalize List Tests

    def test_canonicalize_list(self):
        """Test canonicalizing a list of citations."""
        citations = [
            "Art. 1 ZGB",
            "BGE 116 Ia 56 E. 2b",
            "Art. 41 OR",
        ]
        result = self.normalizer.canonicalize_list(citations)

        assert len(result) == 3
        assert "Art. 1 ZGB" in result
        assert "BGE 116 Ia 56 E. 2b" in result
        assert "Art. 41 OR" in result

    def test_canonicalize_list_deduplication(self):
        """Test that canonicalize_list removes duplicates."""
        citations = [
            "Art. 1 ZGB",
            "Art. 1 ZGB",  # Duplicate
        ]
        result = self.normalizer.canonicalize_list(citations)

        # Should only have one unique citation
        assert len(result) == 1
        assert result[0] == "Art. 1 ZGB"

    def test_canonicalize_list_filters_invalid(self):
        """Test that canonicalize_list filters invalid citations."""
        citations = [
            "Art. 1 ZGB",
            "invalid citation",
            "BGE 116 Ia 56 E. 2b",
        ]
        result = self.normalizer.canonicalize_list(citations)

        assert len(result) == 2
        assert "invalid" not in str(result)

    # Equivalence Tests

    def test_are_equivalent_same_citation(self):
        """Test equivalence of same citation."""
        assert self.normalizer.are_equivalent("Art. 1 ZGB", "Art. 1 ZGB")

    def test_are_equivalent_same_bge(self):
        """Test equivalence of same BGE citation."""
        assert self.normalizer.are_equivalent(
            "BGE 116 Ia 56 E. 2b", "BGE 116 Ia 56 E. 2b"
        )

    def test_are_not_equivalent(self):
        """Test non-equivalence of different citations."""
        assert not self.normalizer.are_equivalent("Art. 1 ZGB", "Art. 1 OR")
