"""Law abbreviation utilities.

Loads Swiss law abbreviations from data/abbrev-translations.json.
This file contains 4362 entries including:
- 1026 non-numeric abbreviations (ZGB, OR, StGB, etc.)
- SR numbers with multilingual translations
"""

import json
from functools import lru_cache
from pathlib import Path


# Path to abbreviations JSON file
def _resolve_abbrev_file() -> Path:
    """Resolve the path to abbrev-translations.json across environments."""
    # 1. Try local project structure (relative to this file)
    local_path = (
        Path(__file__).parent.parent.parent.parent
        / "utils"
        / "abbrev-translations.json"
    )
    if local_path.exists():
        return local_path

    # 2. Try Kaggle source dataset structure
    kaggle_path = Path("/kaggle/input/omnilex-src/utils/abbrev-translations.json")
    if kaggle_path.exists():
        return kaggle_path

    # 3. Try standard project root (if running from root)
    root_path = Path("utils/abbrev-translations.json")
    if root_path.exists():
        return root_path

    # Default to local path for error message if nothing found
    return local_path


ABBREV_FILE = _resolve_abbrev_file()


@lru_cache(maxsize=1)
def load_abbreviations() -> dict[str, dict[str, str]]:
    """Load abbreviation translations from JSON.

    Returns:
        Dict mapping abbreviation id to {"de": ..., "fr": ..., "it": ...}
        e.g., {"ZGB": {"de": "ZGB", "fr": "CC", "it": "CC"}}
    """
    with open(ABBREV_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {entry["id"]: entry["abbrev"] for entry in data}


def get_german_abbreviations() -> list[str]:
    """Get list of all German law abbreviations (non-numeric only).

    Returns sorted by length (longest first) for proper regex matching.
    """
    abbrevs = load_abbreviations()
    return sorted(
        [a["de"] for a in abbrevs.values() if a.get("de") and not a["de"][0].isdigit()],
        key=len,
        reverse=True,  # Longest first for regex matching
    )


def get_all_abbreviations_mapping() -> dict[str, str]:
    """Get a mapping from all language abbreviations to their German canonical ID.

    Mapping includes de, fr, it keys pointing to the 'de' value.
    Example: {"CC": "ZGB", "ZGB": "ZGB", ...}
    """
    abbrevs = load_abbreviations()
    mapping = {}
    for entry_id, langs in abbrevs.items():
        de_val = langs.get("de")
        if not de_val:
            continue
        for lang, val in langs.items():
            if val and not val[0].isdigit():
                mapping[val] = de_val
    return mapping


def is_valid_abbreviation(abbrev: str) -> bool:
    """Check if an abbreviation is valid (exists in any language)."""
    all_abbrevs = load_abbreviations()
    # Check if it's an id or matches any language variant
    if abbrev in all_abbrevs:
        return True
    return any(
        a.get("de") == abbrev or a.get("fr") == abbrev or a.get("it") == abbrev
        for a in all_abbrevs.values()
    )


def get_abbreviation_translations(abbrev: str) -> dict[str, str] | None:
    """Get multilingual translations for an abbreviation.

    Args:
        abbrev: Abbreviation in any language

    Returns:
        Dict with de/fr/it translations, or None if not found
    """
    all_abbrevs = load_abbreviations()

    # Direct lookup by id
    if abbrev in all_abbrevs:
        return all_abbrevs[abbrev]

    # Search by language variant
    for entry in all_abbrevs.values():
        if abbrev in (entry.get("de"), entry.get("fr"), entry.get("it")):
            return entry

    return None
