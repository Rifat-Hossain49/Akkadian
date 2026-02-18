import re
import unicodedata

_WS = re.compile(r"\s+")
_BRACKET_CONTENT = re.compile(r"(\[[^\]]*\]|\([^\)]*\)|\<[^\>]*\>)")
_CURLY = re.compile(r"\{([^}]*)\}")
_ANGLE_SIGNS = re.compile(r"[⸢⸣⌈⌉〈〉]")

def normalize_unicode(s: str) -> str:
    # NFKC helps normalize visually-similar forms
    return unicodedata.normalize("NFKC", s)

def clean_akkadian_transliteration(s: str) -> str:
    """Conservative cleaning:
    - keep hyphens (important in Akkadian transliteration)
    - remove bracketed editorial notes [] () <>
    - unwrap curly-brace logograms: {KUR} -> KUR
    - normalize unicode + whitespace
    """
    if s is None:
        return ""
    s = normalize_unicode(str(s))
    s = s.replace("\u200b", "")  # zero-width
    s = _ANGLE_SIGNS.sub("", s)
    s = _CURLY.sub(r"\1", s)
    s = _BRACKET_CONTENT.sub("", s)
    s = s.replace("…", "...")
    s = s.strip()
    s = _WS.sub(" ", s)
    return s

def clean_english_translation(s: str) -> str:
    """Clean target text while keeping meaning."""
    if s is None:
        return ""
    s = normalize_unicode(str(s))
    s = s.replace("\u200b", "")
    s = s.strip()
    # Remove common “damaged” placeholders but keep the rest
    s = s.replace("…", "...")
    s = _WS.sub(" ", s)
    return s

def split_pseudo_sentences(translit: str, english: str):
    """Heuristic splitting when you only have doc-level pairs.
    This is intentionally conservative: it only yields sentence pairs when
    both sides can be split into the same number of segments.
    """
    if not translit or not english:
        return []

    # split transliteration by line breaks if present (ORACC-like)
    t_parts = [p.strip() for p in re.split(r"[\n\r]+", translit) if p.strip()]

    # split English by sentence punctuation; keep short segments
    e_parts = [p.strip() for p in re.split(r"(?<=[\.!\?])\s+", english) if p.strip()]

    if 2 <= len(t_parts) <= 20 and len(t_parts) == len(e_parts):
        return list(zip(t_parts, e_parts))

    return []
