"""
Utility functions for MoSAIC-LiITA.

This module contains pure helper functions for:
- String normalization and quoting
- Pattern extraction from natural language
- Part-of-speech mapping
- Relation seed extraction
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


def norm(nl: str) -> str:
    """Normalize natural language input: lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", nl.strip().lower())


def extract_quoted_strings(nl_norm: str) -> List[str]:
    """Extract all quoted strings (single or double quotes) from normalized text."""
    return re.findall(r"[\"']([^\"']+)[\"']", nl_norm)


def contains_any(nl_norm: str, keywords: Iterable[str]) -> bool:
    """Check if any of the keywords appear in the normalized text."""
    return any(k in nl_norm for k in keywords)


def sparql_quote(s: str) -> str:
    """Safely quote a string for use in SPARQL."""
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def build_rel_triple(rel_pred: str, rel_dir: str) -> str:
    """Build a SPARQL triple pattern for a semantic relation."""
    if rel_dir == "seed_to_rel":
        return f"?seedSense {rel_pred} ?senseRel ."
    if rel_dir == "rel_to_seed":
        return f"?senseRel {rel_pred} ?seedSense ."
    raise ValueError("rel_dir must be seed_to_rel|rel_to_seed")


def extract_relation_seed(q: str, quoted: List[str]) -> Optional[str]:
    """
    Extract the seed word for a semantic relation query.

    Examples:
    - "synonyms of 'antico'" -> "antico"
    - "hyponyms of sentimento" -> "sentimento"
    """
    if quoted:
        # Try to capture quoted seed after of/di
        m = re.search(r"(?:of|di)\s+['\"]([^'\"]+)['\"]", q)
        if m:
            return m.group(1).strip()
        # fallback: first quoted
        return quoted[0].strip() if quoted[0].strip() else None

    m = re.search(r"(hyponyms|hypernyms|synonyms|antonyms|meronyms|holonyms)\s+of\s+([^\s\.,;:]+)", q)
    if m:
        return m.group(2)

    m = re.search(r"(iponim[io]|iperonim[io]|sinonim[io]|antonim[io]|meronim[io]|olonim[io])\s+di\s+([^\s\.,;:]+)", q)
    if m:
        return m.group(2)

    return None


def extract_pattern_request(nl: str) -> Optional[Dict[str, Any]]:
    """
    Extract a simple pattern request from natural language.

    Supports:
    - prefix:  "starting with X", "starts with X", "inizia con X"
    - suffix:  "ending with X", "ends with X", "finisce con X"
              also common Italian morphology: "ending in -ire", "ending in 'ire'"
    - contains: "contains X", "containing X", "che contiene X", "contenente X"

    Returns:
        dict with "mode" ("prefix"|"suffix"|"contains") and "text",
        or None if not found.
    """
    q = re.sub(r"\s+", " ", nl.strip().lower())

    # 1) quoted text after pattern keywords
    # prefix
    m = re.search(r"(starting with|starts with|inizia con)\s+['\"]([^'\"]+)['\"]", q)
    if m:
        return {"mode": "prefix", "text": m.group(2).strip()}

    # suffix (also allow -ire)
    m = re.search(r"(ending with|ends with|finisce con)\s+['\"]([^'\"]+)['\"]", q)
    if m:
        return {"mode": "suffix", "text": m.group(2).strip()}

    m = re.search(r"(ending in|ends in)\s+[-'\" ]*([a-zàèéìòù]+)\b", q)
    if m:
        return {"mode": "suffix", "text": m.group(2).strip()}

    # contains
    m = re.search(r"(contains|containing|che contiene|contenente)\s+['\"]([^'\"]+)['\"]", q)
    if m:
        return {"mode": "contains", "text": m.group(2).strip()}

    # 2) unquoted single-token fallback
    m = re.search(r"(starting with|starts with|inizia con)\s+([^\s\.,;:]+)", q)
    if m:
        return {"mode": "prefix", "text": m.group(2).strip()}

    m = re.search(r"(ending with|ends with|finisce con)\s+([^\s\.,;:]+)", q)
    if m:
        return {"mode": "suffix", "text": m.group(2).strip()}

    m = re.search(r"(contains|containing|che contiene|contenente)\s+([^\s\.,;:]+)", q)
    if m:
        return {"mode": "contains", "text": m.group(2).strip()}

    return None


def build_filter_for_var(var: str, mode: str, text: str) -> str:
    """
    Build a SPARQL FILTER clause for pattern matching.

    Args:
        var: SPARQL variable name (e.g., "?wr")
        mode: "prefix", "suffix", or "contains"
        text: The pattern text to match

    Returns:
        A SPARQL FILTER clause string
    """
    safe = text.replace("\\", "\\\\").replace('"', '\\"')

    if mode == "prefix":
        return f'FILTER(STRSTARTS(STR({var}), "{safe}")) .'
    if mode == "contains":
        return f'FILTER(CONTAINS(LCASE(STR({var})), LCASE("{safe}"))) .'
    if mode == "suffix":
        return f'FILTER(STRENDS(STR({var}), "{safe}")) .'
    return ""


def map_pos(nl: str) -> Dict[str, Optional[str]]:
    """
    Map natural language PoS mentions to:
    - LiITA PoS IRI (lila:hasPOS lila:verb etc.)
    - CompL-IT PoS label (FILTER(str(?posLabel) = "verb"))

    Returns:
        {
            "liita_pos_iri": "lila:verb" | None,
            "complit_pos_label": "\"verb\"" | None
        }
    """
    q = re.sub(r"\s+", " ", nl.strip().lower())

    # (trigger words) -> (liita IRI, compl-it label)
    mapping = [
        (["verb", "verbs", "verbo", "verbi"], "lila:verb", '"verb"'),
        (["noun", "nouns", "sostantivo", "sostantivi", "nome", "nomi"], "lila:noun", '"noun"'),
        (["adjective", "adjectives", "aggettivo", "aggettivi"], "lila:adjective", '"adjective"'),
        (["adverb", "adverbs", "avverbio", "avverbi"], "lila:adverb", '"adverb"'),
        (["proper noun", "proper nouns", "nome proprio", "nomi propri"], "lila:proper_noun", '"proper noun"'),
        (["pronoun", "pronouns", "pronome", "pronomi"], "lila:pronoun", '"pronoun"'),
        (["determiner", "determiners", "determinante", "determinanti"], "lila:determiner", '"determiner"'),
        (["preposition", "prepositions", "adposition", "adpositions", "preposizione", "preposizioni"], "lila:adposition", '"adposition"'),
        (["numeral", "numerals", "numero", "numeri"], "lila:numeral", '"numeral"'),
        (["conjunction", "conjunctions", "congiunzione", "congiunzioni"], None, '"conjunction"'),
        (["interjection", "interjections", "interiezione", "interiezioni"], "lila:interjection", '"interjection"'),
        (["particle", "particles", "particella", "particelle"], "lila:particle", '"particle"'),
    ]

    for triggers, liita_iri, complit_label in mapping:
        for t in triggers:
            if t in q:
                return {"liita_pos_iri": liita_iri, "complit_pos_label": complit_label}

    return {"liita_pos_iri": None, "complit_pos_label": None}
