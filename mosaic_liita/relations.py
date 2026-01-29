"""
Semantic relation resolution for MoSAIC-LiITA.

This module provides deterministic resolution of natural language relation
requests (e.g., "synonyms", "hyponyms") to SPARQL predicate IRIs and query
directions, using an enriched ontology catalog.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# Relation keyword sets
REL_SYNONYM = {"synonym", "synonyms", "sinonimo", "sinonimi", "equivalent", "equivalenti", "same meaning", "simile"}
REL_ANTONYM = {"antonym", "antonyms", "antonimo", "antonimi", "opposite", "opposti", "contrary", "contrario"}
REL_HYPONYM = {"hyponym", "hyponyms", "iponimo", "iponimi", "narrower", "more specific", "termine specifico"}
REL_HYPERNYM = {"hypernym", "hypernyms", "iperonimo", "iperonimi", "broader", "more general", "termine generico"}
REL_MERONYM = {"meronym", "meronyms", "meronimo", "meronimi", "part of", "parte di", "component", "componente"}
REL_HOLONYM = {"holonym", "holonyms", "olonimo", "olonimi", "whole", "insieme", "comprende", "contains"}


def resolve_relation(nl: str, catalog: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    """
    Deterministically resolve an NL relation request (e.g., "synonyms", "hyponyms", "antonyms", "part of")
    to a (predicate IRI, direction) pair, using your enriched ontology catalog.

    Returns a dict like:
      {
        "predicate_iri": "lexinfo:hyponym",
        "predicate_uri": "http://www.lexinfo.net/ontology/3.0/lexinfo#hyponym",
        "direction": "rel_to_seed",
        "confidence": 0.83,
        "candidates": [ ... top candidates with scores ... ]
      }

    Direction meaning (for your generic semantic relation block):
      - "seed_to_rel":   ?seedSense  P  ?senseRel .
      - "rel_to_seed":   ?senseRel   P  ?seedSense .
      - "both":          UNION of both directions (useful for symmetric-ish relations, e.g. synonym).

    Notes:
    - This function uses only local signals in `catalog` (label/keywords/query_hints/searchable_text + occurrence_count).
    - It does not call any LLM.
    - It prefers LexInfo relations when present, but will also accept any catalog property with semantic_category="semantic_relation".
    """
    q = re.sub(r"\s+", " ", nl.strip().lower())

    # --- 1) Determine the requested relation "intent" + preferred direction
    def detect_direction() -> str:
        if re.search(r"\bis a(n)?\s+hyponym of\b", q) or re.search(r"\bè un\s+iponimo di\b", q):
            return "rel_to_seed"
        if re.search(r"\bis a(n)?\s+hypernym of\b", q) or re.search(r"\bè un\s+iperonimo di\b", q):
            return "rel_to_seed"
        # default direction for "X of SEED"
        return "seed_to_rel"

    direction_pref = detect_direction()

    rel_kind: Optional[str] = None
    if any(k in q for k in REL_SYNONYM):
        rel_kind = "synonym"
    elif any(k in q for k in REL_ANTONYM):
        rel_kind = "antonym"
    elif any(k in q for k in REL_HYPONYM):
        rel_kind = "hyponym"
    elif any(k in q for k in REL_HYPERNYM):
        rel_kind = "hypernym"
    elif any(k in q for k in REL_MERONYM):
        rel_kind = "partMeronym"
    elif any(k in q for k in REL_HOLONYM):
        rel_kind = "partHolonym"

    # If nothing matched, fall back to a generic "related" relation if present
    if rel_kind is None:
        rel_kind = "related"

    # --- 2) Build query tokens for matching catalog entries
    token_sets = {
        "synonym": list(REL_SYNONYM) + ["approximate synonym", "near synonym", "quasi-synonym"],
        "antonym": list(REL_ANTONYM),
        "hyponym": list(REL_HYPONYM),
        "hypernym": list(REL_HYPERNYM),
        "partMeronym": list(REL_MERONYM),
        "partHolonym": list(REL_HOLONYM),
        "related": ["relates", "related", "relation", "semantic relation", "relazione", "collegato"],
    }
    tokens = [t for t in token_sets.get(rel_kind, []) if t]

    # --- 3) Candidate selection: semantic_relation properties only (preferred)
    def is_semrel_property(item: Dict[str, Any]) -> bool:
        if item.get("type") != "property":
            return False
        md = item.get("metadata", {})
        semcat = (md.get("enrichment", {}) or {}).get("semantic_category")
        if semcat == "semantic_relation":
            return True
        doms = md.get("domains", []) or []
        rngs = md.get("ranges", []) or []
        return ("http://www.w3.org/ns/lemon/ontolex#LexicalSense" in doms and
                "http://www.w3.org/ns/lemon/ontolex#LexicalSense" in rngs)

    candidates = [it for it in catalog if is_semrel_property(it)]

    # --- 4) Scoring
    def score_item(item: Dict[str, Any]) -> float:
        md = item.get("metadata", {})
        label = (md.get("label") or "").lower()
        uri = (md.get("uri") or "").lower()

        enr = md.get("enrichment", {}) or {}
        kws_en = [k.lower() for k in (enr.get("keywords_en") or [])]
        kws_it = [k.lower() for k in (enr.get("keywords_it") or [])]
        hints = [h.lower() for h in (enr.get("query_hints") or [])]

        searchable = (item.get("searchable_text") or "").lower()
        short_text = (item.get("short_text") or "").lower()

        occ = md.get("occurrence_count") or 0

        # token matches
        token_hits = 0
        for t in tokens:
            t = t.lower()
            if t in label:
                token_hits += 4
            if t in uri:
                token_hits += 2
            if t in searchable:
                token_hits += 2
            if t in short_text:
                token_hits += 1
            if t in kws_en or t in kws_it:
                token_hits += 3
            if t in hints:
                token_hits += 2

        # structural boosts: lexinfo namespace tends to be stable
        ns_boost = 0.0
        if "lexinfo.net/ontology/3.0/lexinfo#" in uri:
            ns_boost += 1.5

        # frequency boost (log-scaled)
        freq_boost = 0.0
        if occ > 0:
            freq_boost = min(3.0, (len(str(int(occ))) - 1) * 0.6)

        return token_hits + ns_boost + freq_boost

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in candidates:
        s = score_item(it)
        if s > 0:
            scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)

    # If nothing scored, fall back to any property with "relat" in label/searchable_text
    if not scored:
        for it in catalog:
            if it.get("type") == "property" and "relat" in (it.get("searchable_text") or "").lower():
                scored.append((1.0, it))
        scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:top_k] if scored else []

    # --- 5) Choose best candidate + compute direction
    chosen = top[0][1] if top else None

    symmetric_pref = False
    if rel_kind in ("synonym", "related"):
        symmetric_pref = True

    # If we have multiple candidates with close scores and synonym/related, prefer the most frequent one.
    if symmetric_pref and len(top) >= 2:
        s0, it0 = top[0]
        s1, it1 = top[1]
        if abs(s0 - s1) < 1.0:
            occ0 = (it0.get("metadata", {}) or {}).get("occurrence_count") or 0
            occ1 = (it1.get("metadata", {}) or {}).get("occurrence_count") or 0
            if occ1 > occ0:
                chosen = it1

    if not chosen:
        return {
            "predicate_iri": None,
            "predicate_uri": None,
            "direction": direction_pref,
            "confidence": 0.0,
            "candidates": [],
            "relation_kind": rel_kind,
        }

    md = chosen.get("metadata", {}) or {}
    uri = md.get("uri")
    label = md.get("label", "")

    # Build a compact prefixed name if it's LexInfo; otherwise return <IRI>
    predicate_iri = f"<{uri}>"
    if uri and uri.startswith("http://www.lexinfo.net/ontology/3.0/lexinfo#"):
        predicate_iri = "lexinfo:" + uri.split("#", 1)[1]

    # Direction refinements based on relation semantics
    # Given: A lexinfo:hyponym B means "A is more specific than B" (hyponym is subproperty of narrower)
    # - hyponym: rel -> seed (find ?senseRel where ?senseRel is more specific than seed)
    # - hypernym: seed -> rel (find ?senseRel where seed is more specific than ?senseRel, i.e., ?senseRel is more general)
    # - synonym/antonym: both directions (symmetric relations)
    # - meronym: rel -> seed (find parts OF the seed)
    # - holonym: rel -> seed (find wholes that CONTAIN the seed)
    if rel_kind == "hyponym":
        # To find hyponyms of seed: ?senseRel lexinfo:hyponym ?seedSense
        direction = "rel_to_seed"
    elif rel_kind == "hypernym":
        if uri and uri.endswith("#hyponym"):
            # Using hyponym predicate to find hypernyms: ?seedSense lexinfo:hyponym ?senseRel
            direction = "seed_to_rel"
        else:
            direction = direction_pref
    elif rel_kind in ("partMeronym", "partHolonym"):
        # Meronyms/holonyms: the related sense is the subject
        # ?senseRel partMeronym ?seedSense (rel is part of seed)
        # ?senseRel partHolonym ?seedSense (rel contains seed)
        direction = "rel_to_seed"
    else:
        direction = direction_pref

    # For synonym-ish relations, return "both" (caller can UNION both dirs)
    if rel_kind in ("synonym", "antonym"):
        direction = "both"

    # Confidence heuristic from score distribution
    best_score = top[0][0]
    second = top[1][0] if len(top) > 1 else 0.0
    confidence = 0.5 if best_score <= 2 else 0.7
    if best_score - second >= 2:
        confidence = min(0.95, confidence + 0.15)

    # Prepare candidate summary
    cand_out = []
    for s, it in top:
        md2 = it.get("metadata", {}) or {}
        cand_out.append({
            "score": s,
            "uri": md2.get("uri"),
            "label": md2.get("label"),
            "occurrence_count": md2.get("occurrence_count"),
        })

    return {
        "relation_kind": rel_kind,
        "predicate_iri": predicate_iri,
        "predicate_uri": uri,
        "predicate_label": label,
        "direction": direction,
        "confidence": confidence,
        "candidates": cand_out,
    }
