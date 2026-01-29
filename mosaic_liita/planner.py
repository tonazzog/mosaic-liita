"""
Query planner for MoSAIC-LiITA.

The Planner takes natural language input and produces a QuerySpec by:
1. Analyzing the query intent (semantic relations, definitions, patterns, etc.)
2. Selecting appropriate blocks from the registry
3. Filling slot values based on extracted patterns and parameters
4. Configuring output shape (SELECT, GROUP BY, aggregates)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .blocks import BlockRegistry

from .blocks import Block, BlockCall
from .query import QuerySpec
from .constants import EMOTION_MAP
from .relations import resolve_relation
from .utils import (
    norm,
    extract_quoted_strings,
    contains_any,
    sparql_quote,
    extract_relation_seed,
    extract_pattern_request,
    build_filter_for_var,
    map_pos,
)


class Planner:
    """
    Converts natural language queries to QuerySpec.

    The planner analyzes the input to determine:
    - Query type (semantic relations, definitions, pattern matching, etc.)
    - Required blocks and their slot values
    - Output variables and aggregations
    """

    def __init__(self, registry: "BlockRegistry", catalog: List[Dict[str, Any]]) -> None:
        """
        Initialize the planner.

        Args:
            registry: BlockRegistry containing available blocks
            catalog: Ontology catalog for relation resolution
        """
        self.R = registry
        self.catalog = catalog

    def plan(self, nl: str) -> QuerySpec:
        """
        Plan a query from natural language input.

        Args:
            nl: Natural language query string

        Returns:
            QuerySpec ready for compilation
        """
        q = norm(nl)
        quoted = extract_quoted_strings(q)

        # --- Detect query intent flags ---
        wants_semrel = contains_any(q, [
            "hyponym", "hyponyms", "iponim", "iponimi",
            "hypernym", "hypernyms", "iperonim", "iperonimi",
            "synonym", "synonyms", "sinonim", "sinonimi",
            "antonym", "antonyms", "antonim", "antonimi", "opposite", "opposto",
            "meronym", "meronyms", "meronim", "meronimi", "part of", "parte di",
            "holonym", "holonyms", "olonim", "olonimi", "whole"
        ])

        wants_complit_def = contains_any(q, [
            "compl-it", "compl it", "definition", "definitions", "definizione", "definizioni"
        ])

        wants_def_pattern = wants_complit_def and contains_any(q, [
            "starting with", "starts with", "inizia con",
            "ending with", "ends with", "finisce con",
            "ending in", "ends in",
            "contains", "containing", "che contiene", "contenente"
        ])

        wants_liita_pattern = (not wants_complit_def) and contains_any(q, [
            "starting with", "starts with", "inizia con",
            "ending with", "ends with", "finisce con",
            "ending in", "ends in",
            "contains", "containing", "che contiene", "contenente"
        ]) and contains_any(q, ["lemma", "lemmas", "word", "words", "parola", "parole", "verbi", "verb", "noun", "sostantivi"])

        wants_sic = contains_any(q, ["siciliano", "sicilian"])
        wants_par = contains_any(q, ["parmigiano"])
        wants_sentix = contains_any(q, ["sentix", "polarity", "positive", "negative", "neutral", "polarità", "polarita"])
        wants_elita = contains_any(q, ["elita", "emotion", "emozion", "gioia", "felicità", "felicita", "joy", "happiness"])

        wants_count = contains_any(q, [
            "count", "how many", "number of", "quantitative distribution",
            "quanti", "quante", "numero di", "conteggio", "distribuzione"
        ])
        count_by_pos = wants_count and contains_any(q, [
            "per pos", "by pos", "by part of speech", "part of speech distribution",
            "per parte del discorso", "categorie grammaticali", "parti del discorso"
        ])
        count_variants = wants_count and contains_any(q, [
            "multiple written", "more than one written", "variants", "variant", "piu forme", "varianti"
        ])

        # Dialect-first lemma pattern detection
        pat = extract_pattern_request(q)
        has_pattern = pat is not None

        dialect_first_cues = contains_any(q, [
            "lemma", "lemmas", "lemmata",
            "word", "words", "parola", "parole",
            "form", "forms",
            "ending", "ends", "starting", "starts",
            "inizia", "finisce", "che iniziano", "che finiscono",
        ])

        to_italian_cues = contains_any(q, [
            "into italian", "to italian", "in italian", "in italiano", "in italiano", "in lingua italiana",
            "traduci in italiano", "traduzione in italiano",
        ])

        wants_dialect_lemma_pattern = (
            has_pattern
            and (wants_sic or wants_par)
            and dialect_first_cues
            and not wants_semrel
            and not wants_complit_def
        )

        # PoS mapping
        pos_info = map_pos(q)
        complit_pos_label = pos_info["complit_pos_label"]
        liita_pos_iri = pos_info["liita_pos_iri"]
        pos_filter = f"FILTER(str(?posLabel) = {complit_pos_label}) ." if complit_pos_label else ""

        blocks: List[BlockCall] = []
        known_vars: Set[str] = set()

        # --- 1) CompL-IT semantic relation anchor ---
        if wants_semrel:
            seed = extract_relation_seed(q, quoted)
            if seed:
                rel = resolve_relation(nl=q, catalog=self.catalog)
                rel_pred = rel["predicate_iri"]
                rel_dir = rel["direction"]

                if rel_dir == "both":
                    rel_triple = (
                        "{ ?seedSense " + rel_pred + " ?senseRel . } UNION "
                        "{ ?senseRel " + rel_pred + " ?seedSense . }"
                    )
                elif rel_dir == "seed_to_rel":
                    rel_triple = f"?seedSense {rel_pred} ?senseRel ."
                else:
                    rel_triple = f"?senseRel {rel_pred} ?seedSense ."

                bid = "COMPLIT_SEMREL_OF_SEED_LEMMA"
                blocks.append(BlockCall(
                    block_id=bid,
                    slots={
                        "seed_lemma": sparql_quote(seed),
                        "seed_pos_filter": pos_filter,
                        "rel_triple": rel_triple,
                        "rel_extra": "",
                    },
                ))
                known_vars |= self.R.get(bid).provides

                bid = "JOIN_WORDREL_TO_LIITA"
                blocks.append(BlockCall(block_id=bid))
                known_vars |= self.R.get(bid).provides

        # --- 2) CompL-IT definition filter by pattern ---
        elif wants_def_pattern:
            pat = extract_pattern_request(q)
            if not pat and quoted:
                pat = {"mode": "prefix", "text": quoted[0]}

            def_filter = ""
            lemma_filter = ""
            if pat:
                lemma_cues = contains_any(q, [
                    "word", "words", "lemma", "lemmas", "parola", "parole"
                ])
                if lemma_cues:
                    lemma_filter = build_filter_for_var("?itLemmaString", pat["mode"], pat["text"])
                else:
                    def_filter = build_filter_for_var("?definition", pat["mode"], pat["text"])

            bid = "COMPLIT_DEF_FILTER_BY_PATTERN"
            blocks.append(BlockCall(
                block_id=bid,
                slots={
                    "pos_filter": pos_filter,
                    "def_filter": def_filter,
                    "lemma_filter": lemma_filter,
                },
            ))
            known_vars |= self.R.get(bid).provides

            if wants_sic or wants_par or wants_sentix or wants_elita:
                bid = "JOIN_WORD_TO_LIITA_FROM_WORD"
                blocks.append(BlockCall(block_id=bid))
                known_vars |= self.R.get(bid).provides

        # --- 3) Dialect-first lemma filter by pattern ---
        elif wants_dialect_lemma_pattern:
            if wants_sic:
                block_id = "SICILIANO_LEMMA_FILTER_BY_PATTERN_AND_POS"
                wr_var = "?sicWR"
                lemma_var = "?sicLemma"
            else:
                block_id = "PARMIGIANO_LEMMA_FILTER_BY_PATTERN_AND_POS"
                wr_var = "?parWR"
                lemma_var = "?parLemma"

            wr_filter = build_filter_for_var(wr_var, pat["mode"], pat["text"])

            pos_clause = ""
            if liita_pos_iri:
                pos_clause = f"{lemma_var} lila:hasPOS {liita_pos_iri} ."

            bid = block_id
            blocks.append(BlockCall(
                block_id=bid,
                slots={
                    "pos_clause": pos_clause,
                    "wr_filter": wr_filter,
                },
            ))
            known_vars |= self.R.get(bid).provides

            if to_italian_cues:
                if wants_sic:
                    bid = "TRANSLATE_FROM_SICILIANO"
                    blocks.append(BlockCall(block_id=bid))
                    known_vars |= self.R.get(bid).provides

                if wants_par:
                    bid = "TRANSLATE_FROM_PARMIGIANO"
                    blocks.append(BlockCall(block_id=bid))
                    known_vars |= self.R.get(bid).provides

        # --- 4) Local LiITA lemma filter by pattern ---
        elif wants_liita_pattern:
            pat = extract_pattern_request(q)
            wr_filter = ""
            if pat:
                wr_filter = build_filter_for_var("?wr", pat["mode"], pat["text"])

            pos_clause = ""
            if liita_pos_iri:
                pos_clause = f"?lemma lila:hasPOS {liita_pos_iri} ."

            bid = "LIITA_LEMMA_FILTER_BY_PATTERN_AND_POS"
            blocks.append(BlockCall(
                block_id=bid,
                slots={
                    "pos_clause": pos_clause,
                    "wr_filter": wr_filter,
                },
            ))
            known_vars |= self.R.get(bid).provides

            # Add BIND block for downstream enrichments
            if wants_sentix or wants_elita or wants_sic or wants_par:
                bind_block = Block(
                    id="BIND_LEMMA_TO_LIITALEMMA",
                    requires={"?lemma"},
                    provides={"?liitaLemma"},
                    prefixes=set(),
                    where=["BIND(?lemma AS ?liitaLemma)"],
                )
                if bind_block.id not in self.R.blocks:
                    self.R.blocks[bind_block.id] = bind_block
                blocks.append(BlockCall(block_id=bind_block.id))
                known_vars |= bind_block.provides

        # --- 5) Fallback: LiITA by writtenRep or list ---
        else:
            if quoted:
                word = quoted[0]
                b = Block(
                    id="LIITA_BY_WRITTENREP",
                    requires=set(),
                    provides={"?liitaLemma", "?itLemmaString"},
                    prefixes={"lila", "ontolex"},
                    where=[
                        "?liitaLemma a lila:Lemma .",
                        "?liitaLemma ontolex:writtenRep ?itLemmaString .",
                        f'FILTER(str(?itLemmaString) = {sparql_quote(word)}) .',
                    ],
                )
                if b.id not in self.R.blocks:
                    self.R.blocks[b.id] = b

                blocks.append(BlockCall(block_id=b.id))
                known_vars |= b.provides
            else:
                b = Block(
                    id="LIITA_LIST_LEMMAS",
                    requires=set(),
                    provides={"?liitaLemma"},
                    prefixes={"lila"},
                    where=["?liitaLemma a lila:Lemma ."],
                )
                if b.id not in self.R.blocks:
                    self.R.blocks[b.id] = b

                blocks.append(BlockCall(block_id=b.id))
                known_vars |= b.provides

        # --- Attach optional enrichments ---
        if "?liitaLemma" in known_vars:
            bid = "ASSERT_LIITA_LEMMA_TYPE"
            blocks.append(BlockCall(block_id=bid))

            if wants_elita:
                emotion_terms = self._extract_emotion_terms(q, quoted)
                emotion_filter_clause = self._make_emotion_filter_clause(emotion_terms)

                bid = "ELITA_EMOTION_FILTER"
                blocks.append(BlockCall(
                    block_id=bid,
                    slots={"emotion_filter_clause": emotion_filter_clause},
                ))
                known_vars |= self.R.get(bid).provides

            if wants_sentix:
                bid = "SENTIX_POLARITY"
                blocks.append(BlockCall(block_id=bid))
                known_vars |= self.R.get(bid).provides

            if wants_sic and not (wants_dialect_lemma_pattern and to_italian_cues):
                bid = "TRANSLATE_TO_SICILIANO"
                blocks.append(BlockCall(block_id=bid))
                known_vars |= self.R.get(bid).provides

            if wants_par and not (wants_dialect_lemma_pattern and to_italian_cues):
                bid = "TRANSLATE_TO_PARMIGIANO"
                blocks.append(BlockCall(block_id=bid))
                known_vars |= self.R.get(bid).provides

        # Ensure POS variable when counting by POS
        if count_by_pos and "?pos" not in known_vars:
            if "?lemma" in known_vars:
                bid = "LIITA_LEMMA_POS"
                blocks.append(BlockCall(block_id=bid))
                known_vars |= self.R.get(bid).provides
            elif "?liitaLemma" in known_vars:
                bid = "LIITA_LEMMA_POS_FROM_LIITA"
                blocks.append(BlockCall(block_id=bid))
                known_vars |= self.R.get(bid).provides

        # --- Build SELECT, aggregates, GROUP BY ---
        select_vars: List[str] = []
        aggregates: Dict[str, str] = {}
        group_by: List[str] = []
        having: Optional[str] = None
        order_by: Optional[str] = None

        def pick_count_var() -> Optional[str]:
            for v in ["?liitaLemma", "?lemma", "?word", "?sicLemma", "?parLemma"]:
                if v in known_vars:
                    return v
            return None

        if wants_count:
            count_var = pick_count_var() or "?s"
            aggregates["?count"] = f"COUNT(DISTINCT {count_var})"
            if count_by_pos and "?pos" in known_vars:
                select_vars = ["?pos"]
                group_by = ["?pos"]
                order_by = "ORDER BY DESC(?count)"
            else:
                select_vars = []
                group_by = []
                order_by = None

            if count_variants and "?lemma" in known_vars and "?wr" in known_vars:
                aggregates = {"?variantCount": "COUNT(?wr)"}
                select_vars = ["?lemma"]
                group_by = ["?lemma"]
                having = "HAVING (COUNT(?wr) > 1)"

        if not wants_count:
            if "?lemma" in known_vars:
                select_vars.append("?lemma")
                group_by.append("?lemma")
            if "?wr" in known_vars:
                select_vars.append("?wr")
                group_by.append("?wr")

            if "?liitaLemma" in known_vars:
                select_vars.append("?liitaLemma")
                group_by.append("?liitaLemma")

            if "?itLemmaString" in known_vars:
                select_vars.append("?itLemmaString")
                group_by.append("?itLemmaString")

            if "?definition" in known_vars:
                aggregates["?definitionSample"] = "SAMPLE(?definition)"

            if "?emotionLabel" in known_vars:
                aggregates["?emotions"] = 'GROUP_CONCAT(DISTINCT ?emotionLabel; SEPARATOR=", ")'

            if "?polarityLabel" in known_vars:
                aggregates["?polarityLabels"] = 'GROUP_CONCAT(DISTINCT ?polarityLabel; SEPARATOR=", ")'
            if "?polarityValue" in known_vars:
                aggregates["?polarityValues"] = 'GROUP_CONCAT(DISTINCT ?polarityValue; SEPARATOR=", ")'

            if "?liitaLemma" in known_vars and not wants_dialect_lemma_pattern:
                if "?sicWR" in known_vars:
                    aggregates["?sicilianoWRs"] = 'GROUP_CONCAT(DISTINCT ?sicWR; SEPARATOR=", ")'
                if "?parWR" in known_vars:
                    aggregates["?parmigianoWRs"] = 'GROUP_CONCAT(DISTINCT ?parWR; SEPARATOR=", ")'

            if "?sicLemma" in known_vars:
                select_vars += ["?sicLemma", "?sicWR"]
                group_by += ["?sicLemma", "?sicWR"]

            if "?parLemma" in known_vars:
                select_vars += ["?parLemma", "?parWR"]
                group_by += ["?parLemma", "?parWR"]

            group_by = list(dict.fromkeys(group_by))
            order_by = "ORDER BY ?itLemmaString" if "?itLemmaString" in known_vars else None

            if aggregates and not group_by:
                group_by = [v for v in select_vars if v.startswith("?")]

            if aggregates:
                group_by = [v for v in group_by if v in select_vars]
                group_by = list(dict.fromkeys(group_by))

        return QuerySpec(
            blocks=blocks,
            select_vars=select_vars,
            aggregates=aggregates,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=None if wants_count else 200,
        )

    def _extract_emotion_terms(self, q: str, quoted: List[str]) -> List[str]:
        """Extract emotion terms from query."""
        terms = []
        for s in quoted:
            if s.strip():
                terms.append(s.strip().lower())
        for k in ["gioia", "felicità", "felicita", "joy", "happiness"]:
            if re.search(rf"\b{re.escape(k)}\b", q):
                terms.append(k)
        return list(dict.fromkeys(terms))

    def _make_emotion_filter_clause(self, emotion_terms: List[str]) -> str:
        """Build SPARQL filter clause for emotions."""
        iris = []
        label_terms = []
        for t in emotion_terms:
            t_norm = t.lower()
            if t_norm in EMOTION_MAP:
                iris.append(EMOTION_MAP[t_norm])
            else:
                label_terms.append(t_norm)

        if iris and not label_terms:
            values = " ".join(iris)
            return f"VALUES ?emotion {{ {values} }}"

        if emotion_terms:
            items = ", ".join(sparql_quote(t.replace("'", "'")) for t in emotion_terms)
            return f"FILTER(LCASE(STR(?emotionLabel)) IN ({items}))"
        return ""
