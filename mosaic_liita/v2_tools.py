"""
MoSAIC v2 (I3): tools and blocks for the parts of the problem perimeter not covered by v1.

Agentic mode only. Each tool maps to one block (registered by register_tool_blocks) and, when it has
parameters, to a slot builder. Variable names follow the gold conventions where possible.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict

from .blocks import Block, BlockRegistry
from .constants import COMPLIT_SERVICE

LIITA_TYPE = "FILTER(?{v}Type IN (lila:Lemma, lila:Hypolemma))"


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _name(s: Any) -> str:
    """Bare identifier from an LLM value (e.g. 'lila:feminine' -> 'feminine', 'SIMPLE Fruit' -> 'Fruit')."""
    s = str(s or "").strip()
    s = s.rsplit(":", 1)[-1].rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_\- ]", "", s).strip()


# -------------------------------------------------------------------------------------------------- tools
V2_TOOLS: Dict[str, Dict[str, Any]] = {
    "get_lemma_features": {
        "description": "Grammatical features of a LiITA lemma: gender, degree (adjectives), inflection type "
                       "(verb conjugation class such as c1r, c3i). Give a value to FILTER on it, leave empty to return it.",
        "block": "LIITA_LEMMA_FEATURES_V2",
        "parameters": {
            "gender": {"type": "string", "description": "e.g. masculine, feminine (optional)", "required": False},
            "degree": {"type": "string", "description": "e.g. positive, comparative, superlative (optional)", "required": False},
            "inflection_type": {"type": "string", "description": "LiITA inflection type code, e.g. c1r, c2i, c3i (optional)", "required": False},
        },
        "provides": ["?gender", "?degree", "?inflectionType"],
        "requires": ["?liitaLemma"],
    },
    "get_hypolemmas": {
        "description": "Hypolemmas (participles, deadjectival adverbs...) linked to a LiITA lemma.",
        "block": "LIITA_HYPOLEMMAS_V2",
        "provides": ["?hypolemma", "?hypolemmaWR"],
        "requires": ["?liitaLemma"],
    },
    "get_lemma_variants": {
        "description": "Lemma variants recorded in LiITA for a lemma (e.g. reflexive or plural-only forms), both directions.",
        "block": "LIITA_LEMMA_VARIANTS_V2",
        "provides": ["?variantLemma", "?variantWR"],
        "requires": ["?liitaLemma"],
    },
    "find_words_by_semantic_type": {
        "description": "CompL-it words having a sense of a given SIMPLE semantic type (e.g. 'Fruit', 'Profession', "
                       "'Opening', 'Act'). Provides the word, its written form and the typed sense.",
        "block": "COMPLIT_WORDS_BY_SEMTYPE_V2",
        "parameters": {"semantic_type": {"type": "string", "description": "SIMPLE type name, e.g. Fruit", "required": True}},
        "provides": ["?word", "?itLemmaString", "?sense"],
        "requires": [],
    },
    "find_words_by_trait": {
        "description": "CompL-it words having a sense with a given trait (e.g. 'LIVESTOCK_FARMING', 'BehaviourNegative').",
        "block": "COMPLIT_WORDS_BY_TRAIT_V2",
        "parameters": {"trait": {"type": "string", "description": "Trait name or part of it", "required": True}},
        "provides": ["?word", "?itLemmaString", "?sense"],
        "requires": [],
    },
    "get_semantic_type": {
        "description": "SIMPLE semantic type of a CompL-it sense (?senseRel from find_semantic_relations or ?sense).",
        "block_variants": {"?senseRel": "COMPLIT_SEMTYPE_OF_SENSEREL_V2", "?sense": "COMPLIT_SEMTYPE_OF_SENSE_V2"},
        "parameters": {"source_var": {"type": "string", "enum": ["?senseRel", "?sense"], "description": "The sense variable", "required": True}},
        "provides": ["?semanticType"],
        "requires_one_of": ["?senseRel", "?sense"],
    },
    "get_senses": {
        "description": "All senses of a CompL-it word (?word or ?wordRel), e.g. to count senses per word.",
        "block_variants": {"?word": "COMPLIT_SENSES_OF_WORD_V2", "?wordRel": "COMPLIT_SENSES_OF_WORDREL_V2"},
        "parameters": {"source_var": {"type": "string", "enum": ["?word", "?wordRel"], "description": "The word variable", "required": True}},
        "provides": ["?anySense"],
        "requires_one_of": ["?word", "?wordRel"],
    },
    "list_lemma_banks": {
        "description": "Metadata: the lemma banks of the LiITA knowledge base (Italian, Sicilian, Parmigiano), with their lemmas.",
        "block": "LIITA_LEMMA_BANKS_V2",
        "provides": ["?lemmaBank", "?lemmaBankTitle", "?bankLemma"],
        "requires": [],
    },
    "list_emotion_categories": {
        "description": "Metadata: the emotion categories used by ELIta (with their labels).",
        "block": "ELITA_CATEGORIES_V2",
        "provides": ["?emotion", "?emotionLabel"],
        "requires": [],
    },
}

# extra relation types for find_semantic_relations: name -> (predicate, direction)
# direction "seed_to_rel": ?seedSense P ?senseRel ; "rel_to_seed": ?senseRel P ?seedSense
V2_RELATIONS = {
    "used_for": ("complit:usedFor", "seed_to_rel"),
    "created_by": ("complit:createdBy", "seed_to_rel"),
    "telic": ("complit:telic", "seed_to_rel"),
    "derived_from_adjective": ("complit:deadjectivalNoun", "seed_to_rel"),   # noun -> its adjective
    "deadjectival_noun": ("complit:deadjectivalNoun", "rel_to_seed"),        # adjective -> nouns derived from it
    "derived_from_noun": ("complit:denominalAdjective", "seed_to_rel"),      # adjective -> its noun
    "denominal_adjective": ("complit:denominalAdjective", "rel_to_seed"),    # noun -> adjectives derived from it
}


# -------------------------------------------------------------------------------------------------- slots
def _features(p: Dict[str, Any]) -> Dict[str, str]:
    """Required (filtered) features first, OPTIONAL ones after: Virtuoso returns nothing when a group starts
    with OPTIONAL patterns on a variable that is bound only by a later required pattern."""
    req, opt = [], []
    for key, var, prop, base in (("gender", "gender", "lila:hasGender", "lila:"),
                                 ("degree", "degree", "lila:hasDegree", "lila:"),
                                 ("inflection_type", "inflectionType", "lila:hasInflectionType",
                                  "<http://liita.it/ontologies/liita/")):
        val = _name(p.get(key))
        if val:
            iri = f"{base}{val}>" if base.startswith("<") else f"{base}{val}"
            req.append(f"?liitaLemma {prop} ?{var} . FILTER(?{var} = {iri})")
        else:
            opt.append(f"OPTIONAL {{ ?liitaLemma {prop} ?{var} . }}")
    return {"features_clause": " ".join(req + opt)}


V2_SLOTS: Dict[str, Callable[[Dict[str, Any]], Dict[str, str]]] = {
    "get_lemma_features": _features,
    "find_words_by_semantic_type": lambda p: {"type_filter": f"FILTER(STRENDS(LCASE(STR(?semType)), LCASE({_q('-' + _name(p.get('semantic_type')).replace(' ', '_'))})))"},
    "find_words_by_trait": lambda p: {"trait_filter": f"FILTER(CONTAINS(LCASE(STR(?trait)), LCASE({_q(_name(p.get('trait')))})))"},
}


# -------------------------------------------------------------------------------------------------- blocks
def register_tool_blocks(R: BlockRegistry) -> None:
    def reg(**kw):
        if kw["id"] not in R.blocks:
            R.register(Block(**kw))

    reg(id="LIITA_LEMMA_FEATURES_V2", requires={"?liitaLemma"}, provides={"?gender", "?degree", "?inflectionType"},
        prefixes={"lila"},
        where=["GRAPH <http://liita.it/data> {{ ?liitaLemma a ?featLemmaType . {features_clause} }}"])
    reg(id="LIITA_HYPOLEMMAS_V2", requires={"?liitaLemma"}, provides={"?hypolemma", "?hypolemmaWR"},
        prefixes={"lila", "ontolex"},
        where=["GRAPH <http://liita.it/data> {{ ?hypolemma lila:isHypolemma ?liitaLemma ; ontolex:writtenRep ?hypolemmaWR . }}"])
    reg(id="LIITA_LEMMA_VARIANTS_V2", requires={"?liitaLemma"}, provides={"?variantLemma", "?variantWR"},
        prefixes={"lila", "ontolex"},
        where=["GRAPH <http://liita.it/data> {{",
               "  {{ ?liitaLemma lila:lemmaVariant ?variantLemma }} UNION {{ ?variantLemma lila:lemmaVariant ?liitaLemma }}",
               "  ?variantLemma ontolex:writtenRep ?variantWR .", "}}"])
    reg(id="COMPLIT_WORDS_BY_SEMTYPE_V2", requires=set(), provides={"?word", "?itLemmaString", "?sense"},
        prefixes={"ontolex", "complit"}, service_iri=COMPLIT_SERVICE,
        where=["?word ontolex:sense ?sense ; ontolex:canonicalForm [ ontolex:writtenRep ?itLemmaString ] .",
               "?sense complit:hasSemanticType ?semType .", "{type_filter}"])
    reg(id="COMPLIT_WORDS_BY_TRAIT_V2", requires=set(), provides={"?word", "?itLemmaString", "?sense"},
        prefixes={"ontolex", "complit"}, service_iri=COMPLIT_SERVICE,
        where=["?word ontolex:sense ?sense ; ontolex:canonicalForm [ ontolex:writtenRep ?itLemmaString ] .",
               "?sense complit:trait ?trait .", "{trait_filter}"])
    for src, bid in (("?senseRel", "COMPLIT_SEMTYPE_OF_SENSEREL_V2"), ("?sense", "COMPLIT_SEMTYPE_OF_SENSE_V2")):
        reg(id=bid, requires={src}, provides={"?semanticType"}, prefixes={"complit"}, service_iri=COMPLIT_SERVICE,
            where=[f"{src} complit:hasSemanticType ?semanticType ."])
    for src, bid in (("?word", "COMPLIT_SENSES_OF_WORD_V2"), ("?wordRel", "COMPLIT_SENSES_OF_WORDREL_V2")):
        reg(id=bid, requires={src}, provides={"?anySense"}, prefixes={"ontolex"}, service_iri=COMPLIT_SERVICE,
            where=[f"{src} ontolex:sense ?anySense ."])
    reg(id="LIITA_LEMMA_BANKS_V2", requires=set(), provides={"?lemmaBank", "?lemmaBankTitle", "?bankLemma"},
        prefixes={"dcterms"},
        where=["?bankLemma dcterms:isPartOf ?lemmaBank .",
               "OPTIONAL {{ ?lemmaBank dcterms:title ?lemmaBankTitle . }}"])
    reg(id="ELITA_CATEGORIES_V2", requires=set(), provides={"?emotion", "?emotionLabel"},
        prefixes={"elita", "rdfs"},
        where=["?emotionEntry elita:HasEmotion ?emotion .", "?emotion rdfs:label ?emotionLabel ."])
