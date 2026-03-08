"""
Constants and configuration for MoSAIC-LiITA.

This module contains all static configuration values used across the package:
- RDF namespace prefixes
- Service URIs
- Dialect lemma bank IRIs
- Emotion mappings
- Relation keyword mappings
"""

from typing import Dict, List, Tuple, Set

# ---------------------------
# RDF Prefixes
# ---------------------------
PREFIXES: Dict[str, str] = {
    "lila": "http://lila-erc.eu/ontologies/lila/",
    "ontolex": "http://www.w3.org/ns/lemon/ontolex#",
    "vartrans": "http://www.w3.org/ns/lemon/vartrans#",
    "lime": "http://www.w3.org/ns/lemon/lime#",
    "lexinfo": "http://www.lexinfo.net/ontology/3.0/lexinfo#",
    "marl": "http://www.gsi.upm.es/ontologies/marl/ns#",
    "elita": "http://w3id.org/elita/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "dcterms": "http://purl.org/dc/terms/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

# ---------------------------
# Service URIs
# ---------------------------
COMPLIT_SERVICE = "https://klab.ilc.cnr.it/graphdb-compl-it/"

# ---------------------------
# Dialect Lemma Banks
# ---------------------------
LEMMA_BANK: Dict[str, str] = {
    "siciliano": "http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank",
    "parmigiano": "http://liita.it/data/id/DialettoParmigiano/lemma/LemmaBank",
}

DIALECT_LEXICON_IRI: Dict[str, str] = {
    "parmigiano": "<http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon>",
    "siciliano": "<http://liita.it/data/id/LexicalReources/DialettoSiciliano/Lexicon>",
}

# ---------------------------
# Emotion Mapping (ELIta)
# ---------------------------
EMOTION_MAP: Dict[str, str] = {
    # Italian
    "gioia": "elita:Gioia",
    "tristezza": "elita:Tristezza",
    "paura": "elita:Paura",
    "rabbia": "elita:Rabbia",
    "disgusto": "elita:Disgusto",
    "sorpresa": "elita:Sorpresa",
    "aspettativa": "elita:Aspettativa",
    "fiducia": "elita:Fiducia",
    "amore": "elita:Amore",
    # English aliases
    "joy": "elita:Gioia",
    "sadness": "elita:Tristezza",
    "fear": "elita:Paura",
    "anger": "elita:Rabbia",
    "disgust": "elita:Disgusto",
    "surprise": "elita:Sorpresa",
    "anticipation": "elita:Aspettativa",
    "trust": "elita:Fiducia",
    "love": "elita:Amore",
}

# ---------------------------
# Semantic Relation Keywords
# ---------------------------
# Format: (keywords_it_en, predicate_iri, default_direction)
RELATION_KEYWORDS: List[Tuple[List[str], str, str]] = [
    (["iponimo", "iponimi", "hyponym", "hyponyms", "narrower"], "lexinfo:hyponym", "seed_to_rel"),
    (["iperonimo", "hypernym", "broader"], "lexinfo:hyponym", "rel_to_seed"),  # invert using same predicate
    (["sinonimo", "sinonimi", "synonym", "synonyms", "simile", "near synonym"], "lexinfo:approximateSynonym", "seed_to_rel"),
    (["antonimo", "antonym", "opposto", "opposite"], "lexinfo:antonym", "seed_to_rel"),
    (["meronimo", "part of", "parte di", "component"], "lexinfo:partMeronym", "seed_to_rel"),
    (["olonimo", "whole of", "insieme", "whole"], "lexinfo:partHolonym", "seed_to_rel"),
]

ALLOWED_REL_PREDS: Set[str] = {
    "lexinfo:hyponym",
    "lexinfo:approximateSynonym",
    "lexinfo:antonym",
    "lexinfo:partMeronym",
    "lexinfo:partHolonym",
}
