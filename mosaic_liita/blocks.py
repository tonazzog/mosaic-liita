"""
Block model and registry for MoSAIC-LiITA.

This module provides the building blocks for constructing SPARQL queries:
- Block: A reusable SPARQL pattern template
- BlockInstance: A block with filled slot values
- BlockCall: A reference to a block with slot values (used in QuerySpec)
- BlockRegistry: Container for all available blocks
- make_registry: Factory function to create a populated registry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .constants import COMPLIT_SERVICE, LEMMA_BANK


@dataclass(frozen=True)
class Block:
    """
    A reusable macro block.

    Attributes:
        id: Unique identifier for the block
        requires: Variables that must already exist in the query
        provides: Variables that the block creates/binds
        prefixes: Prefix keys that must be declared
        where: List of SPARQL lines (no surrounding WHERE {})
        service_iri: If set, wraps where-lines inside SERVICE <iri> { ... }
    """
    id: str
    requires: Set[str]
    provides: Set[str]
    prefixes: Set[str]
    where: List[str]
    service_iri: Optional[str] = None

    def render_where(self) -> List[str]:
        """Render the WHERE clause lines, optionally wrapped in SERVICE."""
        if not self.service_iri:
            return self.where
        lines = [f"SERVICE <{self.service_iri}> {{"] + [f"  {ln}" for ln in self.where] + ["}"]
        return lines


@dataclass
class BlockInstance:
    """A block with slots filled (string templating)."""
    block: Block
    slots: Dict[str, str] = field(default_factory=dict)

    def render_where(self) -> List[str]:
        """Render WHERE clause with slot values substituted."""
        def fmt(line: str) -> str:
            return line.format(**self.slots)

        rendered = [fmt(ln) for ln in self.block.where]
        if not self.block.service_iri:
            return rendered
        return [f"SERVICE <{self.block.service_iri}> {{"] + [f"  {ln}" for ln in rendered] + ["}"]

    @property
    def requires(self) -> Set[str]:
        return self.block.requires

    @property
    def provides(self) -> Set[str]:
        return self.block.provides

    @property
    def prefixes(self) -> Set[str]:
        return self.block.prefixes


@dataclass(frozen=True)
class BlockCall:
    """A single macro call: a block id + the slot values to fill its {placeholders}."""
    block_id: str
    slots: Dict[str, str] = field(default_factory=dict)


class BlockRegistry:
    """Container for all available blocks."""

    def __init__(self) -> None:
        self.blocks: Dict[str, Block] = {}

    def register(self, block: Block) -> None:
        """Register a block. Raises KeyError if id already exists."""
        if block.id in self.blocks:
            raise KeyError(f"Block id already registered: {block.id}")
        self.blocks[block.id] = block

    def get(self, block_id: str) -> Block:
        """Get a block by id. Raises KeyError if not found."""
        return self.blocks[block_id]


def make_registry() -> BlockRegistry:
    """Create and populate a BlockRegistry with all predefined blocks."""
    R = BlockRegistry()

    # A) CompL-IT semantic relations
    R.register(Block(
        id="COMPLIT_SEMREL_OF_SEED_LEMMA",
        requires=set(),
        provides={"?wordRel", "?senseRel", "?itLemmaString", "?definition"},
        prefixes={"ontolex", "lexinfo", "rdfs", "skos"},
        service_iri=COMPLIT_SERVICE,
        where=[
            "?seedWord a ontolex:Word ;",
            "          lexinfo:partOfSpeech [ rdfs:label ?posLabel ] ;",
            "          ontolex:canonicalForm [ ontolex:writtenRep ?seedForm ] ;",
            "          ontolex:sense ?seedSense .",
            "{seed_pos_filter}",
            "FILTER(str(?seedForm) = {seed_lemma}) .",
            "{rel_triple}",
            "OPTIONAL {{ ?senseRel skos:definition ?definition . }}",
            "?wordRel ontolex:sense ?senseRel ;",
            "         ontolex:canonicalForm ?formRel .",
            "?formRel ontolex:writtenRep ?itLemmaString .",
            "{rel_extra}",
        ],
    ))

    R.register(Block(
        id="COMPLIT_SEMREL_BETWEEN_TWO_LEMMAS",
        requires=set(),
        provides={"?senseA", "?senseB"},
        prefixes={"ontolex", "lexinfo", "rdfs"},
        service_iri=COMPLIT_SERVICE,
        where=[
            "?wA a ontolex:Word ;",
            "    ontolex:canonicalForm [ ontolex:writtenRep ?wrA ] ;",
            "    ontolex:sense ?senseA .",
            "FILTER(str(?wrA) = {lemma_a}) .",
            "?wB a ontolex:Word ;",
            "    ontolex:canonicalForm [ ontolex:writtenRep ?wrB ] ;",
            "    ontolex:sense ?senseB .",
            "FILTER(str(?wrB) = {lemma_b}) .",
            "{rel_triple_ab}",
        ],
    ))

    # B) CompL-IT definition filters
    R.register(Block(
        id="COMPLIT_DEF_STARTS_WITH",
        requires=set(),
        provides={"?word", "?itLemmaString", "?definition"},
        prefixes={"ontolex", "lexinfo", "rdfs", "skos"},
        service_iri=COMPLIT_SERVICE,
        where=[
            "?word a ontolex:Word ;",
            "      lexinfo:partOfSpeech [ rdfs:label ?posLabel ] ;",
            "      ontolex:sense ?sense ;",
            "      ontolex:canonicalForm ?form .",
            "?form ontolex:writtenRep ?itLemmaString .",
            "OPTIONAL {{ ?sense skos:definition ?definition . }}",
            "{pos_filter}",
            "FILTER(strstarts(str(?definition), {definition_prefix})) .",
        ],
    ))

    R.register(Block(
        id="COMPLIT_DEF_FILTER_BY_PATTERN",
        requires=set(),
        provides={"?word", "?itLemmaString", "?definition"},
        prefixes={"ontolex", "lexinfo", "rdfs", "skos"},
        service_iri=COMPLIT_SERVICE,
        where=[
            "?word a ontolex:Word ;",
            "      lexinfo:partOfSpeech [ rdfs:label ?posLabel ] ;",
            "      ontolex:sense ?sense ;",
            "      ontolex:canonicalForm ?form .",
            "?form ontolex:writtenRep ?itLemmaString .",
            "{lemma_filter}",
            "OPTIONAL {{ ?sense skos:definition ?definition . }}",
            "{pos_filter}",
            "{def_filter}",
        ],
    ))

    # C) Join CompL-IT word to LiITA lemma
    R.register(Block(
        id="JOIN_WORD_TO_LIITA_FROM_WORD",
        requires={"?word"},
        provides={"?liitaLemma"},
        prefixes={"ontolex"},
        where=[
            "?word ontolex:canonicalForm ?liitaLemma ."
        ],
    ))

    R.register(Block(
        id="JOIN_WORDREL_TO_LIITA",
        requires={"?wordRel"},
        provides={"?liitaLemma"},
        prefixes={"ontolex"},
        where=[
            "?wordRel ontolex:canonicalForm ?liitaLemma ."
        ],
    ))

    # D) LiITA lemma type assertion
    R.register(Block(
        id="ASSERT_LIITA_LEMMA_TYPE",
        requires={"?liitaLemma"},
        provides=set(),
        prefixes={"lila"},
        where=[
            "?liitaLemma a lila:Lemma ."
        ],
    ))

    # E) LiITA lemma pattern filter
    R.register(Block(
        id="LIITA_LEMMA_FILTER_BY_PATTERN_AND_POS",
        requires=set(),
        provides={"?lemma", "?wr"},
        prefixes={"lila", "ontolex"},
        where=[
            "GRAPH <http://liita.it/data> {{",
            "  ?lemma a lila:Lemma .",
            "  {pos_clause}",
            "  ?lemma ontolex:writtenRep ?wr .",
            "  {wr_filter}",
            "}}",
        ],
    ))

    R.register(Block(
        id="LIITA_LEMMA_POS",
        requires={"?lemma"},
        provides={"?pos"},
        prefixes={"lila"},
        where=[
            "GRAPH <http://liita.it/data> {{",
            "  ?lemma lila:hasPOS ?pos .",
            "}}",
        ],
    ))

    R.register(Block(
        id="LIITA_LEMMA_POS_FROM_LIITA",
        requires={"?liitaLemma"},
        provides={"?pos"},
        prefixes={"lila"},
        where=[
            "GRAPH <http://liita.it/data> {{",
            "  ?liitaLemma lila:hasPOS ?pos .",
            "}}",
        ],
    ))

    # F) Sentix polarity
    R.register(Block(
        id="SENTIX_POLARITY",
        requires={"?liitaLemma"},
        provides={"?polarityLabel", "?polarityValue"},
        prefixes={"marl", "ontolex", "rdfs"},
        where=[
            "?sentixLemma ontolex:canonicalForm ?liitaLemma .",
            "OPTIONAL {{ ?sentixLemma marl:hasPolarityValue ?polarityValue . }}",
            "OPTIONAL {{",
            "  ?sentixLemma marl:hasPolarity ?polarity .",
            "  ?polarity rdfs:label ?polarityLabel .",
            "}}",
        ],
    ))

    # G) ELIta emotion filter
    R.register(Block(
        id="ELITA_EMOTION_FILTER",
        requires={"?liitaLemma"},
        provides={"?emotion", "?emotionLabel"},
        prefixes={"elita", "ontolex", "rdfs"},
        where=[
            "?elitaLemma ontolex:canonicalForm ?liitaLemma .",
            "?elitaLemma elita:HasEmotion ?emotion .",
            "?emotion rdfs:label ?emotionLabel .",
            "{emotion_filter_clause}",
        ],
    ))

    # H) Translate Italian -> Siciliano
    R.register(Block(
        id="TRANSLATE_TO_SICILIANO",
        requires={"?liitaLemma"},
        provides={"?sicLemma", "?sicWR"},
        prefixes={"ontolex", "vartrans", "dcterms"},
        where=[
            "?itToSicEntry ontolex:canonicalForm ?liitaLemma ;",
            "             vartrans:translatableAs ?sicEntry .",
            "?sicEntry ontolex:canonicalForm ?sicLemma .",
            "?sicLemma dcterms:isPartOf <" + LEMMA_BANK["siciliano"] + "> ;",
            "         ontolex:writtenRep ?sicWR .",
        ],
    ))

    # I) Translate Siciliano -> Italian
    R.register(Block(
        id="TRANSLATE_FROM_SICILIANO",
        requires={"?sicLemma"},
        provides={"?liitaLemma", "?itLemmaString"},
        prefixes={"ontolex", "vartrans"},
        where=[
            "?sicEntry ontolex:canonicalForm ?sicLemma .",
            "?itEntry vartrans:translatableAs ?sicEntry .",
            "?itEntry ontolex:canonicalForm ?liitaLemma .",
            "?liitaLemma ontolex:writtenRep ?itLemmaString .",
        ],
    ))

    # J) Translate Italian -> Parmigiano
    R.register(Block(
        id="TRANSLATE_TO_PARMIGIANO",
        requires={"?liitaLemma"},
        provides={"?parLemma", "?parWR"},
        prefixes={"ontolex", "vartrans", "dcterms"},
        where=[
            "?itToParEntry ontolex:canonicalForm ?liitaLemma ;",
            "             vartrans:translatableAs ?parEntry .",
            "?parEntry ontolex:canonicalForm ?parLemma .",
            "?parLemma dcterms:isPartOf <" + LEMMA_BANK["parmigiano"] + "> ;",
            "         ontolex:writtenRep ?parWR .",
        ],
    ))

    # K) Translate Parmigiano -> Italian
    R.register(Block(
        id="TRANSLATE_FROM_PARMIGIANO",
        requires={"?parLemma"},
        provides={"?liitaLemma", "?itLemmaString"},
        prefixes={"ontolex", "vartrans"},
        where=[
            "?parEntry ontolex:canonicalForm ?parLemma .",
            "?itEntry vartrans:translatableAs ?parEntry .",
            "?itEntry ontolex:canonicalForm ?liitaLemma .",
            "?liitaLemma ontolex:writtenRep ?itLemmaString .",
        ],
    ))

    # L) CompL-IT sense counting (single word)
    R.register(Block(
        id="COMPLIT_COUNT_SENSES_SINGLE",
        requires=set(),
        provides={"?sense", "?writtenRep"},
        prefixes={"ontolex"},
        service_iri=COMPLIT_SERVICE,
        where=[
            "?word ontolex:canonicalForm [ ontolex:writtenRep ?writtenRep ] ;",
            "      ontolex:sense ?sense .",
            'FILTER(STR(?writtenRep) = {seed_lemma}) .',
        ],
    ))

    # M) CompL-IT sense counting (multiple words)
    R.register(Block(
        id="COMPLIT_COUNT_SENSES_MULTI",
        requires=set(),
        provides={"?sense", "?writtenRep"},
        prefixes={"ontolex"},
        service_iri=COMPLIT_SERVICE,
        where=[
            "?lexicalEntry ontolex:canonicalForm [ ontolex:writtenRep ?writtenRep ] .",
            "?lexicalEntry ontolex:sense ?sense .",
            "{wr_regex_filter}",
        ],
    ))

    # N) Siciliano lemma pattern filter
    R.register(Block(
        id="SICILIANO_LEMMA_FILTER_BY_PATTERN_AND_POS",
        requires=set(),
        provides={"?sicLemma", "?sicWR"},
        prefixes={"lila", "ontolex", "dcterms"},
        where=[
            "  ?sicLemma a lila:Lemma .",
            "  ?sicLemma dcterms:isPartOf <" + LEMMA_BANK["siciliano"] + "> .",
            "  {pos_clause}",
            "  ?sicLemma ontolex:writtenRep ?sicWR .",
            "  {wr_filter}",
        ],
    ))

    # M) Parmigiano lemma pattern filter
    R.register(Block(
        id="PARMIGIANO_LEMMA_FILTER_BY_PATTERN_AND_POS",
        requires=set(),
        provides={"?parLemma", "?parWR"},
        prefixes={"lila", "ontolex", "dcterms"},
        where=[
            "  ?parLemma a lila:Lemma .",
            "  ?parLemma dcterms:isPartOf <" + LEMMA_BANK["parmigiano"] + "> .",
            "  {pos_clause}",
            "  ?parLemma ontolex:writtenRep ?parWR .",
            "  {wr_filter}",
        ],
    ))

    return R
