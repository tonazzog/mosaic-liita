"""
MoSAIC-LiITA: MOdular Sparql Assembler for Interlinked Corpora.

This package provides tools for translating natural language queries into
SPARQL queries for the LiITA (Linguistic Linked Data for Italian) database.

Example usage:
    from mosaic_liita import Planner, Assembler, make_registry

    registry = make_registry()
    planner = Planner(registry, catalog)
    assembler = Assembler()

    spec = planner.plan("Find synonyms of 'antico'")
    plan = spec.compile(registry)
    sparql = assembler.assemble(plan)
"""

# Core classes
from .blocks import Block, BlockInstance, BlockCall, BlockRegistry, make_registry
from .query import QuerySpec, QueryPlan, QuerySpecError
from .planner import Planner
from .assembler import Assembler

# Optional LLM refinement
from .llm_refinement import llm_refine_queryspec

# Agentic query decomposition
from .agent import QueryAgent, AgentPlan, ToolCall, AGENT_TOOLS

# Utility functions
from .utils import (
    norm,
    extract_quoted_strings,
    contains_any,
    sparql_quote,
    extract_pattern_request,
    build_filter_for_var,
    map_pos,
)

# Relation resolution
from .relations import resolve_relation

# Constants
from .constants import (
    PREFIXES,
    COMPLIT_SERVICE,
    LEMMA_BANK,
    EMOTION_MAP,
    RELATION_KEYWORDS,
)

__all__ = [
    # Core classes
    "Block",
    "BlockInstance",
    "BlockCall",
    "BlockRegistry",
    "make_registry",
    "QuerySpec",
    "QueryPlan",
    "QuerySpecError",
    "Planner",
    "Assembler",
    # LLM refinement
    "llm_refine_queryspec",
    # Utils
    "norm",
    "extract_quoted_strings",
    "contains_any",
    "sparql_quote",
    "extract_pattern_request",
    "build_filter_for_var",
    "map_pos",
    # Relations
    "resolve_relation",
    # Constants
    "PREFIXES",
    "COMPLIT_SERVICE",
    "LEMMA_BANK",
    "EMOTION_MAP",
    "RELATION_KEYWORDS",
]
