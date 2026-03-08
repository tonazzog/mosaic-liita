"""
Agentic query decomposition for MoSAIC-LiITA.

This module provides an LLM-powered agent that decomposes complex natural language
queries into a sequence of block operations. The agent understands the block system
as "tools" and produces a structured plan that can be validated and assembled.

Key concepts:
- Tools: Block operations exposed to the LLM with friendly descriptions
- AgentPlan: Structured plan produced by the LLM
- QueryAgent: Orchestrates decomposition, validation, and assembly
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from shared.llm import BaseLLM

from .blocks import Block, BlockCall, BlockRegistry
from .query import QuerySpec, QueryPlan, QuerySpecError, validate_queryspec, toposort_blocks
from .constants import EMOTION_MAP
from .utils import sparql_quote, build_filter_for_var
from .relations import resolve_relation


# =============================================================================
# Tool Definitions
# =============================================================================

# Tools are high-level operations the agent can use.
# Each tool maps to one or more blocks and has parameters the LLM can set.

AGENT_TOOLS: Dict[str, Dict[str, Any]] = {
    "find_semantic_relations": {
        "description": "Find words that have a semantic relation (hyponym, hypernym, synonym, antonym, meronym, holonym) to a seed word. Use this for queries like 'find hyponyms of X' or 'find synonyms of Y'.",
        "block": "COMPLIT_SEMREL_OF_SEED_LEMMA",
        "parameters": {
            "seed_word": {"type": "string", "description": "The word to find relations for", "required": True},
            "relation_type": {"type": "string", "enum": ["hyponym", "hypernym", "synonym", "antonym", "meronym", "holonym"], "description": "Type of semantic relation", "required": True},
            "pos_filter": {"type": "string", "enum": ["noun", "verb", "adjective", "adverb", None], "description": "Optional part-of-speech filter", "required": False},
        },
        "provides": ["?wordRel", "?senseRel", "?itLemmaString", "?definition"],
        "requires": [],
    },

    "find_definitions_by_pattern": {
        "description": "Find CompL-IT words and definitions where the lemma OR definition matches a pattern (starts with, ends with, contains). Use this for queries about definitions.",
        "block": "COMPLIT_DEF_FILTER_BY_PATTERN",
        "parameters": {
            "pattern_type": {"type": "string", "enum": ["prefix", "suffix", "contains"], "description": "How to match the pattern", "required": True},
            "pattern_text": {"type": "string", "description": "The text pattern to match", "required": True},
            "apply_to": {"type": "string", "enum": ["lemma", "definition"], "description": "Apply pattern to lemma or definition text", "required": True},
            "pos_filter": {"type": "string", "enum": ["noun", "verb", "adjective", "adverb", None], "description": "Optional part-of-speech filter", "required": False},
        },
        "provides": ["?word", "?itLemmaString", "?definition"],
        "requires": [],
    },

    "find_liita_lemmas_by_pattern": {
        "description": "Find Italian lemmas in LiITA that match a pattern (starts with, ends with, contains). Use this for queries about Italian words/lemmas without needing definitions.",
        "block": "LIITA_LEMMA_FILTER_BY_PATTERN_AND_POS",
        "parameters": {
            "pattern_type": {"type": "string", "enum": ["exact", "prefix", "suffix", "contains"], "description": "How to match: 'exact' for a specific word (e.g. translate 'pane'), 'prefix'/'suffix'/'contains' for morphological patterns", "required": True},
            "pattern_text": {"type": "string", "description": "The text to match. Use the bare word for 'exact' (e.g. 'pane'), a prefix/suffix for the others.", "required": True},
            "pos_filter": {"type": "string", "enum": ["noun", "verb", "adjective", "adverb", None], "description": "Optional part-of-speech filter. MUST be one of the exact strings: 'noun', 'verb', 'adjective', 'adverb'. Do NOT use abbreviations (ADJ, ADV, VERB, etc.).", "required": False},
        },
        "provides": ["?lemma", "?wr"],
        "requires": [],
    },

    "find_sicilian_lemmas_by_pattern": {
        "description": "Find Sicilian dialect lemmas that match a pattern. Use this when the query specifically asks about Sicilian words.",
        "block": "SICILIANO_LEMMA_FILTER_BY_PATTERN_AND_POS",
        "parameters": {
            "pattern_type": {"type": "string", "enum": ["prefix", "suffix", "contains"], "description": "How to match the pattern", "required": True},
            "pattern_text": {"type": "string", "description": "The text pattern to match", "required": True},
        },
        "provides": ["?sicLemma", "?sicWR"],
        "requires": [],
    },

    "find_parmigiano_lemmas_by_pattern": {
        "description": "Find Parmigiano dialect lemmas that match a pattern. Use this when the query specifically asks about Parmigiano words.",
        "block": "PARMIGIANO_LEMMA_FILTER_BY_PATTERN_AND_POS",
        "parameters": {
            "pattern_type": {"type": "string", "enum": ["prefix", "suffix", "contains"], "description": "How to match the pattern", "required": True},
            "pattern_text": {"type": "string", "description": "The text pattern to match", "required": True},
        },
        "provides": ["?parLemma", "?parWR"],
        "requires": [],
    },

    "join_to_liita": {
        "description": "Connect CompL-IT word results to LiITA lemmas. Required before getting translations, sentiment, or emotions.",
        "block_variants": {
            "?word": "JOIN_WORD_TO_LIITA_FROM_WORD",
            "?wordRel": "JOIN_WORDREL_TO_LIITA",
        },
        "parameters": {
            "source_var": {"type": "string", "enum": ["?word", "?wordRel"], "description": "The CompL-IT variable to join from", "required": True},
        },
        "provides": ["?liitaLemma"],
        "requires_one_of": ["?word", "?wordRel"],
    },

    "bind_lemma_to_liita": {
        "description": "Bind ?lemma variable to ?liitaLemma. Use after find_liita_lemmas_by_pattern when you need translations/sentiment/emotions.",
        "block": "BIND_LEMMA_TO_LIITALEMMA",
        "parameters": {},
        "provides": ["?liitaLemma"],
        "requires": ["?lemma"],
    },

    "translate_to_sicilian": {
        "description": "Get Sicilian translations for Italian lemmas.",
        "block": "TRANSLATE_TO_SICILIANO",
        "parameters": {},
        "provides": ["?sicLemma", "?sicWR"],
        "requires": ["?liitaLemma"],
    },

    "translate_to_parmigiano": {
        "description": "Get Parmigiano translations for Italian lemmas.",
        "block": "TRANSLATE_TO_PARMIGIANO",
        "parameters": {},
        "provides": ["?parLemma", "?parWR"],
        "requires": ["?liitaLemma"],
    },

    "translate_from_sicilian": {
        "description": "Get Italian translations for Sicilian lemmas.",
        "block": "TRANSLATE_FROM_SICILIANO",
        "parameters": {},
        "provides": ["?liitaLemma", "?itLemmaString"],
        "requires": ["?sicLemma"],
    },

    "translate_from_parmigiano": {
        "description": "Get Italian translations for Parmigiano lemmas.",
        "block": "TRANSLATE_FROM_PARMIGIANO",
        "parameters": {},
        "provides": ["?liitaLemma", "?itLemmaString"],
        "requires": ["?parLemma"],
    },

    "get_pos": {
        "description": "Get the part of speech (POS) for LiITA lemmas. Provides ?pos (IRI like lila:noun). Use this when you need to group or filter by POS category.",
        "block_variants": {
            "?lemma": "LIITA_LEMMA_POS",
            "?liitaLemma": "LIITA_LEMMA_POS_FROM_LIITA",
        },
        "parameters": {
            "source_var": {"type": "string", "enum": ["?lemma", "?liitaLemma"], "description": "The lemma variable to get POS for", "required": True},
        },
        "provides": ["?pos"],
        "requires_one_of": ["?lemma", "?liitaLemma"],
    },

    "get_sentiment": {
        "description": "Get sentiment polarity (positive/negative/neutral) from Sentix.",
        "block": "SENTIX_POLARITY",
        "parameters": {},
        "provides": ["?polarityLabel", "?polarityValue"],
        "requires": ["?liitaLemma"],
    },

    "get_emotions": {
        "description": "Get emotion annotations from ELIta. Can filter to specific emotions. Provides ?emotion (IRI like elita:Gioia) and ?emotionLabel (string like 'Gioia'). Use ?emotion for grouping/counting, ?emotionLabel for display.",
        "block": "ELITA_EMOTION_FILTER",
        "parameters": {
            "emotions": {"type": "array", "items": "string", "description": "Optional list of emotions to filter (e.g., ['gioia', 'tristezza']). Leave empty for all emotions.", "required": False},
        },
        "provides": ["?emotion", "?emotionLabel"],
        "requires": ["?liitaLemma"],
    },

    "filter_variable": {
        "description": "Add a pattern filter to any string variable. Use this to add additional constraints on results. IMPORTANT: For written representations use ?sicWR (Sicilian), ?parWR (Parmigiano), ?wr (Italian), ?itLemmaString (CompL-IT lemma text). Do NOT filter on lemma IRIs like ?sicLemma.",
        "type": "filter",
        "parameters": {
            "variable": {"type": "string", "description": "The variable to filter - must be a string variable like ?sicWR, ?parWR, ?wr, ?itLemmaString, ?definition", "required": True},
            "pattern_type": {"type": "string", "enum": ["prefix", "suffix", "contains"], "description": "How to match: 'prefix' for starts-with, 'suffix' for ends-with, 'contains' for substring", "required": True},
            "pattern_text": {"type": "string", "description": "The text pattern", "required": True},
        },
        "provides": [],
        "requires": [],  # Will be inferred from variable
    },

    "count_senses": {
        "description": "Count the number of senses (meanings) a word has in CompL-IT. Use for queries about polysemy, 'how many senses/meanings does X have'. For multiple words, pass them all in the 'words' parameter.",
        "parameters": {
            "words": {"type": "array", "items": "string", "description": "One or more words to count senses for", "required": True},
        },
        "provides": ["?sense", "?writtenRep"],
        "requires": [],
    },

    "aggregate_results": {
        "description": (
            "Compute an aggregate (AVG or COUNT) over results, with optional GROUP BY and HAVING. "
            "Use for 'average polarity', 'mean score', 'distribution', etc. "
            "For AVG on polarity values, use agg_function='AVG', agg_variable='?polarityValue'. "
            "For COUNT, use agg_function='COUNT', agg_variable='?liitaLemma'. "
            "HAVING filters on the aggregate value (e.g., '> 0' for positive, '< 0' for negative). "
            "Use xsd_cast='float' when averaging string-typed numeric values like polarity scores."
        ),
        "type": "aggregation",
        "parameters": {
            "agg_function": {"type": "string", "enum": ["COUNT", "AVG", "SUM", "MIN", "MAX"], "description": "Aggregation function", "required": True},
            "agg_variable": {"type": "string", "description": "Variable to aggregate (e.g., ?polarityValue, ?liitaLemma)", "required": True},
            "distinct": {"type": "boolean", "description": "Use DISTINCT in aggregate (default false)", "required": False},
            "xsd_cast": {"type": "string", "enum": ["float", "integer", None], "description": "Cast variable to xsd type before aggregating (needed for AVG on string-typed values)", "required": False},
            "group_by": {"type": "array", "items": "string", "description": "Variables to group by (e.g., ['?emotion'])", "required": False},
            "having_op": {"type": "string", "enum": [">", "<", ">=", "<=", "=", None], "description": "HAVING comparison operator on aggregate value", "required": False},
            "having_value": {"type": "string", "description": "Value to compare in HAVING (e.g., '0')", "required": False},
            "order": {"type": "string", "enum": ["asc", "desc", None], "description": "Sort order for the aggregate value", "required": False},
            "limit": {"type": "integer", "description": "Limit number of results (e.g., 1 for 'fewest'/'most')", "required": False},
        },
        "provides": [],
        "requires": [],
    },

    "count_results": {
        "description": (
            "Count the results instead of listing them. Use for 'how many' queries. "
            "IMPORTANT: include ALL dimensions that the question groups or breaks down by in group_by. "
            "E.g. 'distribution of emotions by POS' → group_by=['?emotionLabel','?pos']. "
            "For a single total with no breakdown, use group_by=[] (empty). "
            "For superlative queries ('fewest', 'most'), set order='asc' or 'desc' and limit=1."
        ),
        "type": "aggregation",
        "parameters": {
            "count_variable": {"type": "string", "description": "Variable to count (e.g., ?lemma, ?liitaLemma)", "required": True},
            "group_by": {"type": "array", "items": "string", "description": "Variables to group by. Include EVERY dimension mentioned in the question (e.g., ['?emotionLabel', '?pos'] for 'by emotion and POS'). Use [] for a single total.", "required": False},
            "order": {"type": "string", "enum": ["asc", "desc", None], "description": "Sort order for the count: 'asc' for fewest/lowest, 'desc' for most/highest. Omit if no ordering needed.", "required": False},
            "limit": {"type": "integer", "description": "Limit number of results (e.g., 1 for 'the fewest' or 'the most'). Omit if all results are needed.", "required": False},
        },
        "provides": ["?count"],
        "requires": [],
    },
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ToolCall:
    """A single tool invocation in the agent's plan."""
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "params": self.params}


@dataclass
class AgentPlan:
    """Structured plan produced by the agent."""
    reasoning: str
    steps: List[ToolCall]
    output_vars: List[str]
    filters: List[Dict[str, Any]] = field(default_factory=list)
    aggregation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "steps": [s.to_dict() for s in self.steps],
            "output_vars": self.output_vars,
            "filters": self.filters,
            "aggregation": self.aggregation,
        }


@dataclass
class ValidationResult:
    """Result of plan validation."""
    valid: bool
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Agent Implementation
# =============================================================================

class QueryAgent:
    """
    LLM-powered agent that decomposes natural language queries into block operations.

    The agent:
    1. Receives a natural language query
    2. Uses an LLM to decompose it into tool calls
    3. Validates the resulting plan
    4. Converts the plan to a QuerySpec
    5. Returns the assembled SPARQL
    """

    def __init__(
        self,
        registry: BlockRegistry,
        catalog: List[Dict[str, Any]],
        llm_client: "BaseLLM",
    ) -> None:
        """
        Initialize the agent.

        Args:
            registry: Block registry with available blocks
            catalog: Ontology catalog for relation resolution
            llm_client: LLM client for query decomposition
        """
        self.registry = registry
        self.catalog = catalog
        self.llm = llm_client

        # Ensure dynamic blocks are registered
        self._register_dynamic_blocks()

    def _register_dynamic_blocks(self) -> None:
        """Register blocks that may be created dynamically."""
        # BIND block for ?lemma -> ?liitaLemma
        if "BIND_LEMMA_TO_LIITALEMMA" not in self.registry.blocks:
            bind_block = Block(
                id="BIND_LEMMA_TO_LIITALEMMA",
                requires={"?lemma"},
                provides={"?liitaLemma"},
                prefixes=set(),
                where=["BIND(?lemma AS ?liitaLemma)"],
            )
            self.registry.blocks[bind_block.id] = bind_block

    def decompose(
        self,
        nl: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> AgentPlan:
        """
        Decompose a natural language query into an agent plan.

        Args:
            nl: Natural language query
            temperature: LLM temperature (0 for determinism)
            max_tokens: Maximum response tokens

        Returns:
            AgentPlan with tool calls and output specification
        """
        prompt = self._build_prompt(nl)
        system = self._build_system_prompt()

        raw = self.llm.complete(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return self._parse_response(raw)

    def plan_to_queryspec(
        self,
        plan: AgentPlan,
    ) -> QuerySpec:
        """
        Convert an AgentPlan to a QuerySpec.

        Args:
            plan: Validated agent plan

        Returns:
            QuerySpec ready for compilation
        """
        blocks: List[BlockCall] = []
        known_vars: Set[str] = set()
        custom_filters: List[str] = []
        filter_requires: Set[str] = set()  # Track variables used in filters
        captured_aggregation: Optional[Dict[str, Any]] = None  # Aggregation from tool steps

        for step in plan.steps:
            tool_def = AGENT_TOOLS.get(step.tool)
            if not tool_def:
                raise QuerySpecError(f"Unknown tool: {step.tool}")

            # Handle different tool types
            if tool_def.get("type") == "filter":
                # Collect filters to add later
                var = step.params.get("variable", "")
                ptype = step.params.get("pattern_type", "")
                ptext = step.params.get("pattern_text", "")

                # Normalize pattern_type aliases
                ptype_map = {
                    "starts_with": "prefix",
                    "startswith": "prefix",
                    "begins_with": "prefix",
                    "ends_with": "suffix",
                    "endswith": "suffix",
                    "includes": "contains",
                    "has": "contains",
                }
                ptype = ptype_map.get(ptype.lower(), ptype.lower()) if ptype else ""

                # Normalize variable to written representation if needed
                var_wr_map = {
                    "?sicLemma": "?sicWR",
                    "?parLemma": "?parWR",
                    "?lemma": "?wr",
                    "?liitaLemma": "?wr",
                }
                var = var_wr_map.get(var, var)

                if var and ptype and ptext:
                    filter_clause = build_filter_for_var(var, ptype, ptext)
                    if filter_clause:
                        custom_filters.append(filter_clause)
                        filter_requires.add(var)
                continue

            if tool_def.get("type") == "aggregation":
                if step.tool == "aggregate_results":
                    # Rich aggregation: AVG, COUNT, etc. with HAVING
                    captured_aggregation = {
                        "agg_function": step.params.get("agg_function", "COUNT"),
                        "agg_variable": step.params.get("agg_variable", "?lemma"),
                        "distinct": step.params.get("distinct", False),
                        "xsd_cast": step.params.get("xsd_cast"),
                        "group_by": step.params.get("group_by") or [],
                        "having_op": step.params.get("having_op"),
                        "having_value": step.params.get("having_value"),
                        "order": step.params.get("order"),
                        "limit": step.params.get("limit"),
                    }
                else:
                    # Legacy count_results tool
                    captured_aggregation = {
                        "agg_function": "COUNT",
                        "agg_variable": step.params.get("count_variable", "?lemma"),
                        "distinct": True,
                        "group_by": step.params.get("group_by") or [],
                        "order": step.params.get("order"),
                        "limit": step.params.get("limit"),
                    }
                continue

            # Get block ID
            block_id = self._resolve_block_id(step, tool_def, known_vars)
            if not block_id:
                continue

            # Build slots
            slots = self._build_slots(step, tool_def, block_id)

            blocks.append(BlockCall(block_id=block_id, slots=slots))

            # Update known vars
            if block_id in self.registry.blocks:
                known_vars |= self.registry.get(block_id).provides

        # Add custom filter block if needed
        if custom_filters:
            # Use unique block ID to avoid collisions
            filter_block_id = f"CUSTOM_FILTERS_{id(custom_filters)}"
            filter_block = Block(
                id=filter_block_id,
                requires=filter_requires,
                provides=set(),
                prefixes=set(),
                where=custom_filters,
            )
            self.registry.blocks[filter_block_id] = filter_block
            blocks.append(BlockCall(block_id=filter_block_id))

        # Use captured aggregation from tool steps if plan.aggregation is not set
        effective_aggregation = plan.aggregation or captured_aggregation

        # Build output specification
        select_vars, aggregates, group_by = self._build_output_spec(plan, known_vars, effective_aggregation)

        # Determine order_by, having — prefer explicit LLM instruction, then defaults
        agg_order = effective_aggregation.get("order") if effective_aggregation else None
        agg_limit = effective_aggregation.get("limit") if effective_aggregation else None
        having_op = effective_aggregation.get("having_op") if effective_aggregation else None
        having_value = effective_aggregation.get("having_value") if effective_aggregation else None

        # Find the aggregate alias for ordering/having
        agg_alias = None
        agg_expr_for_having = None
        if aggregates:
            agg_alias = list(aggregates.keys())[0]  # e.g., ?count or ?avgPolarityValue
            agg_expr_for_having = list(aggregates.values())[0]

        # Build HAVING clause
        having = None
        if having_op and having_value and agg_expr_for_having:
            having = f"HAVING ({agg_expr_for_having} {having_op} {having_value})"

        order_by = None
        if agg_order and agg_alias:
            order_by = f"ORDER BY {'ASC' if agg_order == 'asc' else 'DESC'}({agg_alias})"
        elif "?itLemmaString" in select_vars:
            order_by = "ORDER BY ?itLemmaString"
        elif agg_alias and group_by:
            order_by = f"ORDER BY DESC({agg_alias})"

        limit = agg_limit if agg_limit else (None if aggregates else 200)

        return QuerySpec(
            blocks=blocks,
            select_vars=select_vars,
            aggregates=aggregates,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
        )

    def validate_plan(self, plan: AgentPlan) -> ValidationResult:
        """
        Validate an agent plan for correctness.

        Checks:
        - All tools exist
        - Required parameters are provided
        - Variable dependencies are satisfied
        - Output vars are available
        """
        available_vars: Set[str] = set()
        warnings: List[str] = []

        for i, step in enumerate(plan.steps):
            tool_def = AGENT_TOOLS.get(step.tool)
            if not tool_def:
                return ValidationResult(
                    valid=False,
                    error=f"Step {i+1}: Unknown tool '{step.tool}'"
                )

            # Check required parameters
            params_def = tool_def.get("parameters", {})
            for param_name, param_spec in params_def.items():
                if param_spec.get("required") and param_name not in step.params:
                    return ValidationResult(
                        valid=False,
                        error=f"Step {i+1} ({step.tool}): Missing required parameter '{param_name}'"
                    )

            # Check variable dependencies
            requires = set(tool_def.get("requires", []))
            requires_one_of = tool_def.get("requires_one_of", [])

            if requires and not requires.issubset(available_vars):
                missing = requires - available_vars
                return ValidationResult(
                    valid=False,
                    error=f"Step {i+1} ({step.tool}): Requires {missing} but not available. Available: {available_vars}"
                )

            if requires_one_of and not any(v in available_vars for v in requires_one_of):
                return ValidationResult(
                    valid=False,
                    error=f"Step {i+1} ({step.tool}): Requires one of {requires_one_of} but none available"
                )

            # Add provided vars
            provides = set(tool_def.get("provides", []))
            available_vars |= provides

        # Check output vars
        missing_outputs = set(plan.output_vars) - available_vars
        if missing_outputs:
            # Try to be helpful
            warnings.append(f"Output vars {missing_outputs} not explicitly provided. Will attempt to infer.")

        return ValidationResult(valid=True, warnings=warnings)

    def translate(
        self,
        nl: str,
        temperature: float = 0.0,
    ) -> Tuple[str, AgentPlan, QuerySpec]:
        """
        Full translation pipeline: decompose -> validate -> assemble.

        Args:
            nl: Natural language query
            temperature: LLM temperature

        Returns:
            Tuple of (sparql_query, agent_plan, query_spec)
        """
        # Decompose
        plan = self.decompose(nl, temperature=temperature)

        # Validate
        validation = self.validate_plan(plan)
        if not validation.valid:
            raise QuerySpecError(f"Invalid plan: {validation.error}")

        # Convert to QuerySpec
        spec = self.plan_to_queryspec(plan)

        # Compile and assemble
        from .assembler import Assembler
        query_plan = spec.compile(self.registry)
        sparql = Assembler().assemble(query_plan)

        return sparql, plan, spec

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return """You are a query planner for the LiITA Italian linguistic database. Your job is to decompose natural language queries into a structured plan using the available tools.

IMPORTANT RULES:
1. Output ONLY valid JSON matching the schema below
2. Only use tools from the provided list
3. Ensure variable dependencies are satisfied (check "requires" for each tool)
4. Be minimal - don't add unnecessary tools
5. Never invent new tools or parameters

When a query needs:
- Translations: First get ?liitaLemma (via join_to_liita or bind_lemma_to_liita), then use translate_to_*
- Sentiment: First get ?liitaLemma, then use get_sentiment
- Emotions: First get ?liitaLemma, then use get_emotions
- Additional filtering: Use filter_variable after the main query tools

How to get ?liitaLemma:
- If the query starts from a specific word pattern: use find_liita_lemmas_by_pattern (pattern_type/pattern_text) then bind_lemma_to_liita
- If the query filters Italian words by emotion/sentiment (NO specific word): use find_liita_lemmas_by_pattern with pattern_type="contains" and pattern_text="" (empty = match all), then bind_lemma_to_liita, then get_emotions/get_sentiment
- If the query starts from a CompL-IT word (?word from find_definitions_by_pattern): use join_to_liita with source_var="?word"
- join_to_liita requires ?word or ?wordRel to already be available — it CANNOT be used as Step 1

NEVER place get_emotions, get_sentiment, or join_to_liita as Step 1 — they always require variables from a preceding step.

For translation queries (translate a specific word to a dialect):
- Use find_liita_lemmas_by_pattern with pattern_type="exact" and pattern_text=<the word> to constrain to that word.
- Always include ?wr in output_vars so the Italian word appears alongside the dialect word.

For emotion + polarity queries (e.g., "average polarity by emotion", "which emotions have positive polarity"):
- Use find_liita_lemmas_by_pattern (match all), bind_lemma_to_liita, get_emotions, get_sentiment
- Use aggregate_results with agg_function="AVG", agg_variable="?polarityValue", xsd_cast="float", group_by=["?emotion"]
- For "positive average" add having_op=">", having_value="0"; for "negative average" use having_op="<", having_value="0"
- Group by ?emotion (IRI), NOT ?emotionLabel, for aggregation queries
- For "distribution of emotions by POS", include group_by=["?emotion", "?pos"] and use aggregate_results with COUNT

For "fewest"/"most" queries, use aggregate_results or count_results with order="asc"/"desc" and limit=1.

Output JSON schema:
{
  "reasoning": "Brief explanation of your decomposition approach",
  "steps": [
    {"tool": "tool_name", "params": {"param1": "value1", ...}}
  ],
  "output_vars": ["?var1", "?var2"],
  "filters": [],
  "aggregation": null or {"count_variable": "?var", "group_by": ["?var2"]}
}"""

    def _build_prompt(self, nl: str) -> str:
        """Build the user prompt with tool descriptions."""
        tools_desc = []
        for name, spec in AGENT_TOOLS.items():
            params_desc = []
            for pname, pspec in spec.get("parameters", {}).items():
                req = " (required)" if pspec.get("required") else " (optional)"
                params_desc.append(f"    - {pname}: {pspec.get('description', '')}{req}")

            provides = ", ".join(spec.get("provides", []))
            requires = ", ".join(spec.get("requires", [])) or "none"

            tool_str = f"""
{name}:
  Description: {spec.get('description', '')}
  Parameters:
{chr(10).join(params_desc) if params_desc else '    (none)'}
  Provides: {provides}
  Requires: {requires}"""
            tools_desc.append(tool_str)

        return f"""Available tools:
{''.join(tools_desc)}

User query: {nl}

Analyze the query and produce a JSON plan. Think about:
1. What is the user asking for?
2. What tools do I need and in what order?
3. Are there any variable dependencies I need to satisfy?
4. What variables should be in the output?

Output your plan as JSON:"""

    def _parse_response(self, raw: str) -> AgentPlan:
        """Parse LLM response into an AgentPlan."""
        # Extract JSON from response
        raw = raw.strip()

        # Try to find JSON object — use a balanced-brace scan to avoid
        # greedily capturing text that follows the JSON block (which would
        # produce "Extra data" parse errors when the model appends commentary).
        start = raw.find('{')
        if start == -1:
            raise ValueError(f"No JSON found in LLM response: {raw[:200]}")
        depth = 0
        end = start
        for i, ch in enumerate(raw[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        json_str = raw[start:end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        # Parse steps
        steps = []
        for step_data in (data.get("steps") or []):
            steps.append(ToolCall(
                tool=step_data.get("tool", ""),
                params=step_data.get("params", {}),
            ))

        return AgentPlan(
            reasoning=data.get("reasoning", ""),
            steps=steps,
            output_vars=data.get("output_vars") or [],
            filters=data.get("filters") or [],
            aggregation=data.get("aggregation"),
        )

    def _resolve_block_id(
        self,
        step: ToolCall,
        tool_def: Dict[str, Any],
        known_vars: Set[str],
    ) -> Optional[str]:
        """Resolve the actual block ID for a tool call."""
        # Dynamic block selection for count_senses
        if step.tool == "count_senses":
            words = step.params.get("words", [])
            return "COMPLIT_COUNT_SENSES_SINGLE" if len(words) <= 1 else "COMPLIT_COUNT_SENSES_MULTI"

        # Direct block mapping
        if "block" in tool_def:
            return tool_def["block"]

        # Block variants based on available variables
        if "block_variants" in tool_def:
            variants = tool_def["block_variants"]
            source_var = step.params.get("source_var")
            if source_var and source_var in variants:
                return variants[source_var]
            # Auto-detect from known vars
            for var, block_id in variants.items():
                if var in known_vars:
                    return block_id

        return None

    def _build_slots(
        self,
        step: ToolCall,
        tool_def: Dict[str, Any],
        block_id: str,
    ) -> Dict[str, str]:
        """Build slot values for a block call."""
        slots: Dict[str, str] = {}
        tool_name = step.tool
        params = step.params

        # Handle specific tools
        if tool_name == "find_semantic_relations":
            seed = params.get("seed_word", "")
            rel_type = params.get("relation_type", "hyponym")
            pos = params.get("pos_filter")

            # Build relation triple
            rel_pred = self._get_relation_predicate(rel_type)

            # Determine direction based on relation type.
            # CompL-IT semantics: A lexinfo:hyponym B = "A is a hyponym of B"
            #   (A is more specific, B is more general)
            # - hyponym: ?senseRel lexinfo:hyponym ?seedSense
            #     → senseRel is more specific than seed → hyponym of seed ✓
            # - hypernym: ?seedSense lexinfo:hyponym ?senseRel
            #     → seed is more specific than senseRel → senseRel is hypernym of seed ✓
            # - meronym: ?senseRel lexinfo:partMeronym ?seedSense
            #     → senseRel is a part of seed ✓
            # - synonym/antonym: symmetric (UNION both directions)
            if rel_type == "synonym" or rel_type == "antonym":
                # Symmetric - use both directions
                slots["rel_triple"] = (
                    f"{{ ?seedSense {rel_pred} ?senseRel . }} UNION "
                    f"{{ ?senseRel {rel_pred} ?seedSense . }}"
                )
            elif rel_type == "hypernym":
                # seed is a hyponym of senseRel → senseRel is more general = hypernym
                slots["rel_triple"] = f"?seedSense {rel_pred} ?senseRel ."
            else:
                # hyponym / meronym / holonym: rel_to_seed direction
                slots["rel_triple"] = f"?senseRel {rel_pred} ?seedSense ."

            slots["seed_lemma"] = sparql_quote(seed)
            pos = self._normalize_pos(pos) if pos else pos
            slots["seed_pos_filter"] = self._build_pos_filter(pos, "complit") if pos else ""
            slots["rel_extra"] = ""

        elif tool_name == "find_definitions_by_pattern":
            ptype = params.get("pattern_type", "prefix")
            ptext = params.get("pattern_text", "")
            apply_to = params.get("apply_to", "definition")
            pos = params.get("pos_filter")

            if apply_to == "lemma":
                slots["lemma_filter"] = build_filter_for_var("?itLemmaString", ptype, ptext)
                slots["def_filter"] = ""
            else:
                slots["def_filter"] = build_filter_for_var("?definition", ptype, ptext)
                slots["lemma_filter"] = ""

            pos = self._normalize_pos(pos) if pos else pos
            slots["pos_filter"] = self._build_pos_filter(pos, "complit") if pos else ""

        elif tool_name in ("find_liita_lemmas_by_pattern", "find_sicilian_lemmas_by_pattern", "find_parmigiano_lemmas_by_pattern"):
            ptype = params.get("pattern_type", "prefix")
            ptext = params.get("pattern_text", "")
            pos_raw = params.get("pos_filter")
            pos = self._normalize_pos(pos_raw) if pos_raw else None

            # Determine the variable to filter
            if tool_name == "find_liita_lemmas_by_pattern":
                wr_var = "?wr"
                lemma_var = "?lemma"
            elif tool_name == "find_sicilian_lemmas_by_pattern":
                wr_var = "?sicWR"
                lemma_var = "?sicLemma"
            else:
                wr_var = "?parWR"
                lemma_var = "?parLemma"

            # Only add filter if pattern_text is non-empty
            slots["wr_filter"] = build_filter_for_var(wr_var, ptype, ptext) if ptext else ""
            slots["pos_clause"] = f"{lemma_var} lila:hasPOS lila:{pos} ." if pos else ""

        elif tool_name == "count_senses":
            words = params.get("words", [])
            if len(words) == 1:
                slots["seed_lemma"] = sparql_quote(words[0])
            elif len(words) > 1:
                regex_alt = "|".join(re.escape(w) for w in words)
                slots["wr_regex_filter"] = f'FILTER(regex(str(?writtenRep), "^({regex_alt})$", "i")) .'

        elif tool_name == "get_emotions":
            emotions = params.get("emotions", [])
            if emotions:
                # Build emotion filter
                iris = [EMOTION_MAP.get(e.lower(), f"elita:{e}") for e in emotions if e]
                if iris:
                    slots["emotion_filter_clause"] = f"VALUES ?emotion {{ {' '.join(iris)} }}"
                else:
                    slots["emotion_filter_clause"] = ""
            else:
                slots["emotion_filter_clause"] = ""

        # Fill in any missing required slots with empty strings
        if block_id in self.registry.blocks:
            block = self.registry.get(block_id)
            for line in block.where:
                for match in re.finditer(r'\{(\w+)\}', line):
                    slot_name = match.group(1)
                    if slot_name not in slots:
                        slots[slot_name] = ""

        return slots

    def _get_relation_predicate(self, rel_type: str) -> str:
        """Get the SPARQL predicate for a relation type."""
        mapping = {
            # CompL-IT stores hyponymy as A lexinfo:hyponym B meaning
            # "A is a hyponym of B" (A is more specific, B is more general).
            # hyponym: ?senseRel lexinfo:hyponym ?seedSense  (senseRel more specific than seed)
            # hypernym: ?seedSense lexinfo:hyponym ?senseRel (seed more specific than senseRel)
            "hyponym": "lexinfo:hyponym",
            "hypernym": "lexinfo:hyponym",
            "synonym": "lexinfo:approximateSynonym",
            "antonym": "lexinfo:antonym",
            "meronym": "lexinfo:partMeronym",
            "holonym": "lexinfo:partHolonym",
        }
        return mapping.get(rel_type, "lexinfo:hyponym")

    # Accepted LLM variants → canonical LiLA/CompL-IT POS string
    _POS_NORMALIZE: Dict[str, str] = {
        "noun": "noun", "nouns": "noun", "n": "noun",
        "verb": "verb", "verbs": "verb", "v": "verb",
        "adjective": "adjective", "adjectives": "adjective", "adj": "adjective",
        "adverb": "adverb", "adverbs": "adverb", "adv": "adverb",
        "pronoun": "pronoun", "pronouns": "pronoun", "pron": "pronoun",
        "preposition": "preposition", "prep": "preposition",
        "conjunction": "conjunction", "conj": "conjunction",
        "interjection": "interjection", "interj": "interjection",
        "numeral": "numeral", "num": "numeral",
    }

    def _normalize_pos(self, pos: str) -> str:
        """Normalise LLM-supplied POS values to canonical lowercase LiLA form."""
        return self._POS_NORMALIZE.get(pos.lower(), pos.lower())

    def _build_pos_filter(self, pos: str, source: str) -> str:
        """Build a POS filter clause."""
        if not pos:
            return ""

        if source == "complit":
            return f'FILTER(str(?posLabel) = "{pos}") .'
        else:  # liita
            return f"?lemma lila:hasPOS lila:{pos} ."

    def _build_output_spec(
        self,
        plan: AgentPlan,
        known_vars: Set[str],
        effective_aggregation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], Dict[str, str], List[str]]:
        """Build SELECT, aggregates, and GROUP BY from the plan."""
        select_vars: List[str] = []
        aggregates: Dict[str, str] = {}
        group_by: List[str] = []

        # Check for aggregation (use effective_aggregation which may come from tool steps)
        if effective_aggregation:
            agg_fn = effective_aggregation.get("agg_function", "COUNT")
            agg_var = effective_aggregation.get("agg_variable", "?lemma")
            distinct = effective_aggregation.get("distinct", agg_fn == "COUNT")
            xsd_cast = effective_aggregation.get("xsd_cast")
            group_vars = effective_aggregation.get("group_by", [])

            # Build the inner expression
            inner_var = agg_var
            if xsd_cast:
                inner_var = f"xsd:{xsd_cast}({agg_var})"
            distinct_kw = "DISTINCT " if distinct else ""
            agg_expr = f"{agg_fn}({distinct_kw}{inner_var})"

            # Alias: ?avgPolarityValue for AVG on ?polarityValue, ?count for COUNT
            if agg_fn == "AVG":
                # Derive alias from variable name: ?polarityValue -> ?avgPolarityValue
                var_name = agg_var.lstrip("?")
                alias = f"?avg{var_name[0].upper()}{var_name[1:]}"
            else:
                alias = "?count"
            aggregates[alias] = agg_expr

            select_vars = [v for v in group_vars if v in known_vars]
            group_by = list(select_vars)
        else:
            # Use plan's output vars, filtered to what's available
            for var in plan.output_vars:
                if var in known_vars:
                    select_vars.append(var)

            # Add common useful vars only when the LLM didn't specify output_vars.
            # If output_vars was given, trust it — don't silently inject extra columns
            # (e.g., ?itLemmaString would break pure semantic-relation queries).
            if not select_vars:
                # ?wordRel must come before ?itLemmaString so positional variable
                # mapping prefers the CompL-IT IRI (matches gold hypernymWord etc.)
                priority_vars = ["?wordRel", "?itLemmaString", "?lemma", "?wr", "?liitaLemma"]
                for var in priority_vars:
                    if var in known_vars and var not in select_vars:
                        select_vars.insert(0, var)
            elif "?wr" in known_vars and ("?parWR" in known_vars or "?sicWR" in known_vars):
                # Translation query: always prepend the Italian written rep so the
                # evaluator can correctly pair (italianWord, dialectWord) tuples.
                # Without ?wr, it maps gold's italianWord → parWR/sicWR (wrong language)
                # and gets 0 true positives even when counts match exactly.
                if "?wr" not in select_vars:
                    select_vars.insert(0, "?wr")

            # Add definition aggregate only when explicitly requested.
            # Unconditionally adding it converts simple SELECT queries into
            # GROUP BY queries and creates columns absent from gold queries.
            if "?definition" in known_vars and "?definition" in plan.output_vars:
                aggregates["?definitionSample"] = "SAMPLE(?definition)"
            if "?sicWR" in known_vars and "?sicWR" not in select_vars:
                aggregates["?sicilianoWRs"] = 'GROUP_CONCAT(DISTINCT ?sicWR; SEPARATOR=", ")'
            if "?parWR" in known_vars and "?parWR" not in select_vars:
                aggregates["?parmigianoWRs"] = 'GROUP_CONCAT(DISTINCT ?parWR; SEPARATOR=", ")'
            if "?polarityLabel" in known_vars:
                aggregates["?polarityLabels"] = 'GROUP_CONCAT(DISTINCT ?polarityLabel; SEPARATOR=", ")'
            if "?emotionLabel" in known_vars:
                aggregates["?emotions"] = 'GROUP_CONCAT(DISTINCT ?emotionLabel; SEPARATOR=", ")'

            # GROUP BY all non-aggregate select vars if we have aggregates
            if aggregates:
                group_by = [v for v in select_vars if v.startswith("?")]

        # Dedupe
        select_vars = list(dict.fromkeys(select_vars))
        group_by = list(dict.fromkeys(group_by))

        return select_vars, aggregates, group_by
