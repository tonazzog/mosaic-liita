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
            "pattern_type": {"type": "string", "enum": ["prefix", "suffix", "contains"], "description": "How to match the pattern", "required": True},
            "pattern_text": {"type": "string", "description": "The text pattern to match", "required": True},
            "pos_filter": {"type": "string", "enum": ["noun", "verb", "adjective", "adverb", None], "description": "Optional part-of-speech filter (uses LiITA POS)", "required": False},
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

    "get_sentiment": {
        "description": "Get sentiment polarity (positive/negative/neutral) from Sentix.",
        "block": "SENTIX_POLARITY",
        "parameters": {},
        "provides": ["?polarityLabel", "?polarityValue"],
        "requires": ["?liitaLemma"],
    },

    "get_emotions": {
        "description": "Get emotion annotations from ELIta. Can filter to specific emotions.",
        "block": "ELITA_EMOTION_FILTER",
        "parameters": {
            "emotions": {"type": "array", "items": "string", "description": "Optional list of emotions to filter (e.g., ['gioia', 'tristezza']). Leave empty for all emotions.", "required": False},
        },
        "provides": ["?emotionLabel"],
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

    "count_results": {
        "description": "Count the results instead of listing them. Use for 'how many' queries.",
        "type": "aggregation",
        "parameters": {
            "count_variable": {"type": "string", "description": "Variable to count (e.g., ?lemma, ?liitaLemma)", "required": True},
            "group_by": {"type": "array", "items": "string", "description": "Optional variables to group by (e.g., ['?pos'] for count by POS)", "required": False},
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
                # Capture aggregation parameters from the tool call
                captured_aggregation = {
                    "count_variable": step.params.get("count_variable", "?lemma"),
                    "group_by": step.params.get("group_by", []),
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

        # Determine order_by
        order_by = None
        if "?itLemmaString" in select_vars:
            order_by = "ORDER BY ?itLemmaString"
        elif "?count" in aggregates and group_by:
            order_by = "ORDER BY DESC(?count)"

        return QuerySpec(
            blocks=blocks,
            select_vars=select_vars,
            aggregates=aggregates,
            group_by=group_by,
            order_by=order_by,
            limit=None if aggregates else 200,
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

        # Try to find JSON object
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            raise ValueError(f"No JSON found in LLM response: {raw[:200]}")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        # Parse steps
        steps = []
        for step_data in data.get("steps", []):
            steps.append(ToolCall(
                tool=step_data.get("tool", ""),
                params=step_data.get("params", {}),
            ))

        return AgentPlan(
            reasoning=data.get("reasoning", ""),
            steps=steps,
            output_vars=data.get("output_vars", []),
            filters=data.get("filters", []),
            aggregation=data.get("aggregation"),
        )

    def _resolve_block_id(
        self,
        step: ToolCall,
        tool_def: Dict[str, Any],
        known_vars: Set[str],
    ) -> Optional[str]:
        """Resolve the actual block ID for a tool call."""
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

            # Determine direction based on relation type
            # - hyponym: seed -> rel (seed has rel as hyponym)
            # - hypernym: rel -> seed (rel has seed as hyponym, i.e., seed is hypernym of rel)
            # - synonym/antonym: symmetric (use UNION)
            # - meronym: rel -> seed (rel is part of seed)
            # - holonym: rel -> seed (rel has seed as part, i.e., rel contains seed)
            if rel_type == "synonym" or rel_type == "antonym":
                # Symmetric - use both directions
                slots["rel_triple"] = (
                    f"{{ ?seedSense {rel_pred} ?senseRel . }} UNION "
                    f"{{ ?senseRel {rel_pred} ?seedSense . }}"
                )
            elif rel_type in ("hypernym", "meronym", "holonym"):
                # rel_to_seed direction
                slots["rel_triple"] = f"?senseRel {rel_pred} ?seedSense ."
            else:
                # Default (hyponym): seed_to_rel direction
                slots["rel_triple"] = f"?seedSense {rel_pred} ?senseRel ."

            slots["seed_lemma"] = sparql_quote(seed)
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

            slots["pos_filter"] = self._build_pos_filter(pos, "complit") if pos else ""

        elif tool_name in ("find_liita_lemmas_by_pattern", "find_sicilian_lemmas_by_pattern", "find_parmigiano_lemmas_by_pattern"):
            ptype = params.get("pattern_type", "prefix")
            ptext = params.get("pattern_text", "")
            pos = params.get("pos_filter")

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
            "hyponym": "lexinfo:hyponym",
            "hypernym": "lexinfo:hyponym",  # Same predicate, different direction
            "synonym": "lexinfo:approximateSynonym",
            "antonym": "lexinfo:antonym",
            "meronym": "lexinfo:partMeronym",
            "holonym": "lexinfo:partHolonym",
        }
        return mapping.get(rel_type, "lexinfo:hyponym")

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
            count_var = effective_aggregation.get("count_variable", "?lemma")
            group_vars = effective_aggregation.get("group_by", [])

            aggregates["?count"] = f"COUNT(DISTINCT {count_var})"
            select_vars = [v for v in group_vars if v in known_vars]
            group_by = list(select_vars)
        else:
            # Use plan's output vars, filtered to what's available
            for var in plan.output_vars:
                if var in known_vars:
                    select_vars.append(var)

            # Add common useful vars if available
            priority_vars = ["?itLemmaString", "?lemma", "?wr", "?liitaLemma"]
            for var in priority_vars:
                if var in known_vars and var not in select_vars:
                    select_vars.insert(0, var)

            # Add aggregates for multi-valued fields
            if "?definition" in known_vars:
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
