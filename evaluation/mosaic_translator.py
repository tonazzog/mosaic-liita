"""MoSAIC-LiITA translator adapter for F1 evaluation.

This module provides:
- deaggregate_sparql(): SPARQL post-processor that removes GROUP_CONCAT
  aggregation so that predicted queries return individual rows comparable
  to gold standard queries.
- MosaicTranslatorAdapter: wraps either Planner (deterministic) or
  QueryAgent (agentic) and exposes the translate(question) -> SimpleNamespace
  interface expected by F1Evaluator.

The de-aggregation is critical because the deterministic planner groups
results with GROUP_CONCAT for dialect translation queries, while the gold
standard returns individual rows.

Known limitation: When the planner selects ?liitaLemma (URI) instead of
?wr (written representation) for Italian words, F1 for ?italianWord will
be 0 regardless of de-aggregation, because URIs don't match strings.
This is documented and considered out of scope for this evaluation module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional


# ---------------------------------------------------------------------------
# De-aggregation spec
# ---------------------------------------------------------------------------

@dataclass
class DeaggregationSpec:
    """Describes a single GROUP_CONCAT or SAMPLE pattern to de-aggregate.

    Attributes:
        alias: The aggregate alias variable name (without ?)
        raw_var: The raw variable to replace it with (without ?)
        pattern: Compiled regex matching the full aggregate expression in SELECT
    """
    alias: str
    raw_var: str
    pattern: re.Pattern


# Known aggregate expressions and their de-aggregated replacements.
# Each pattern matches the full "(... AS ?alias)" expression in the SELECT clause.
_DEAGG_SPECS: List[DeaggregationSpec] = [
    DeaggregationSpec(
        alias="sicilianoWRs",
        raw_var="sicWR",
        pattern=re.compile(
            r'\(\s*GROUP_CONCAT\s*\(\s*DISTINCT\s+\?sicWR\s*;\s*SEPARATOR\s*=\s*"[^"]*"\s*\)\s*AS\s+\?sicilianoWRs\s*\)',
            re.IGNORECASE,
        ),
    ),
    DeaggregationSpec(
        alias="parmigianoWRs",
        raw_var="parWR",
        pattern=re.compile(
            r'\(\s*GROUP_CONCAT\s*\(\s*DISTINCT\s+\?parWR\s*;\s*SEPARATOR\s*=\s*"[^"]*"\s*\)\s*AS\s+\?parmigianoWRs\s*\)',
            re.IGNORECASE,
        ),
    ),
    DeaggregationSpec(
        alias="emotions",
        raw_var="emotionLabel",
        pattern=re.compile(
            r'\(\s*GROUP_CONCAT\s*\(\s*DISTINCT\s+\?emotionLabel\s*;\s*SEPARATOR\s*=\s*"[^"]*"\s*\)\s*AS\s+\?emotions\s*\)',
            re.IGNORECASE,
        ),
    ),
    DeaggregationSpec(
        alias="polarityLabels",
        raw_var="polarityLabel",
        pattern=re.compile(
            r'\(\s*GROUP_CONCAT\s*\(\s*DISTINCT\s+\?polarityLabel\s*;\s*SEPARATOR\s*=\s*"[^"]*"\s*\)\s*AS\s+\?polarityLabels\s*\)',
            re.IGNORECASE,
        ),
    ),
    DeaggregationSpec(
        alias="polarityValues",
        raw_var="polarityValue",
        pattern=re.compile(
            r'\(\s*GROUP_CONCAT\s*\(\s*DISTINCT\s+\?polarityValue\s*;\s*SEPARATOR\s*=\s*"[^"]*"\s*\)\s*AS\s+\?polarityValues\s*\)',
            re.IGNORECASE,
        ),
    ),
    DeaggregationSpec(
        alias="definitionSample",
        raw_var="definition",
        pattern=re.compile(
            r'\(\s*SAMPLE\s*\(\s*\?definition\s*\)\s*AS\s+\?definitionSample\s*\)',
            re.IGNORECASE,
        ),
    ),
]

# Regex matching a COUNT aggregate (we preserve these)
_COUNT_PATTERN = re.compile(
    r'\(\s*COUNT\s*\(',
    re.IGNORECASE,
)

# Regex matching the entire GROUP BY clause line(s)
_GROUP_BY_PATTERN = re.compile(
    r'\bGROUP\s+BY\b[^\n]*(?:\n(?!\s*(?:HAVING|ORDER|LIMIT|OFFSET|\})).*)*',
    re.IGNORECASE,
)

# Matches the SELECT clause body (between SELECT [DISTINCT] and WHERE/FROM/{)
_SELECT_CLAUSE_PATTERN = re.compile(
    r'\bSELECT\b(\s+DISTINCT)?(.*?)\s+(?=WHERE\b|FROM\b|\{)',
    re.IGNORECASE | re.DOTALL,
)


def _standalone_vars_in_select(sparql: str) -> set[str]:
    """Return variable names that appear as plain ?var tokens at SELECT depth 0.

    Variables inside aggregate expressions like (GROUP_CONCAT(?x ...) AS ?y)
    are NOT included — only top-level tokens are returned.
    """
    select_match = _SELECT_CLAUSE_PATTERN.search(sparql)
    if not select_match:
        return set()

    clause = select_match.group(2)
    vars_found: set[str] = set()
    depth = 0
    i = 0
    while i < len(clause):
        ch = clause[i]
        if ch == '(':
            depth += 1
            i += 1
        elif ch == ')':
            depth -= 1
            i += 1
        elif ch == '?' and depth == 0:
            j = i + 1
            while j < len(clause) and (clause[j].isalnum() or clause[j] == '_'):
                j += 1
            vars_found.add(clause[i + 1:j])
            i = j
        else:
            i += 1
    return vars_found


def deaggregate_sparql(sparql: str) -> str:
    """Remove GROUP_CONCAT and SAMPLE aggregation from a MoSAIC SPARQL query.

    Transforms the query to return individual rows instead of grouped
    concatenated values, making it comparable to gold standard queries.

    The transformation:
    1. For each known GROUP_CONCAT/SAMPLE alias:
       - If the raw variable is already a standalone SELECT variable,
         remove the aggregate expression entirely (no duplicate introduced).
       - Otherwise, replace the aggregate expression with the raw variable.
    2. Removes GROUP BY (unless a COUNT aggregate is still present)

    COUNT aggregates and their GROUP BY variables are preserved.

    Args:
        sparql: Input SPARQL query (possibly with GROUP_CONCAT aggregation)

    Returns:
        De-aggregated SPARQL query with individual-row semantics
    """
    result = sparql

    # Apply each de-aggregation rule
    any_replaced = False
    for spec in _DEAGG_SPECS:
        if not spec.pattern.search(result):
            continue

        # If the raw variable is already present as a standalone SELECT var,
        # just remove the aggregate expression to avoid duplicating it.
        existing = _standalone_vars_in_select(result)
        if spec.raw_var in existing:
            replacement = ""
        else:
            replacement = f"?{spec.raw_var}"

        new_result = spec.pattern.sub(replacement, result)
        if new_result != result:
            any_replaced = True
            result = new_result

    # If nothing was replaced, the query has no aggregation — return as-is
    if not any_replaced:
        return result

    # Clean up whitespace left by empty substitutions (e.g. "?x  ?y" or "?x \n?y")
    result = re.sub(r'[ \t]{2,}', ' ', result)

    # Check whether any COUNT aggregates remain
    has_count = bool(_COUNT_PATTERN.search(result))

    if not has_count:
        # No COUNT remaining — remove GROUP BY entirely
        result = _GROUP_BY_PATTERN.sub("", result)
        # Clean up extra blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MosaicTranslatorAdapter:
    """Adapts MoSAIC's Planner or QueryAgent to the F1Evaluator interface.

    F1Evaluator expects translator.translate(question) to return an object
    with a .sparql attribute. This adapter wraps either the deterministic
    Planner pipeline or the LLM-powered QueryAgent and optionally applies
    SPARQL de-aggregation before returning.

    Args:
        registry: BlockRegistry (from make_registry())
        catalog: Ontology catalog list (from ontology_filtered.json)
        mode: "deterministic" (default) or "agentic"
        llm_client: Required when mode="agentic"
        deaggregate: Whether to apply SPARQL de-aggregation (default True)
        verbose: Print generated SPARQL for debugging
    """

    def __init__(
        self,
        registry,
        catalog: list,
        mode: str = "deterministic",
        llm_client=None,
        deaggregate: bool = True,
        verbose: bool = False,
    ) -> None:
        self._registry = registry
        self._catalog = catalog
        self._mode = mode
        self._deaggregate = deaggregate
        self._verbose = verbose

        if mode == "deterministic":
            from mosaic_liita.planner import Planner
            from mosaic_liita.assembler import Assembler
            self._planner = Planner(registry, catalog)
            self._assembler = Assembler()
            self._agent = None
        elif mode == "agentic":
            if llm_client is None:
                raise ValueError("llm_client is required for agentic mode")
            from mosaic_liita.agent import QueryAgent
            self._agent = QueryAgent(registry, catalog, llm_client)
            self._planner = None
            self._assembler = None
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'deterministic' or 'agentic'.")

    def translate(self, question: str) -> SimpleNamespace:
        """Translate a natural language question to a SPARQL query.

        Args:
            question: Natural language question string

        Returns:
            SimpleNamespace with .sparql attribute containing the SPARQL query

        Raises:
            RuntimeError: If translation fails
        """
        try:
            if self._mode == "deterministic":
                sparql = self._translate_deterministic(question)
            else:
                sparql = self._translate_agentic(question)
        except Exception as e:
            raise RuntimeError(f"MoSAIC translation failed: {e}") from e

        if self._deaggregate:
            sparql = deaggregate_sparql(sparql)

        if self._verbose:
            print(f"\n[MoSAIC] Q: {question}")
            print(f"[MoSAIC] SPARQL:\n{sparql}\n")

        return SimpleNamespace(sparql=sparql)

    def _translate_deterministic(self, question: str) -> str:
        """Run the deterministic planner pipeline."""
        spec = self._planner.plan(question)
        query_plan = spec.compile(self._registry)
        return self._assembler.assemble(query_plan)

    def _translate_agentic(self, question: str) -> str:
        """Run the LLM-powered agent pipeline."""
        sparql, _plan, _spec = self._agent.translate(question)
        return sparql
