"""
Query specification and plan model for MoSAIC-LiITA.

This module provides:
- QuerySpec: High-level query specification (blocks + output shape)
- QueryPlan: Compiled query ready for assembly
- Validation functions for ensuring query correctness
- Topological sorting for dependency ordering
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .blocks import BlockRegistry

from .blocks import BlockCall, BlockInstance


class QuerySpecError(ValueError):
    """Exception raised for invalid query specifications."""
    pass


# Regex for validating SPARQL variables
_VAR_RE = re.compile(r"^\?[A-Za-z_][A-Za-z0-9_]*$")

# Conservative aggregate allowlist
# Supports: COUNT(?var), AVG(DISTINCT ?var), AVG(xsd:float(?var)), GROUP_CONCAT(...)
_ALLOWED_AGG_RE = re.compile(
    r"""^
    (COUNT|SAMPLE|MIN|MAX|AVG|SUM)\(\s*(DISTINCT\s+)?
        (?:xsd:(?:float|integer|double)\()?\?[A-Za-z_][A-Za-z0-9_]*(?:\))?\s*\)
    |
    GROUP_CONCAT\(\s*(DISTINCT\s+)?\?[A-Za-z_][A-Za-z0-9_]*\s*;\s*SEPARATOR\s*=\s*"[^"]*"\s*\)
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Detect unexpanded format braces
_UNSAFE_BRACES_RE = re.compile(r"[{}]")


@dataclass
class QueryPlan:
    """Compiled query plan with ordered block instances."""
    blocks: List[BlockInstance]
    select_vars: List[str]
    aggregates: Dict[str, str]  # alias -> expr
    group_by: List[str]
    having: Optional[str] = None
    order_by: Optional[str] = None
    limit: Optional[int] = None


@dataclass
class QuerySpec:
    """
    High-level query specification.

    This is the input format for building queries. It contains:
    - blocks: List of BlockCall references with slot values
    - Output shape (select_vars, aggregates, group_by)
    - Optional clauses (having, order_by, limit)
    """
    # Core structure
    blocks: List[BlockCall]

    # Output shape
    select_vars: List[str] = field(default_factory=list)
    aggregates: Dict[str, str] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)

    # Optional clauses
    having: Optional[str] = None
    order_by: Optional[str] = None
    limit: Optional[int] = 200

    # Metadata
    intent: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def compile(self, registry: "BlockRegistry", *, toposort: bool = True) -> QueryPlan:
        """
        Turn the spec into a QueryPlan by instantiating BlockInstances
        and ordering them by dependency (requires/provides).
        """
        validate_queryspec(self, registry)

        instances: List[BlockInstance] = []
        for bc in self.blocks:
            block = registry.get(bc.block_id)
            instances.append(BlockInstance(block=block, slots=dict(bc.slots)))

        if toposort:
            instances = toposort_blocks(instances)

        return QueryPlan(
            blocks=instances,
            select_vars=list(self.select_vars),
            aggregates=dict(self.aggregates),
            group_by=list(self.group_by),
            having=self.having,
            order_by=self.order_by,
            limit=self.limit,
        )

    @staticmethod
    def from_plan(plan: QueryPlan) -> "QuerySpec":
        """Convert a QueryPlan back to a QuerySpec."""
        blocks = [
            BlockCall(block_id=bi.block.id, slots=dict(bi.slots))
            for bi in plan.blocks
        ]
        return QuerySpec(
            blocks=blocks,
            select_vars=list(plan.select_vars),
            aggregates=dict(plan.aggregates),
            group_by=list(plan.group_by),
            having=plan.having,
            order_by=plan.order_by,
            limit=plan.limit,
        )


def extract_placeholders(where_lines: List[str]) -> Set[str]:
    """Extract {name} placeholders from Block.where lines."""
    placeholders: Set[str] = set()
    for ln in where_lines:
        for m in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", ln):
            placeholders.add(m.group(1))
    return placeholders


def validate_queryspec(spec: QuerySpec, registry: "BlockRegistry") -> None:
    """
    Validate a QuerySpec for correctness.

    Checks:
    1. All blocks exist in registry
    2. Slot values are safe strings
    3. Select vars are valid SPARQL variables
    4. Aggregates match allowlist
    5. GROUP BY is consistent with aggregates
    6. HAVING/ORDER BY have correct prefixes
    7. LIMIT is in valid range
    """
    # 1) Blocks must exist; slots must match placeholders.
    for bc in spec.blocks:
        if bc.block_id not in registry.blocks:
            raise QuerySpecError(f"Unknown block_id: {bc.block_id}")

        block = registry.get(bc.block_id)
        required_placeholders = extract_placeholders(block.where)

        # Slots that intentionally contain SPARQL fragments
        ALLOW_BRACES_SLOTS = {
            "rel_triple", "rel_extra",
            "pos_clause",
            "wr_filter", "def_filter", "lemma_filter",
            "emotion_filter_clause",
        }

        for k, v in bc.slots.items():
            if not isinstance(v, str):
                raise QuerySpecError(f"Slot value for '{bc.block_id}.{k}' must be str, got {type(v)}")
            if _UNSAFE_BRACES_RE.search(v) and k not in ALLOW_BRACES_SLOTS:
                raise QuerySpecError(
                    f"Slot value for '{bc.block_id}.{k}' contains '{{' or '}}' which is unsafe for str.format()."
                )

        unknown = set(bc.slots.keys()) - required_placeholders
        if unknown:
            raise QuerySpecError(
                f"Block '{bc.block_id}' got unknown slots {sorted(unknown)}; "
                f"allowed: {sorted(required_placeholders)}"
            )

        missing = required_placeholders - set(bc.slots.keys())
        if missing:
            raise QuerySpecError(
                f"Block '{bc.block_id}' missing required slots {sorted(missing)}"
            )

    # 2) select_vars must be SPARQL vars
    for v in spec.select_vars:
        if not _VAR_RE.match(v):
            raise QuerySpecError(f"Invalid select var: {v}")

    # 3) aggregates: alias must be a var; expr must match allowlist
    for alias, expr in spec.aggregates.items():
        if not _VAR_RE.match(alias):
            raise QuerySpecError(f"Aggregate alias must be a SPARQL var like '?n': {alias}")
        if not isinstance(expr, str) or not expr.strip():
            raise QuerySpecError(f"Aggregate expr for {alias} must be non-empty str")
        if not _ALLOWED_AGG_RE.match(expr.strip()):
            raise QuerySpecError(
                f"Aggregate expr not allowed (too permissive is dangerous): {expr}"
            )

    # 4) GROUP BY consistency
    if spec.aggregates:
        if spec.select_vars and not spec.group_by:
            raise QuerySpecError("Aggregates present and select_vars is non-empty, but group_by is empty.")

        for v in spec.group_by:
            if not _VAR_RE.match(v):
                raise QuerySpecError(f"Invalid GROUP BY var: {v}")
            if v not in spec.select_vars:
                raise QuerySpecError(f"GROUP BY var {v} must be in select_vars")

    # 5) HAVING / ORDER BY format
    if spec.having is not None:
        if not spec.having.strip().upper().startswith("HAVING "):
            raise QuerySpecError("having must start with 'HAVING '")
    if spec.order_by is not None:
        if not spec.order_by.strip().upper().startswith("ORDER BY "):
            raise QuerySpecError("order_by must start with 'ORDER BY '")

    # 6) LIMIT sanity
    if spec.limit is not None:
        if not isinstance(spec.limit, int) or spec.limit <= 0 or spec.limit > 5000:
            raise QuerySpecError("limit must be an int in [1, 5000] or None")


def toposort_blocks(instances: List[BlockInstance]) -> List[BlockInstance]:
    """
    Topologically sort block instances by dependency.

    A block can be scheduled once all its 'requires' are present in the
    known vars set. Known vars are the union of provides of already
    scheduled blocks.
    """
    remaining = instances[:]
    ordered: List[BlockInstance] = []
    known: Set[str] = set()

    while remaining:
        progressed = False
        for i, bi in enumerate(list(remaining)):
            if bi.requires <= known:
                ordered.append(bi)
                known |= bi.provides
                remaining.pop(i)
                progressed = True
                break
        if not progressed:
            blockers = [(bi.block.id, sorted(bi.requires - known)) for bi in remaining]
            raise QuerySpecError(f"Cannot satisfy block dependencies; missing vars: {blockers}")

    return ordered
