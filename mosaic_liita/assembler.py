"""
SPARQL query assembler for MoSAIC-LiITA.

The Assembler takes a compiled QueryPlan and generates the final SPARQL query string.
"""

from __future__ import annotations

from typing import List, Set

from .constants import PREFIXES
from .query import QueryPlan


class Assembler:
    """
    Assembles a QueryPlan into a SPARQL query string.

    The assembler:
    - Merges prefixes from all blocks
    - Builds the SELECT clause with variables and aggregates
    - Combines WHERE clauses from all blocks
    - Adds GROUP BY, HAVING, ORDER BY, and LIMIT clauses
    """

    def __init__(self) -> None:
        pass

    def assemble(self, plan: QueryPlan) -> str:
        """
        Assemble a QueryPlan into a SPARQL query string.

        Args:
            plan: Compiled QueryPlan with ordered blocks

        Returns:
            Complete SPARQL query string
        """
        # Merge prefixes from all blocks
        needed_prefixes: Set[str] = set()
        where_lines: List[str] = []

        for bi in plan.blocks:
            needed_prefixes |= bi.prefixes
            where_lines.extend(bi.render_where())

        # Detect prefixes used in aggregate expressions (e.g., xsd:float)
        for expr in plan.aggregates.values():
            for pfx in PREFIXES:
                if f"{pfx}:" in expr:
                    needed_prefixes.add(pfx)

        # Build PREFIX declarations
        prefix_lines = []
        for pfx in sorted(needed_prefixes):
            if pfx not in PREFIXES:
                raise KeyError(f"Unknown prefix key: {pfx}")
            prefix_lines.append(f"PREFIX {pfx}: <{PREFIXES[pfx]}>")

        # Build SELECT clause
        select_parts = []
        select_parts.extend(plan.select_vars)
        for alias, expr in plan.aggregates.items():
            select_parts.append(f"({expr} AS {alias})")

        select_clause = "SELECT " + " ".join(select_parts)

        # Build GROUP BY clause
        group_by_clause = ""
        having_clause = plan.having or ""
        if plan.aggregates:
            if plan.select_vars and not plan.group_by:
                raise ValueError("Aggregates present but group_by is empty.")
            if plan.group_by:
                group_by_clause = "GROUP BY " + " ".join(plan.group_by)

        order_by_clause = plan.order_by or ""
        limit_clause = f"LIMIT {plan.limit}" if plan.limit else ""

        # Assemble the full query
        query = "\n".join(prefix_lines) + "\n\n" + select_clause + "\nWHERE {\n"
        query += "\n".join("  " + ln for ln in where_lines)
        query += "\n}\n"
        if group_by_clause:
            query += group_by_clause + "\n"
        if having_clause:
            query += having_clause + "\n"
        if order_by_clause:
            query += order_by_clause + "\n"
        if limit_clause:
            query += limit_clause + "\n"

        return query
