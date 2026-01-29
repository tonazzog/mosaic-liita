"""
LLM-based query refinement for MoSAIC-LiITA.

This module provides optional LLM-powered refinement of QuerySpec objects.
The refinement is constrained to safe operations (filter adjustments, optional
enrichments) and cannot modify core query structure.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Set

if TYPE_CHECKING:
    from shared.llm import BaseLLM

from .blocks import Block, BlockCall, BlockRegistry
from .query import QuerySpec, validate_queryspec


# Regex for JSON extraction
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

# Variable validation regex
_VAR_RE_LOCAL = re.compile(r"^\?[A-Za-z_][A-Za-z0-9_]*$")

# Aggregate expression validation regex
_ALLOWED_AGG_RE_LOCAL = re.compile(
    r"""^
    (COUNT|SAMPLE|MIN|MAX|AVG|SUM)\(\s*(DISTINCT\s+)?\?[A-Za-z_][A-Za-z0-9_]*\s*\)
    |
    GROUP_CONCAT\(\s*(DISTINCT\s+)?(?:\?[A-Za-z_][A-Za-z0-9_]*|STR\(\s*\?[A-Za-z_][A-Za-z0-9_]*\s*\))\s*;\s*SEPARATOR\s*=\s*"[^"]*"\s*\)
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON object from LLM response."""
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError("LLM did not return a JSON object.")
    return json.loads(m.group(0))


def _dedupe(xs: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _block_index(blocks: List[BlockCall], block_id: str) -> List[int]:
    """Find indices of blocks with given ID."""
    return [i for i, b in enumerate(blocks) if b.block_id == block_id]


def _registry_placeholders(registry: BlockRegistry, block_id: str) -> Set[str]:
    """Extract {slot} placeholders from a block's where clause."""
    b = registry.get(block_id)
    ph: Set[str] = set()
    for ln in b.where:
        for m in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", ln):
            ph.add(m.group(1))
    return ph


def _validate_patch_fields(patch: Dict[str, Any]) -> None:
    """Validate patch structure."""
    if not isinstance(patch, dict):
        raise ValueError("Patch must be a JSON object")
    for k in patch.keys():
        if k not in {"set", "edit_slots", "add_blocks", "remove_blocks", "notes"}:
            raise ValueError(f"Unexpected patch key: {k}")


def llm_refine_queryspec(
    *,
    nl: str,
    spec: QuerySpec,
    registry: BlockRegistry,
    llm_client: "BaseLLM",
    temperature: float = 0.0,
    max_tokens: int = 900,
    enable_block_add_remove: bool = True,
) -> QuerySpec:
    """
    Ask an LLM to refine a QuerySpec using a constrained JSON patch.

    The refinement is fail-closed: if the patch is invalid or fails validation,
    the original spec is returned unchanged.

    Args:
        nl: Original natural language query
        spec: QuerySpec to refine
        registry: Block registry for validation
        llm_client: LLM client for generating refinements
        temperature: LLM temperature (default 0.0 for determinism)
        max_tokens: Maximum tokens for LLM response
        enable_block_add_remove: Allow adding/removing optional blocks

    Returns:
        Refined QuerySpec, or original spec if refinement fails
    """

    # Blocks that are safe to add/remove (optional enrichments/translations)
    ADD_REMOVE_SAFE: Set[str] = {
        "SENTIX_POLARITY",
        "ELITA_EMOTION_FILTER",
        "TRANSLATE_TO_SICILIANO",
        "TRANSLATE_TO_PARMIGIANO",
        "ASSERT_LIITA_LEMMA_TYPE",
    }

    # Slots that are safe to edit (soft filters)
    EDITABLE_SLOTS: Dict[str, Set[str]] = {
        "COMPLIT_DEF_STARTS_WITH": {"pos_filter", "definition_prefix"},
        "COMPLIT_DEF_FILTER_BY_PATTERN": {"pos_filter", "lemma_filter", "def_filter"},
        "LIITA_LEMMA_FILTER_BY_PATTERN_AND_POS": {"pos_clause", "wr_filter"},
        "SICILIANO_LEMMA_FILTER_BY_PATTERN_AND_POS": {"pos_clause", "wr_filter"},
        "PARMIGIANO_LEMMA_FILTER_BY_PATTERN_AND_POS": {"pos_clause", "wr_filter"},
        "ELITA_EMOTION_FILTER": {"emotion_filter_clause"},
    }

    # Allowed top-level changes
    ALLOW_TOP_LEVEL = {"limit", "order_by", "having", "select_vars", "group_by", "aggregates"}

    # Build compact context for the LLM
    blocks_view = [{"block_id": b.block_id, "slots": b.slots} for b in spec.blocks]

    allowed_edits_view = {
        "editable_slots": {bid: sorted(list(slots)) for bid, slots in EDITABLE_SLOTS.items()},
        "add_remove_blocks": sorted(list(ADD_REMOVE_SAFE)) if enable_block_add_remove else [],
        "top_level": sorted(list(ALLOW_TOP_LEVEL)),
    }

    placeholders_view = {}
    for b in spec.blocks:
        try:
            placeholders_view[b.block_id] = sorted(list(_registry_placeholders(registry, b.block_id)))
        except Exception:
            placeholders_view[b.block_id] = []

    system = (
        "You refine a structured query plan (QuerySpec) by producing a JSON patch.\n"
        "DO NOT output SPARQL.\n"
        "Output ONLY one JSON object.\n"
        "Never change core joins/semantic-relation triple patterns. Only adjust filters, optional enrichments, and presentation.\n"
    )

    patch_schema = {
        "set": {
            "limit": "int|null",
            "order_by": 'string|null (must start with "ORDER BY ")',
            "having": 'string|null (must start with "HAVING ")',
            "select_vars": "list of variables like ?itLemmaString",
            "group_by": "list of variables like ?itLemmaString",
            "aggregates": "dict mapping ?alias -> aggregate expression (COUNT/SAMPLE/GROUP_CONCAT...)",
        },
        "edit_slots": [
            {"block_id": "ID", "slot": "slot_name", "value": "string"}
        ],
        "add_blocks": [
            {"block_id": "ID", "slots": {"slot_name": "string"}}
        ],
        "remove_blocks": ["ID"],
        "notes": {"reason": "string"}
    }

    prompt = f"""
Natural-language request:
{nl}

Current QuerySpec:
blocks={json.dumps(blocks_view, ensure_ascii=False, indent=2)}
select_vars={spec.select_vars}
aggregates={spec.aggregates}
group_by={spec.group_by}
having={spec.having}
order_by={spec.order_by}
limit={spec.limit}

Allowed edits:
{json.dumps(allowed_edits_view, ensure_ascii=False, indent=2)}

Placeholders available per block (for slot editing):
{json.dumps(placeholders_view, ensure_ascii=False, indent=2)}

Rules:
- Output ONLY a JSON object matching the patch schema below.
- Keep edits minimal.
- If you add aggregates and select_vars is non-empty, group_by must include all select_vars.
- Do NOT add or edit any slot not in the allowed list.
- Do NOT invent new block_ids.

Patch schema:
{json.dumps(patch_schema, ensure_ascii=False, indent=2)}
""".strip()

    # Call LLM
    try:
        raw = llm_client.complete(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        patch = _extract_json(raw)
        _validate_patch_fields(patch)
    except Exception:
        return spec

    refined = deepcopy(spec)

    # Apply patch (fail-closed on any illegal move)
    try:
        # 1) remove_blocks
        if enable_block_add_remove:
            for bid in (patch.get("remove_blocks") or []):
                if not isinstance(bid, str):
                    continue
                if bid not in ADD_REMOVE_SAFE:
                    continue
                refined.blocks = [b for b in refined.blocks if b.block_id != bid]

        # 2) add_blocks
        if enable_block_add_remove:
            for item in (patch.get("add_blocks") or []):
                if not isinstance(item, dict):
                    continue
                bid = item.get("block_id")
                if not isinstance(bid, str):
                    continue
                if bid not in ADD_REMOVE_SAFE:
                    continue
                if bid not in registry.blocks:
                    continue

                slots = item.get("slots") or {}
                if not isinstance(slots, dict):
                    slots = {}

                ph = _registry_placeholders(registry, bid)
                slots_clean: Dict[str, str] = {}
                for k, v in slots.items():
                    if isinstance(k, str) and k in ph:
                        slots_clean[k] = "" if v is None else str(v)

                refined.blocks.append(BlockCall(block_id=bid, slots=slots_clean))

        # 2.5) Ensure required joins for added enrichments
        available: Set[str] = set()
        for bc in refined.blocks:
            if bc.block_id in registry.blocks:
                available |= registry.get(bc.block_id).provides

        wants_liita = any(b.block_id in {
            "SENTIX_POLARITY",
            "ELITA_EMOTION_FILTER",
            "TRANSLATE_TO_SICILIANO",
            "TRANSLATE_TO_PARMIGIANO",
            "ASSERT_LIITA_LEMMA_TYPE",
        } for b in refined.blocks)

        has_liita = "?liitaLemma" in available

        if wants_liita and not has_liita:
            if "?word" in available and "JOIN_WORD_TO_LIITA_FROM_WORD" in registry.blocks:
                refined.blocks.append(BlockCall(block_id="JOIN_WORD_TO_LIITA_FROM_WORD"))
                available |= registry.get("JOIN_WORD_TO_LIITA_FROM_WORD").provides
            elif "?wordRel" in available and "JOIN_WORDREL_TO_LIITA" in registry.blocks:
                refined.blocks.append(BlockCall(block_id="JOIN_WORDREL_TO_LIITA"))
                available |= registry.get("JOIN_WORDREL_TO_LIITA").provides

        # 3) edit_slots
        for e in (patch.get("edit_slots") or []):
            if not isinstance(e, dict):
                continue
            bid = e.get("block_id")
            slot = e.get("slot")
            val = e.get("value")

            if not isinstance(bid, str) or not isinstance(slot, str):
                continue
            if bid not in EDITABLE_SLOTS or slot not in EDITABLE_SLOTS[bid]:
                continue

            idxs = _block_index(refined.blocks, bid)
            if not idxs:
                continue

            i = idxs[0]
            ph = _registry_placeholders(registry, bid)
            if slot not in ph:
                continue

            refined.blocks[i].slots[slot] = "" if val is None else str(val)

        # 4) set top-level fields
        set_obj = patch.get("set") or {}
        if not isinstance(set_obj, dict):
            set_obj = {}

        for k, v in set_obj.items():
            if k not in ALLOW_TOP_LEVEL:
                continue

            if k == "limit":
                refined.limit = v
            elif k == "order_by":
                refined.order_by = v
            elif k == "having":
                refined.having = v
            elif k == "select_vars":
                refined.select_vars = list(v or [])
            elif k == "group_by":
                refined.group_by = list(v or [])
            elif k == "aggregates":
                refined.aggregates = dict(v or {})

        # Normalize and validate
        refined.select_vars = _dedupe([x for x in refined.select_vars if isinstance(x, str)])
        refined.group_by = _dedupe([x for x in refined.group_by if isinstance(x, str)])

        if refined.order_by is not None:
            if not isinstance(refined.order_by, str) or not refined.order_by.strip().upper().startswith("ORDER BY "):
                refined.order_by = spec.order_by

        if refined.having is not None:
            if not isinstance(refined.having, str) or not refined.having.strip().upper().startswith("HAVING "):
                refined.having = spec.having

        refined.select_vars = [v for v in refined.select_vars if _VAR_RE_LOCAL.match(v)]
        refined.group_by = [v for v in refined.group_by if _VAR_RE_LOCAL.match(v)]

        if refined.aggregates:
            clean_aggs: Dict[str, str] = {}
            for alias, expr in refined.aggregates.items():
                if not isinstance(alias, str) or not _VAR_RE_LOCAL.match(alias):
                    continue
                if not isinstance(expr, str) or not _ALLOWED_AGG_RE_LOCAL.match(expr.strip()):
                    continue
                clean_aggs[alias] = expr.strip()
            refined.aggregates = clean_aggs

        if refined.aggregates and refined.select_vars:
            if not refined.group_by:
                refined.group_by = list(refined.select_vars)
            else:
                gb = set(refined.group_by)
                for v in refined.select_vars:
                    if v not in gb:
                        refined.group_by.append(v)
                refined.group_by = _dedupe(refined.group_by)

        validate_queryspec(refined, registry)

        return refined

    except Exception:
        return spec
