"""
MoSAIC v2 (I1): plans with step instances, references and composition operators.

A v2 plan is a list of steps. A step is either a TOOL step (one of the agent tools, i.e. one SPARQL block) or an
OPERATOR step whose children are steps again:

    {"id": "a", "tool": "find_semantic_relations", "params": {"seed_word": "mammifero", "relation_type": "hyponym"}}
    {"id": "b", "tool": "find_semantic_relations", "params": {"seed_word": {"ref": "a.wordRel"}, ...}}
    {"id": "t", "tool": "translate_to_sicilian", "input": "j"}
    {"id": "n", "op": "not_exists", "steps": [...]}
    {"id": "u", "op": "union", "branches": [[...], [...]]}
    {"id": "c", "op": "compare", "left": "a.polarityValue", "cmp": ">", "right": {"ref": "r.value"} | 0.5 | "word"}
    {"id": "r", "op": "reference", "steps": [...], "function": "MAX", "variable": "x.polarityValue"}
    {"id": "g", "op": "aggregate", "steps": [...], "group_by": ["a.wr"], "function": "COUNT",
     "variable": "e.emotion", "having": {"cmp": ">", "value": 6}}

Scoping: every variable of a block instance is renamed ?<id>_<name>, except the variables it REQUIRES, which are
bound to the providing step (explicit "input": "<id>", or by default the latest visible step providing that name).
A reference {"ref": "<id>.<var>"} in a word parameter (seed_word, pattern_text with pattern_type "exact", words)
becomes a join: inside CompL-it, onto the word IRI of the referenced step; elsewhere a FILTER on STR().

Consecutive CompL-it blocks of the same scope are merged into one SERVICE clause, so that chained relations are
evaluated remotely in a single request. NOT EXISTS over a SERVICE block is rendered as OPTIONAL + !BOUND
(Virtuoso refuses NOT EXISTS inside SERVICE).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .constants import PREFIXES

if TYPE_CHECKING:
    from .agent import QueryAgent

_VAR = re.compile(r"\?([A-Za-z_][A-Za-z0-9_]*)")
# conventional names of aggregate columns (v1 block aliases)
_AGG_NAMES = {"count", "avgValue", "sumValue", "minValue", "maxValue", "avgCount", "wordCount", "senseCount", "lemmaCount"}
REF_TOKEN = "__MOSAIC_REF_{}__"
# variables holding a CompL-it word IRI, per block, and the written-form variable compared with the word parameter
_COMPLIT_WORD = {
    "COMPLIT_SEMREL_OF_SEED_LEMMA": ("seedWord", "seedForm"),
    "COMPLIT_COUNT_SENSES_SINGLE": ("word", "writtenRep"),
}
# string variable of a step -> IRI variable of the same step (refs to a word's string become IRI joins in CompL-it)
_STRING_TO_IRI = {"itLemmaString": ("wordRel", "word"), "writtenRep": ("word",), "wr": ("lemma",),
                  "sicWR": ("sicLemma",), "parWR": ("parLemma",)}
# names accepted for a required variable when no step provides it under its own name
_ALIASES = {"liitaLemma": ("lemma",)}
NUMERIC_VARS = {"polarityValue", "value", "count"}


_LEMMA_TYPE_VALUES = re.compile(
    r"VALUES (\?\w+) \{ lila:Lemma lila:Hypolemma \}[ \t]*\n([ \t]*)(\?\w+) a \1 \.")


def _lemma_type_filters(sparql: str) -> str:
    """With a SERVICE in the query, render the lemma-type VALUES as FILTER IN (outside the SERVICE).

    Virtuoso can reject VALUES combined with a SERVICE invocation ("Unsupported combination of subqueries and service
    invocations", SP031; main test 2076), while the equivalent FILTER IN works. Queries without SERVICE are unchanged.
    """
    if "SERVICE <" not in sparql:
        return sparql
    return _LEMMA_TYPE_VALUES.sub(
        lambda m: f"{m.group(3)} a {m.group(1)} .\n{m.group(2)}FILTER({m.group(1)} IN (lila:Lemma, lila:Hypolemma)) .",
        sparql)


class TreePlanError(ValueError):
    """Invalid v2 plan (reported to the LLM in the repair round)."""


def is_tree_plan(raw_steps: List[Dict[str, Any]]) -> bool:
    """A plan uses the v2 language if any step has an id, an input, an operator, or a reference parameter."""
    def has_ref(v: Any) -> bool:
        if isinstance(v, dict):
            return "ref" in v or any(has_ref(x) for x in v.values())
        if isinstance(v, list):
            return any(has_ref(x) for x in v)
        return False
    return any(("id" in s or "op" in s or "input" in s or has_ref(s.get("params", {}))) for s in raw_steps)


@dataclass
class _Scope:
    """Visible variables: qualified name 'id.var' -> rendered var, plus latest provider per bare name."""
    qualified: Dict[str, str] = field(default_factory=dict)
    latest: Dict[str, str] = field(default_factory=dict)
    parent: Optional["_Scope"] = None

    def child(self) -> "_Scope":
        return _Scope(parent=self)

    def lookup(self, qname: str) -> Optional[str]:
        s: Optional[_Scope] = self
        while s:
            if qname in s.qualified:
                return s.qualified[qname]
            s = s.parent
        return None

    def latest_of(self, name: str) -> Optional[str]:
        s: Optional[_Scope] = self
        while s:
            if name in s.latest:
                return s.latest[name]
            s = s.parent
        return None

    def expose(self, sid: str, name: str, rendered: str) -> None:
        self.qualified[f"{sid}.{name}"] = rendered
        self.latest[name] = rendered


# A rendered element: ("plain", [lines]) or ("service", iri, [lines]); services are merged when adjacent
_Elem = Tuple


class TreeBuilder:
    """Render a v2 plan into SPARQL, reusing the agent's tool -> block/slot logic."""

    def __init__(self, agent: "QueryAgent") -> None:
        self.agent = agent
        self.prefixes: set = set()
        self._n = 0
        self.word_iri: Dict[str, str] = {}  # step id -> rendered CompL-it word IRI var
        self.str2iri: Dict[str, str] = {}   # rendered string var -> rendered IRI var of the same step
        self.word_sense: Dict[str, str] = {}  # rendered related-word var -> rendered related-sense var (semrel steps)
        self.agg_vals: Dict[str, str] = {}    # rendered value var of reference/aggregate operators -> function
        self.group_keys: set = set()          # rendered group_by vars of aggregate operators (projected by a subquery)
        self.joined: Dict[str, str] = {}      # group key -> group key it is joined with (compare "=" between groups)
        self.step_input: Dict[str, str] = {}  # tool step id -> its explicit input step id

    # ------------------------------------------------------------------ public
    def build(self, raw_steps: List[Dict[str, Any]], output_vars: List[str],
              aggregation: Optional[Dict[str, Any]] = None, question: Optional[str] = None) -> str:
        return _lemma_type_filters(self._build(raw_steps, output_vars, aggregation, question))

    def _build(self, raw_steps: List[Dict[str, Any]], output_vars: List[str],
               aggregation: Optional[Dict[str, Any]] = None, question: Optional[str] = None) -> str:
        scope = _Scope()
        steps, agg_steps, filters = self._split(raw_steps)
        elems = self._render_steps(steps, scope)
        for f in filters:  # filter_variable steps at top level
            elems.append(("plain", [self._filter_variable(f, scope)]))
        where = self._flatten(self._merge(elems))
        # v1 aggregation tools used as steps: their own outputs (e.g. "b.avgCount") are not step variables
        agg_ids = {str(s.get("id")) for s in agg_steps if s.get("id")}
        output_vars = [v for v in (output_vars or []) if str(v).lstrip("?").split(".", 1)[0] not in agg_ids]
        if agg_steps and agg_steps[-1].get("tool") == "average_count_per_item" and not aggregation:
            return self._average_per_item(where, agg_steps[-1], scope)
        aggregation = aggregation or (self._agg_from_step(agg_steps[-1]) if agg_steps else None)
        # As in v1: a definition column is kept only if the question asks for definitions/meanings (one definition
        # per sense multiplies the rows). The planner adds it often even when told not to.
        from .agent import _asks_for_definitions
        if question is not None and not _asks_for_definitions(question):
            kept = [v for v in output_vars if str(v).lstrip("?").rsplit(".", 1)[-1] != "definition"]
            if kept:
                output_vars = kept
        query = self._select(where, output_vars, aggregation, scope)
        for right, left in self.joined.items():  # group keys joined by compare "=" share one variable
            query = re.sub(re.escape(right) + r"\b", left, query)
        return query

    def _average_per_item(self, where: List[str], s: Dict[str, Any], scope: _Scope) -> str:
        """v1 tool average_count_per_item inside a v2 plan: AVG over a per-item COUNT."""
        p = s.get("params") or {}
        inp = s.get("input")

        def var(name: Any) -> str:
            n = str(name or "").lstrip("?")
            if not n:
                raise TreePlanError("average_count_per_item needs count_variable and item_variable")
            if "." not in n and inp:
                n = f"{inp}.{n}"
            return self._resolve(n, scope, str(s.get("id", "average")))

        cv, iv = var(p.get("count_variable")), var(p.get("item_variable"))
        inner = [f"{{ SELECT {iv} (COUNT(DISTINCT {cv}) AS ?itemCount) WHERE {{"] + ["    " + ln for ln in where] + \
                [f"  }} GROUP BY {iv}", "}"]
        return self._select(inner, [], {"function": "AVG", "variable": "?itemCount", "_alias": "?avgCount",
                                        "_raw": True}, scope)

    # ------------------------------------------------------------------ steps
    def _split(self, raw_steps):
        from .agent import AGENT_TOOLS
        steps, agg, flt = [], [], []
        for s in raw_steps:
            tool = s.get("tool")
            ttype = AGENT_TOOLS.get(tool, {}).get("type") if tool else None
            if ttype == "aggregation":
                agg.append(s)
            elif ttype == "filter":
                flt.append(s)
            else:
                steps.append(s)
        return steps, agg, flt

    def _new_id(self) -> str:
        self._n += 1
        return f"s{self._n}"

    def _render_steps(self, steps: List[Dict[str, Any]], scope: _Scope) -> List[_Elem]:
        from .agent import AGENT_TOOLS
        elems: List[_Elem] = []
        for s in steps:
            sid = str(s.get("id") or self._new_id())
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", sid):
                raise TreePlanError(f"Step id '{sid}' must be alphanumeric (e.g. 'a', 'b2')")
            op = s.get("op")
            if op:
                elems.extend(self._render_op(op, sid, s, scope))
                continue
            tool = s.get("tool")
            ttype = AGENT_TOOLS.get(tool, {}).get("type") if tool else None
            if ttype == "filter":
                elems.append(("plain", [self._filter_variable(s, scope)]))
            elif ttype == "aggregation":
                raise TreePlanError(f"Step '{sid}': aggregation tools are only allowed at top level; "
                                    f"use an 'aggregate' operator inside other operators")
            else:
                elems.extend(self._render_tool(sid, s, scope))
        return elems

    def _render_tool(self, sid: str, s: Dict[str, Any], scope: _Scope) -> List[_Elem]:
        from .agent import AGENT_TOOLS, ToolCall
        tool = s.get("tool", "")
        tool_def = AGENT_TOOLS.get(tool)
        if not tool_def:
            raise TreePlanError(f"Step '{sid}': unknown tool '{tool}'")
        if tool == "bind_lemma_to_liita":
            # ?lemma already is the LiITA lemma: expose it as <sid>.liitaLemma instead of emitting BIND, which
            # Virtuoso evaluates wrongly inside nested groups (FILTER NOT EXISTS { BIND(?outer AS ?x) ... }).
            inp = s.get("input")
            lemma = self._resolve(f"{inp}.lemma", scope, sid) if inp else scope.latest_of("lemma")
            if lemma is None:
                raise TreePlanError(f"Step '{sid}': bind_lemma_to_liita needs a step providing ?lemma")
            scope.expose(sid, "liitaLemma", lemma)
            return []
        params = dict(s.get("params") or {})
        refs: Dict[str, str] = {}  # token -> rendered var

        def deref(v: Any) -> Any:
            if isinstance(v, dict) and "ref" in v:
                tok = REF_TOKEN.format(len(refs))
                refs[tok] = self._resolve(v["ref"], scope, sid)
                return tok
            if isinstance(v, list):
                return [deref(x) for x in v]
            return v
        params = {k: deref(v) for k, v in params.items()}
        if tool == "find_liita_lemmas_by_pattern" and any(t in str(params.get("pattern_text", "")) for t in refs):
            params["pattern_type"] = "exact"

        step = ToolCall(tool=tool, params=params)
        if s.get("input"):
            pre = f"{s['input']}."
            known = {f"?{q[len(pre):]}" for q in self._visible_qnames(scope) if q.startswith(pre)}
        else:
            known = {f"?{n}" for n in self._visible_names(scope)}
        block_id = self.agent._resolve_block_id(step, tool_def, known)
        if not block_id:
            raise TreePlanError(f"Step '{sid}' ({tool}): cannot resolve the SPARQL block")
        block = self.agent.registry.get(block_id)
        slots = self.agent._build_slots(step, tool_def, block_id)
        lines = [ln.format(**slots) for ln in block.where]
        self.prefixes |= set(block.prefixes)
        if any("complit:" in ln for ln in lines):
            self.prefixes.add("complit")

        # bindings for required variables
        rename: Dict[str, str] = {}
        src = s.get("input")
        if src:
            self.step_input[str(sid)] = str(src)
        for req in sorted(block.requires):
            name = req[1:]
            bound = None
            cands = (name,) + _ALIASES.get(name, ())
            if src:
                # the input step may only filter its own input (e.g. get_emotions on a lemma step): follow the input
                # chain back to the step that provides the variable
                hop, seen = src, set()
                while hop and hop not in seen and not bound:
                    seen.add(hop)
                    for c in cands:
                        bound = scope.lookup(f"{hop}.{c}")
                        if bound:
                            break
                    hop = self.step_input.get(hop)
                if not bound:
                    raise TreePlanError(f"Step '{sid}' ({tool}): input step '{src}' does not provide {req}")
            else:
                for c in cands:
                    bound = scope.latest_of(c)
                    if bound:
                        break
                if not bound:
                    raise TreePlanError(f"Step '{sid}' ({tool}) requires {req}, but no previous step provides it")
            rename[name] = bound

        # references in word parameters
        post_filters: List[str] = []
        service = block.service_iri
        word_info = _COMPLIT_WORD.get(block_id)
        out_lines: List[str] = []
        for ln in lines:
            toks = [t for t in refs if t in ln]
            if not toks:
                out_lines.append(ln)
                continue
            tok = toks[0]
            target = refs[tok]
            m = re.search(r"(?i)str\((\?[A-Za-z_]\w*)\)\s*=\s*\"" + re.escape(tok) + r"\"", ln)
            if not m:
                raise TreePlanError(f"Step '{sid}' ({tool}): a reference cannot be used in this parameter")
            compared = m.group(1)[1:]
            if service and word_info and compared == word_info[1] and not self._is_complit_word(target):
                target = self.str2iri.get(target, target)
            if service and word_info and compared == word_info[1] and self._is_complit_word(target):
                rename[word_info[0]] = target  # join on the word IRI, drop the string filter
                if block_id == 'COMPLIT_SEMREL_OF_SEED_LEMMA' and target in self.word_sense:
                    rename['seedSense'] = self.word_sense[target]  # chain on the related SENSE, not on every sense
                continue
            new_var = rename.get(compared) or f"?{sid}_{compared}"
            cond = f"FILTER(STR({new_var}) = STR({target}))"
            if service:
                post_filters.append(cond)  # target is bound outside the remote service
            else:
                out_lines.append(re.sub(r"FILTER\s*\(.*\)\s*\.?\s*$", cond + " .", ln) if ln.strip().startswith("FILTER")
                                 else ln.replace(f'"{tok}"', f"STR({target})"))

        # block-local variables: those of the template AND those introduced by slot values (filters, clauses);
        # rendered variables of other steps never occur here (references are tokens until this point)
        local = {m.group(1) for ln in lines for m in _VAR.finditer(ln)}
        rendered = [self._rename(ln, sid, rename, local) for ln in out_lines]
        provided = {p[1:] for p in block.provides}
        for name in provided:
            scope.expose(sid, name, rename.get(name, f"?{sid}_{name}"))
        if word_info:
            iri = rename.get(word_info[0], f"?{sid}_{word_info[0]}")
            scope.expose(sid, word_info[0], iri)
        for name in ("wordRel", "word"):
            if name in provided:
                self.word_iri[sid] = rename.get(name, f"?{sid}_{name}")
        if tool == "find_semantic_relations":
            self.word_iri[sid] = rename.get("wordRel", f"?{sid}_wordRel")
            self.word_sense[self.word_iri[sid]] = rename.get("senseRel", f"?{sid}_senseRel")
        for sname, iris in _STRING_TO_IRI.items():
            if sname in provided or (word_info and sname == word_info[1]):
                for iname in iris:
                    iv = scope.lookup(f"{sid}.{iname}")
                    if iv:
                        self.str2iri[scope.lookup(f"{sid}.{sname}") or f"?{sid}_{sname}"] = iv
                        break
        if block_id in ("COMPLIT_SEMREL_OF_SEED_LEMMA",):
            scope.expose(sid, "seedWord", rename.get("seedWord", f"?{sid}_seedWord"))

        elems: List[_Elem] = [("service", service, rendered)] if service else [("plain", rendered)]
        if post_filters:
            elems.append(("plain", post_filters))
        return elems

    # ------------------------------------------------------------------ operators
    def _render_op(self, op: str, sid: str, s: Dict[str, Any], scope: _Scope) -> List[_Elem]:
        if op == "not_exists":
            inner = self._merge(self._render_steps(s.get("steps") or [], scope.child()))
            if not inner:
                raise TreePlanError(f"Operator '{sid}' (not_exists) has no steps")
            if any(e[0] == "service" for e in inner):
                probe = self._first_var(inner)
                return [("plain", ["OPTIONAL {"] + ["  " + ln for ln in self._flatten(inner)] + ["}",
                                   f"FILTER(!BOUND({probe}))"])]
            return [("plain", ["FILTER NOT EXISTS {"] + ["  " + ln for ln in self._flatten(inner)] + ["}"])]

        if op == "union":
            branches = s.get("branches") or []
            if len(branches) < 2:
                raise TreePlanError(f"Operator '{sid}' (union) needs at least two branches")
            rendered, exposed = [], []
            for br in branches:
                child = scope.child()
                lines = self._flatten(self._merge(self._render_steps(br, child)))
                rendered.append((lines, child))
                exposed.append(set(child.latest))
            common = set.intersection(*exposed)
            out = []
            for lines, child in rendered:
                for name in common:
                    lines = [re.sub(re.escape(child.latest[name]) + r"\b", f"?{sid}_{name}", ln) for ln in lines]
                out.append(lines)
            for name in common:
                scope.expose(sid, name, f"?{sid}_{name}")
            body = []
            for i, lines in enumerate(out):
                body += (["UNION"] if i else []) + ["{"] + ["  " + ln for ln in lines] + ["}"]
            return [("plain", body)]

        if op == "compare":
            left = self._operand(s.get("left"), scope, sid)
            right = self._operand(s.get("right"), scope, sid)
            cmp = s.get("cmp", "=")
            if cmp not in ("=", "!=", "<", ">", "<=", ">="):
                raise TreePlanError(f"Operator '{sid}' (compare): unknown comparison '{cmp}'")
            if cmp == "=" and left in self.group_keys and right in self.group_keys and left != right:
                # Two aggregate subqueries aligned on their group key: a real join (same variable) instead of a
                # FILTER over their cross product, which Virtuoso may refuse as too expensive.
                self.joined[right] = left
                return []
            if cmp in ("=", "!=") and not (self._is_numeric(left) or self._is_numeric(right)):
                return [("plain", [f"FILTER(STR({left}) {cmp} STR({right}))"])]
            # numbers (aggregate values, polarity values, numeric literals) are compared as numbers: STR() of an AVG
            # and of the MIN of the same values can differ in datatype/format (main test 2094: 0 rows)
            # aggregate values are already numbers and are compared uncast (casting both sides of AVG vs MIN(AVG)
            # matched nothing on Virtuoso, main test 2094)
            num = lambda x: x if x in self.agg_vals else self._num(x)  # noqa: E731
            return [("plain", [f"FILTER({num(left)} {cmp} {num(right)})"])]

        if op in ("reference", "aggregate"):
            child = scope.child()
            inner = self._flatten(self._merge(self._render_steps(s.get("steps") or [], child)))
            fn = str(s.get("function", "COUNT")).upper()
            if fn not in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
                raise TreePlanError(f"Operator '{sid}' ({op}): unknown function '{fn}'")
            var = self._resolve(s.get("variable", ""), child, sid) if s.get("variable") else None
            if fn == "COUNT":
                expr = f"COUNT(DISTINCT {var})" if var else "COUNT(*)"
            else:
                if not var:
                    raise TreePlanError(f"Operator '{sid}' ({op}): '{fn}' needs a 'variable'")
                # MIN/MAX of a value that is already an aggregate (e.g. per-group AVGs) is not cast: Virtuoso returns a
                # MIN(xsd:float(?avg)) that no later comparison matches (main test 2094: 0 rows)
                if fn in ("MIN", "MAX") and var in self.agg_vals:
                    expr = f"{fn}({var})"
                else:
                    expr = f"{fn}({self._num(var)})"
            group = [self._resolve(g, child, sid) for g in (s.get("group_by") or [])] if op == "aggregate" else []
            alias = s.get("as")
            if alias is not None and not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", str(alias)):
                raise TreePlanError(f"Operator '{sid}' ({op}): 'as' must be a plain name like 'positiveCount'")
            vname = f"?{alias}" if alias else f"?{sid}_value"
            sel = " ".join(group + [f"({expr} AS {vname})"])
            sub = [f"{{ SELECT {sel} WHERE {{"] + ["    " + ln for ln in inner] + ["  }"]
            if group:
                sub.append("  GROUP BY " + " ".join(group))
            hv = s.get("having")
            if hv:
                hc = hv.get("cmp", ">")
                hval = self._operand(hv.get("value"), scope, sid)
                sub.append(f"  HAVING ({expr} {hc} {hval})")
            sub.append("}")
            scope.expose(sid, "value", vname)
            self.agg_vals[vname] = fn
            if alias:
                scope.qualified[f"{sid}.{alias}"] = vname
            self.group_keys |= set(group)
            for g, q in zip(group, s.get("group_by") or []):
                scope.expose(sid, q.split(".", 1)[-1], g)  # grouped vars stay visible outside (projected)
                scope.qualified[q] = g
            return [("plain", sub)]

        raise TreePlanError(f"Step '{sid}': unknown operator '{op}' (use not_exists, union, compare, reference, aggregate)")

    # ------------------------------------------------------------------ helpers
    def _resolve(self, qname: Any, scope: _Scope, sid: str) -> str:
        if isinstance(qname, dict) and "ref" in qname:
            qname = qname["ref"]
        if not isinstance(qname, str) or not qname:
            raise TreePlanError(f"Step '{sid}': invalid reference {qname!r}")
        q = qname.lstrip("?")
        if "." in q:
            v = scope.lookup(q)
            if v is None:
                sid2, name = q.split(".", 1)
                for alt in _STRING_TO_IRI.get(name, ()):  # e.g. a.wr not exposed but a.lemma is
                    v = scope.lookup(f"{sid2}.{alt}")
                    if v:
                        break
            if v is None:
                raise TreePlanError(f"Step '{sid}': reference '{qname}' is not visible here "
                                    f"(visible: {sorted(self._visible_qnames(scope))[:30]})")
            return v
        v = scope.latest_of(q)
        if v is None:
            raise TreePlanError(f"Step '{sid}': no visible step provides ?{q}")
        return v

    def _operand(self, v: Any, scope: _Scope, sid: str) -> str:
        if isinstance(v, dict) and "ref" in v:
            return self._resolve(v, scope, sid)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(v)
        if isinstance(v, str):
            if re.fullmatch(r"\??[A-Za-z]\w*\.\w+", v):
                return self._resolve(v, scope, sid)
            if re.fullmatch(r"-?\d+(\.\d+)?", v):
                return v
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        raise TreePlanError(f"Step '{sid}': invalid operand {v!r}")

    def _is_numeric(self, x: str) -> bool:
        return (x in self.agg_vals or bool(re.fullmatch(r"-?\d+(\.\d+)?", x))
                or x.rsplit("_", 1)[-1].lstrip("?") in NUMERIC_VARS)

    def _num(self, x: str) -> str:
        if re.fullmatch(r"-?\d+(\.\d+)?", x):
            return x
        self.prefixes.add("xsd")
        return f"xsd:float({x})"

    def _is_complit_word(self, var: str) -> bool:
        return var in self.word_iri.values() or bool(re.search(r"_(wordRel|word|seedWord)$", var))

    def _visible_names(self, scope: _Scope) -> set:
        out, s = set(), scope
        while s:
            out |= set(s.latest)
            s = s.parent
        return out

    def _visible_qnames(self, scope: _Scope) -> set:
        out, s = set(), scope
        while s:
            out |= set(s.qualified)
            s = s.parent
        return out

    @staticmethod
    def _rename(line: str, sid: str, rename: Dict[str, str], local: set) -> str:
        """Rename the block's own variables; leave already-rendered variables of other steps untouched."""
        def sub(m):
            n = m.group(1)
            if n in rename:
                return rename[n]
            return f"?{sid}_{n}" if n in local else m.group(0)
        return _VAR.sub(sub, line)

    @staticmethod
    def _merge(elems: List[_Elem]) -> List[_Elem]:
        out: List[_Elem] = []
        for e in elems:
            if e[0] == "service" and out and out[-1][0] == "service" and out[-1][1] == e[1]:
                out[-1] = ("service", e[1], out[-1][2] + e[2])
            else:
                out.append(e)
        return out

    @staticmethod
    def _flatten(elems: List[_Elem]) -> List[str]:
        lines: List[str] = []
        for e in elems:
            if e[0] == "service":
                lines += [f"SERVICE <{e[1]}> {{"] + ["  " + ln for ln in e[2]] + ["}"]
            else:
                lines += list(e[1])
        return [ln for ln in lines if ln.strip()]

    @staticmethod
    def _first_var(elems: List[_Elem]) -> str:
        for e in elems:
            for ln in (e[2] if e[0] == "service" else e[1]):
                m = _VAR.search(ln)
                if m:
                    return m.group(0)
        raise TreePlanError("not_exists: no variable to test")

    def _filter_variable(self, s: Dict[str, Any], scope: _Scope) -> str:
        from .utils import build_filter_for_var
        p = s.get("params") or {}
        var = self._resolve(p.get("variable", ""), scope, str(s.get("id", "filter")))
        f = build_filter_for_var(var, p.get("pattern_type", "contains"), str(p.get("pattern_text", "")))
        if not f:
            raise TreePlanError(f"filter_variable: unsupported pattern_type {p.get('pattern_type')!r}")
        return f

    @staticmethod
    def _agg_from_step(s: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(s.get("params") or {})
        if s.get("tool") == "count_results":
            return {"function": "COUNT", "variable": p.get("count_variable"), "group_by": p.get("group_by") or [],
                    "order": p.get("order")}
        return {"function": p.get("agg_function", "COUNT"), "variable": p.get("agg_variable"),
                "group_by": p.get("group_by") or [], "having": ({"cmp": p["having_op"], "value": p["having_value"]}
                                                               if p.get("having_op") else None),
                "order": p.get("order")}

    def _select(self, where: List[str], output_vars: List[str], agg: Optional[Dict[str, Any]], scope: _Scope) -> str:
        outs = [self._resolve(v, scope, "output") for v in (output_vars or [])]
        if not outs and not agg:
            raise TreePlanError("output_vars is empty")
        # rename output columns back to their bare names when unambiguous (the evaluator maps by name)
        bare = [o[1:].split("_", 1)[-1] if "_" in o else o[1:] for o in outs]
        # A single aggregate value in the answer gets the conventional name (count, avgValue...), as in v1:
        # free names like ?total or ?numDefinitions are not recognised as aggregates by the evaluator.
        # Several values (one per statistic) keep the names chosen in the plan.
        agg_outs = [i for i, o in enumerate(outs) if o in self.agg_vals]
        if len(agg_outs) == 1:
            i = agg_outs[0]
            fn = self.agg_vals[outs[i]]
            if bare[i] not in _AGG_NAMES:
                bare[i] = "count" if fn == "COUNT" else f"{fn.lower()}Value"
        cols = []
        for o, b in zip(outs, bare):
            cols.append(o if bare.count(b) > 1 or o == f"?{b}" else f"({o} AS ?{b})")
        head, tail = [], []
        if agg and agg.get("function"):
            fn = str(agg["function"]).upper()
            if agg.get("_raw"):
                var = agg["variable"]
            else:
                var = self._resolve(agg["variable"], scope, "aggregation") if agg.get("variable") else None
            if fn == "COUNT":
                expr = f"COUNT(DISTINCT {var})" if var else "COUNT(*)"
                alias = "?count"
            else:
                expr = f"{fn}({self._num(var)})"
                alias = f"?{fn.lower()}Value"
            alias = agg.get("_alias", alias)
            group = [self._resolve(g, scope, "aggregation") for g in (agg.get("group_by") or [])]
            # Outputs join the GROUP BY only when the plan groups (e.g. a label next to its group key), and never the
            # aggregated variable itself: without group_by the answer is one row ("how many senses" = 1 count,
            # not one count per sense).
            if group:
                for o in outs:
                    if o not in group and o != var:
                        group.append(o)
            gcols = []
            for g in group:
                b = g[1:].split("_", 1)[-1] if "_" in g else g[1:]
                gcols.append(g if g == f"?{b}" or sum(1 for x in group if x.endswith('_' + b)) > 1 else f"({g} AS ?{b})")
            head = gcols + [f"({expr} AS {alias})"]
            if group:
                tail.append("GROUP BY " + " ".join(group))
            hv = agg.get("having")
            if hv:
                tail.append(f"HAVING ({expr} {hv.get('cmp', '>')} {self._operand(hv.get('value'), scope, 'having')})")
            if agg.get("order") in ("asc", "desc"):
                tail.append(f"ORDER BY {'ASC' if agg['order'] == 'asc' else 'DESC'}({alias})")
            select = "SELECT " + " ".join(head)
        else:
            select = "SELECT DISTINCT " + " ".join(cols)
        body = "\n".join(where)
        for p in re.findall(r"\b([a-z]+):", body + " " + select + " " + " ".join(tail)):
            if p in PREFIXES:
                self.prefixes.add(p)
        prefix_lines = [f"PREFIX {p}: <{PREFIXES[p]}>" for p in sorted(self.prefixes) if p in PREFIXES]
        q = "\n".join(prefix_lines) + "\n\n" + select + "\nWHERE {\n" + "\n".join("  " + ln for ln in where) + "\n}\n"
        return q + ("\n".join(tail) + "\n" if tail else "")
