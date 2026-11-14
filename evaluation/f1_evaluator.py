"""Unified F1-on-answers evaluator shared by PRISMA, MoSAIC and NL2SPARQL.

Executes both gold and predicted SPARQL queries against the LiITA endpoint
and compares their result sets using F1 score (precision/recall on answer tuples).

One identical copy of this file is used by all three systems (built 2026-11-03
from mosaic-liita/evaluation/f1_evaluator.py), so that their scores are comparable:
  - ANSWER_VARIABLE_CATEGORIES is the union of the three former tables
    (on conflicts the MoSAIC/PRISMA category is kept: polarity -> uris,
    emotionLabel / lemma / lexicalEntry -> secondary)
  - variable mapping: name / category / similarity phases as in PRISMA and MoSAIC,
    plus a final value-based phase: still-unmatched gold primary variables are
    mapped, in SELECT order, to remaining predicted variables whose values are
    literals (not IRIs), whatever their name
  - queries go through GET, falling back to POST when GET fails or the query is
    too long for a URL (the LiITA proxy rejects some POSTed GROUP_CONCAT queries)
  - variable names missing from the table that end in Wr / Word / WrittenRep are
    classified as primary (written forms are user-facing answers)
  - averages (aggregates named avg* / average* / mean*) match within 1% relative tolerance
  - column mapping by value (2026-11-13, default column_mapping="value"): each gold answer
    column is matched to the predicted column holding the same values (Dice overlap of the
    two value sets, at least 0.1; one-to-one assignment maximising the total overlap); aggregate-only
    answers are matched on the values of the first row. The choice looks only at
    column-wise overlap, never at the resulting F1. Gold columns with no overlapping
    predicted column keep the name-based mapping below; column_mapping="name" reproduces
    the scores computed before this change. Reason: names are arbitrary (?lemma holds a
    written form in one query and a lemma IRI in another), and the name rules mapped
    identical answers to the wrong column (main 2072; held-out 3016, 3055, 3069)
  - echo columns (2026-11-14, default ignore_echo_columns=True): a gold primary column that holds
    the same value on every row, written as a literal in the gold query (the seed word of the
    question, e.g. ?italianWord = "donna"), is not required; answers that list the translations
    without repeating the seed word were scored 0 (held-out 3030, main 2063, 2065). Nothing is
    dropped when all primary columns are echo columns. ignore_echo_columns=False reproduces the
    previous scores
  - F1Evaluator accepts both strip_limit (PRISMA/MoSAIC) and strip_predicted_limit
    (NL2SPARQL) for the same option
  - IRI/literal equivalents: when a gold primary answer is an IRI, the gold may name
    a literal column with the same answer in answer_variables["equivalents"]
    ({iri_var: literal_var}, e.g. a CompL-it Word IRI and its written form). If the
    predicted column mapped to iri_var holds literals, it is compared with
    literal_var instead (gold rows without a literal are then left out), so
    systems are not penalised for returning words instead of IRIs or vice versa

Usage:
    evaluator = F1Evaluator()
    result = evaluator.evaluate_single(gold_sparql, predicted_sparql, answer_vars)
    print(f"F1: {result.f1:.3f}")
"""

import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants (defined locally — no dependency on external packages)
# ---------------------------------------------------------------------------

LIITA_ENDPOINT = "https://liita.it/sparql"

# Relative tolerance for averages (gold aggregates named avg*): a slightly
# different lemma set should not turn a near-identical mean into a miss.
# Counts and other aggregates still require an exact match.
AVG_REL_TOLERANCE = 0.01

ANSWER_VARIABLE_CATEGORIES = {
    "primary": [
        # --- Gold query variables (user-facing strings) ---
        "word", "relatedWord", "italianWR", "wr",
        "parmigianoWR", "sicilianWR",
        "definition", "polarityLabel",
        "relationType", "example", "wordA", "wordB",
        "italianWord", "sicilianWord", "parmigianoWord", "dialectWord",
        "hypernymWord", "hyponymWord", "meronymWord",
        "synonymWord", "antonymWord", "translationWord",
        # --- MoSAIC-specific (after de-aggregation) ---
        "itLemmaString",      # MoSAIC Italian written rep (CompL-IT path)
        "sicWR",              # MoSAIC Sicilian written rep (de-aggregated)
        "parWR",              # MoSAIC Parmigiano written rep (de-aggregated)
        "definitionSample",   # MoSAIC SAMPLE(definition) — if not de-aggregated
        "wordRel",            # MoSAIC related word (CompL-IT lexical entry IRI)
        "writtenRep",         # MoSAIC CompL-IT written rep (sense-count blocks)
    ],
    "secondary": [
        # --- Gold variables ---
        "lemma", "sense", "liitaLemma", "sentixLemma", "elitaLemma",
        "complitWord", "parmigianoLemma", "sicilianLemma",
        "pos", "gender", "senseA", "senseB", "posA", "posB",
        "leParmigianoIta", "leParmigianoSic",
        "leSicilianIta", "leSicilianSic",
        "lexEntryLabel", "lexicalEntry", "italianLemma",
        # emotionLabel is classified as secondary so it doesn't displace dialect
        # variables in position-based matching.
        "emotionLabel",
        # --- MoSAIC-specific ---
        "sicLemma",           # MoSAIC Sicilian lemma IRI
        "parLemma",           # MoSAIC Parmigiano lemma IRI
        "senseRel",           # MoSAIC related sense
    ],
    "aggregates": [
        "count", "aggregated", "wordCount",
        "sumValue", "avgValue", "minValue", "maxValue",
        "avgPolarityValue", "avgForms",
        # --- MoSAIC GROUP_CONCAT aggregates (when not de-aggregated) ---
        "sicilianoWRs",       # GROUP_CONCAT of Sicilian written reps
        "parmigianoWRs",      # GROUP_CONCAT of Parmigiano written reps
        "polarityLabels",     # GROUP_CONCAT of polarity labels
        "polarityValues",     # GROUP_CONCAT of polarity values
        "emotions",           # GROUP_CONCAT of emotion labels
        "variantCount",       # COUNT(wr) > 1 queries
    ],
    "numeric": [
        "polarityValue", "value", "length",
    ],
    "uris": [
        "emotion", "polarity", "graph", "form",
        "emotionLexEntry",
    ],
}

# Variables added from the PRISMA and NL2SPARQL tables (unified evaluator)
_UNION_ADDITIONS = {
    'primary': ['word', 'relatedWord', 'italianWR', 'wr', 'parmigianoWR', 'sicilianWR', 'definition', 'polarityLabel', 'relationType', 'example', 'wordA', 'wordB', 'italianWord', 'sicilianWord', 'parmigianoWord', 'dialectWord', 'hypernymWord', 'hyponymWord', 'meronymWord', 'synonymWord', 'antonymWord', 'translationWord', 'itLemmaString', 'sicWR', 'parWR', 'definitionSample', 'wordRel', 'writtenRep', 'inflectedForm', 'partWord', 'senseDefinition', 'senseLabel', 'wholeWord'],
    'secondary': ['lemma', 'sense', 'liitaLemma', 'sentixLemma', 'elitaLemma', 'complitWord', 'parmigianoLemma', 'sicilianLemma', 'pos', 'gender', 'senseA', 'senseB', 'posA', 'posB', 'leParmigianoIta', 'leParmigianoSic', 'leSicilianIta', 'leSicilianSic', 'lexEntryLabel', 'lexicalEntry', 'italianLemma', 'emotionLabel', 'sicLemma', 'parLemma', 'senseRel', 'dialectLemma', 'dialectName', 'expressionLabel', 'formType', 'grammaticalInfo', 'hypSense', 'italianGender', 'label', 'morphPattern', 'relatedSense', 'senseLexicon', 'sensePOS', 'sicilianGender'],
    'aggregates': ['count', 'aggregated', 'wordCount', 'sumValue', 'avgValue', 'minValue', 'maxValue', 'avgPolarityValue', 'avgForms', 'sicilianoWRs', 'parmigianoWRs', 'polarityLabels', 'polarityValues', 'emotions', 'variantCount', 'entries', 'genders', 'lemmaCount', 'linkCount', 'score', 'senseCount'],
    'numeric': ['polarityValue', 'value', 'length', 'len'],
    'uris': ['emotion', 'polarity', 'graph', 'form', 'emotionLexEntry', 'cf', 'dialectForm', 'dialectLemmaBank', 'dialectLexEntry', 'emotionResource', 'g', 'italianLexEntry', 'italianLexicalEntry', 'lemmaResource', 'o', 'p', 'parmigianoForm', 'parmigianoLexEntry', 'polarityLexEntry', 's', 'senseMeronym', 'sicilianForm', 'sicilianLexEntry', 'translationEntry', 'wType', 'wordMeronym'],
}
for _cat, _names in _UNION_ADDITIONS.items():
    _known = ANSWER_VARIABLE_CATEGORIES.setdefault(_cat, [])
    _known.extend(n for n in _names if n not in _known)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryExecutionResult:
    """Result of executing a SPARQL query against an endpoint."""
    success: bool
    results: list[dict[str, str]]  # full result set
    result_count: int
    variables: list[str]           # variable names from results header
    error: Optional[str] = None


@dataclass
class F1Result:
    """F1 score result for a single test case."""
    test_id: int
    precision: float
    recall: float
    f1: float
    gold_count: int
    predicted_count: int
    true_positives: int
    aggregate_score: Optional[float] = None
    aggregate_details: dict = field(default_factory=dict)
    variable_mapping: dict = field(default_factory=dict)
    gold_error: Optional[str] = None
    predicted_error: Optional[str] = None
    predicted_sparql: Optional[str] = None   # generated query for inspection
    ignored_columns: list = field(default_factory=list)  # gold echo columns left out (see echo_columns)


@dataclass
class F1Report:
    """Aggregate F1 evaluation report."""
    total_evaluated: int
    total_skipped: int
    avg_precision: float
    avg_recall: float
    avg_f1: float
    macro_f1: float
    f1_by_category: dict = field(default_factory=dict)
    f1_by_pattern: dict = field(default_factory=dict)
    results: list[F1Result] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def strip_limit_offset(sparql: str) -> str:
    """Remove LIMIT and OFFSET clauses from a SPARQL query string."""
    sparql = re.sub(r'\bLIMIT\s+\d+\b', '', sparql, flags=re.IGNORECASE)
    sparql = re.sub(r'\bOFFSET\s+\d+\b', '', sparql, flags=re.IGNORECASE)
    return sparql.strip()


WRITTEN_FORM_SUFFIXES = ("wr", "word", "writtenrep")

MAX_GET_QUERY_CHARS = 6000  # longer queries go straight to POST (URL length limits)


def execute_query_full(
    sparql: str,
    endpoint: str = LIITA_ENDPOINT,
    timeout: int = 60,
    max_results: int = 10000,
) -> QueryExecutionResult:
    """Execute a SPARQL query: GET first, POST as fallback (or POST only for long queries)."""
    methods = ["POST"] if len(sparql) > MAX_GET_QUERY_CHARS else ["GET", "POST"]
    result = None
    for method in methods:
        result = _execute_query_once(sparql, endpoint, timeout, max_results, method)
        if result.success:
            return result
    return result


def _execute_query_once(
    sparql: str,
    endpoint: str = LIITA_ENDPOINT,
    timeout: int = 60,
    max_results: int = 10000,
    method: str = "GET",
) -> QueryExecutionResult:
    """Execute a SPARQL query once with the given HTTP method and return the full result set.

    Args:
        sparql: SPARQL query string
        endpoint: SPARQL endpoint URL
        timeout: Query timeout in seconds
        max_results: Maximum results to return (safety limit)

    Returns:
        QueryExecutionResult with full result set
    """
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON, GET, POST, URLENCODED
    except ImportError:
        return QueryExecutionResult(
            success=False, results=[], result_count=0, variables=[],
            error="SPARQLWrapper not installed",
        )

    try:
        client = SPARQLWrapper(endpoint)
        client.setQuery(sparql)
        client.setReturnFormat(JSON)
        client.setTimeout(timeout)
        if method == "POST":
            client.setMethod(POST)
            client.setRequestMethod(URLENCODED)
        else:
            client.setMethod(GET)
        client.addCustomHttpHeader("Accept", "application/sparql-results+json")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="unknown response content type",
                category=RuntimeWarning,
                module="SPARQLWrapper",
            )
            query_result = client.query()
            results = query_result.convert()

        # Handle bytes response
        if isinstance(results, bytes):
            import json
            decoded = results.decode("utf-8", errors="replace").strip()
            if not decoded:
                return QueryExecutionResult(
                    success=True, results=[], result_count=0, variables=[],
                )
            try:
                results = json.loads(decoded)
            except Exception:
                return QueryExecutionResult(
                    success=False, results=[], result_count=0, variables=[],
                    error=f"Invalid JSON response: {decoded[:200]}",
                )

        if not isinstance(results, dict):
            return QueryExecutionResult(
                success=False, results=[], result_count=0, variables=[],
                error=f"Unexpected response type: {type(results).__name__}",
            )

        # Extract variable names from header
        variables = results.get("head", {}).get("vars", [])

        # Extract all bindings
        bindings = results.get("results", {}).get("bindings", [])

        # Apply safety limit
        bindings = bindings[:max_results]

        # Convert to simple dict format
        rows = []
        for binding in bindings:
            row = {}
            for var, value_obj in binding.items():
                row[var] = value_obj.get("value", "")
            rows.append(row)

        return QueryExecutionResult(
            success=True,
            results=rows,
            result_count=len(rows),
            variables=variables,
        )

    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            error_msg = f"Query timeout after {timeout}s"
        return QueryExecutionResult(
            success=False, results=[], result_count=0, variables=[],
            error=error_msg,
        )


def normalize_value(value: str, is_numeric: bool = False) -> str:
    """Normalize a result value for comparison.

    Args:
        value: Raw value string
        is_numeric: Whether to apply numeric normalization

    Returns:
        Normalized value string
    """
    value = value.strip()

    if is_numeric and value:
        try:
            num = float(value)
            return str(round(num, 4))
        except ValueError:
            pass

    return value


def _parse_select_variables(sparql: str) -> list[str]:
    """Extract variable names from a SPARQL SELECT clause.

    Returns variable names without the ? prefix.
    """
    sparql_no_comments = re.sub(r'#[^\n]*', '', sparql)

    select_match = re.search(
        r'SELECT\s+(DISTINCT\s+)?(.+?)\s*(?:WHERE|FROM|\{)',
        sparql_no_comments,
        re.IGNORECASE | re.DOTALL,
    )
    if not select_match:
        return []

    select_clause = select_match.group(2).strip()

    variables = []

    # Find aggregate aliases: (... AS ?var)
    alias_pattern = r'AS\s+\?(\w+)'
    aliases = re.findall(alias_pattern, select_clause, re.IGNORECASE)
    variables.extend(aliases)

    # Track inner aggregate variables to exclude
    aggregate_inner = set()
    for pattern in [
        r'(?:COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?\?(\w+)',
        r'\(\s*[^)]*\?(\w+)[^)]*\s+AS\s+',
    ]:
        aggregate_inner.update(re.findall(pattern, select_clause, re.IGNORECASE))

    # Find standalone ?variables
    alias_set = set(aliases)
    tokens = re.split(r'[\s,()]+', select_clause)
    for token in tokens:
        if token.startswith('?'):
            var_name = token.lstrip('?')
            if var_name not in aggregate_inner and var_name not in alias_set:
                variables.append(var_name)

    # Deduplicate preserving order
    seen = set()
    result = []
    for v in variables:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def _classify_variable(var_name: str) -> str:
    """Classify a variable into a category using ANSWER_VARIABLE_CATEGORIES.

    Returns one of: "primary", "secondary", "aggregates", "numeric", "uris", "unknown".
    """
    for category, names in ANSWER_VARIABLE_CATEGORIES.items():
        if var_name in names:
            return category
    # Unified evaluator: names not in the table that denote a written form
    # (e.g. ?partWr, ?floreWord, ?lemmaWrittenRep) are user-facing answers
    if var_name.lower().endswith(WRITTEN_FORM_SUFFIXES):
        return "primary"
    return "unknown"


def build_variable_mapping(
    gold_answer_vars: dict,
    predicted_sparql: str,
    predicted_results: Optional[list[dict]] = None,
) -> dict[str, str]:
    """Map gold answer variable names to predicted variable names.

    Strategy:
    0. Apply known semantic equivalences (e.g. hyponymWord -> wordRel)
    1. Classify predicted SELECT variables using ANSWER_VARIABLE_CATEGORIES
    2. Match by category: gold primary[i] -> predicted primary[i]
    3. Fallback: direct name match
    4. Fallback: substring similarity

    Args:
        gold_answer_vars: Dict with keys primary, secondary, aggregates, numeric
        predicted_sparql: The predicted SPARQL query string

    Returns:
        Mapping from gold variable name to predicted variable name
    """
    predicted_vars = _parse_select_variables(predicted_sparql)

    # Classify predicted variables by category
    pred_by_category: dict[str, list[str]] = {
        "primary": [], "secondary": [], "aggregates": [], "numeric": [],
    }
    for var in predicted_vars:
        cat = _classify_variable(var)
        if cat in pred_by_category:
            pred_by_category[cat].append(var)

    mapping: dict[str, str] = {}
    used_predicted: set[str] = set()

    # Phase 0: Apply known semantic equivalences.
    # Gold queries use descriptive names (hyponymWord, italianWord, etc.) while
    # MoSAIC uses structural names (wordRel, itLemmaString, etc.).  When both
    # sides refer to the same SPARQL value type (e.g. CompL-IT Word IRI) we
    # must map them explicitly to avoid positional mis-matches.
    _SEMANTIC_EQUIV: dict[str, list[str]] = {
        # CompL-IT Word IRIs — gold names → preferred MoSAIC name(s)
        "hyponymWord":  ["wordRel"],
        "hypernymWord": ["wordRel"],
        "meronymWord":  ["wordRel"],
        "synonymWord":  ["wordRel"],
        "antonymWord":  ["wordRel"],
        "partWord":     ["wordRel"],
        # Written representation strings
        "italianWord":  ["itLemmaString", "wr"],
        "italianWR":    ["itLemmaString", "wr"],
        "sicilianWord": ["sicWR"],
        "parmigianoWord": ["parWR"],
        "dialectWord":  ["sicWR", "parWR"],
    }

    pred_set_all = set(predicted_vars)
    all_gold_flat = []
    for cat in ["primary", "secondary", "aggregates", "numeric"]:
        all_gold_flat.extend(gold_answer_vars.get(cat, []))

    for gold_var in all_gold_flat:
        if gold_var in mapping:
            continue
        candidates = _SEMANTIC_EQUIV.get(gold_var, [])
        for cand in candidates:
            if cand in pred_set_all and cand not in used_predicted:
                mapping[gold_var] = cand
                used_predicted.add(cand)
                break

    # Phase 1: Match by category — same name first, then position
    all_pred_set = set(predicted_vars)
    for category in ["primary", "secondary", "aggregates", "numeric"]:
        gold_vars = gold_answer_vars.get(category, [])
        pred_vars = pred_by_category.get(category, [])
        pred_set = set(pred_vars)

        # First pass: assign same-name matches within this category
        positional_candidates = []
        for gold_var in gold_vars:
            if gold_var in mapping:
                continue  # already resolved in Phase 0
            if gold_var in pred_set and gold_var not in used_predicted:
                mapping[gold_var] = gold_var
                used_predicted.add(gold_var)
            else:
                positional_candidates.append(gold_var)

        # Phase 1.5: cross-category same-name match for remaining candidates
        remaining_after_cross = []
        for gold_var in positional_candidates:
            if gold_var in all_pred_set and gold_var not in used_predicted:
                mapping[gold_var] = gold_var
                used_predicted.add(gold_var)
            else:
                remaining_after_cross.append(gold_var)

        # Phase 1.7: within-category substring similarity before positional fallback
        remaining_pred = [v for v in pred_vars if v not in used_predicted]
        remaining_after_sim: list[str] = []
        used_in_sim: set[str] = set()
        for gold_var in remaining_after_cross:
            gold_lower = gold_var.lower()
            best_match: str | None = None
            best_score = 0
            for pred_var in remaining_pred:
                if pred_var in used_in_sim:
                    continue
                pred_lower = pred_var.lower()
                score = 0
                for length in range(6, min(len(gold_lower), len(pred_lower)) + 1):
                    for start in range(len(gold_lower) - length + 1):
                        sub = gold_lower[start:start + length]
                        if sub in pred_lower:
                            score = max(score, length)
                if score > best_score:
                    best_score = score
                    best_match = pred_var
            if best_match and best_score >= 6:
                mapping[gold_var] = best_match
                used_predicted.add(best_match)
                used_in_sim.add(best_match)
            else:
                remaining_after_sim.append(gold_var)

        # Second pass: positional fallback for truly unmatched vars in this category
        remaining_pred = [v for v in pred_vars if v not in used_predicted]
        for gold_var, pred_var in zip(remaining_after_sim, remaining_pred):
            mapping[gold_var] = pred_var
            used_predicted.add(pred_var)

    # Phase 2: Direct name match for unmatched gold variables
    all_gold_vars = []
    for category in ["primary", "secondary", "aggregates", "numeric"]:
        all_gold_vars.extend(gold_answer_vars.get(category, []))

    unmatched_gold = [v for v in all_gold_vars if v not in mapping]
    available_pred = [v for v in predicted_vars if v not in used_predicted]

    for gold_var in unmatched_gold:
        if gold_var in available_pred:
            mapping[gold_var] = gold_var
            used_predicted.add(gold_var)
            available_pred.remove(gold_var)

    # Phase 3: Substring similarity for remaining unmatched
    still_unmatched = [v for v in all_gold_vars if v not in mapping]
    available_pred = [v for v in predicted_vars if v not in used_predicted]

    for gold_var in still_unmatched:
        best_match = None
        best_score = 0
        gold_lower = gold_var.lower()

        for pred_var in available_pred:
            pred_lower = pred_var.lower()
            score = 0
            for length in range(3, min(len(gold_lower), len(pred_lower)) + 1):
                for start in range(len(gold_lower) - length + 1):
                    sub = gold_lower[start:start + length]
                    if sub in pred_lower:
                        score = max(score, length)

            if score > best_score:
                best_score = score
                best_match = pred_var

        if best_match and best_score >= 3:
            mapping[gold_var] = best_match
            used_predicted.add(best_match)
            available_pred.remove(best_match)

    # Phase 4 (unified evaluator): value-based fallback. Names are not reliable
    # across systems (e.g. ?lemma holds a written form in some queries and a
    # lemma IRI in others), so still-unmatched gold primary variables are mapped,
    # in SELECT order, to remaining predicted variables whose values are literals.
    if predicted_results:
        unmatched_primary = [v for v in gold_answer_vars.get("primary", []) if v not in mapping]
        literal_pred = [
            v for v in predicted_vars
            if v not in used_predicted and _is_literal_column(predicted_results, v)
        ]
        for gold_var, pred_var in zip(unmatched_primary, literal_pred):
            mapping[gold_var] = pred_var
            used_predicted.add(pred_var)

    return mapping


VALUE_MAPPING_MAX_ASSIGNMENTS = 50000
# Below this Dice overlap a column match is taken as accidental (e.g. one Sicilian word spelled like an
# Italian gold word) and the name-based mapping is kept; such a match would give F1 near 0 anyway.
VALUE_MAPPING_MIN_OVERLAP = 0.1


def _value_set(rows: list[dict], var: str, is_numeric: bool) -> set:
    return {normalize_value(r[var], is_numeric=is_numeric) for r in rows if r.get(var)}


def _dice(a: set, b: set) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def _best_assignment(gold_vars: list[str], pred_vars: list[str], score) -> dict[str, str]:
    """One-to-one gold -> predicted assignment maximising the total score (pairs scoring 0 are dropped).

    Exhaustive when small, greedy (best pair first) otherwise. Ties keep the earlier predicted column.
    """
    import itertools

    if not gold_vars or not pred_vars:
        return {}
    k = min(len(gold_vars), len(pred_vars))
    n_assign = math.perm(len(pred_vars), k) * (math.comb(len(gold_vars), k) if len(gold_vars) > k else 1)
    best: dict[str, str] = {}
    if n_assign <= VALUE_MAPPING_MAX_ASSIGNMENTS:
        best_total = 0.0
        for golds in itertools.combinations(gold_vars, k):
            for preds in itertools.permutations(pred_vars, k):
                total = sum(score(g, p) for g, p in zip(golds, preds))
                if total > best_total + 1e-12:
                    best_total = total
                    best = {g: p for g, p in zip(golds, preds) if score(g, p) > 0}
        return best
    pairs = sorted(((score(g, p), -i, g, p) for g in gold_vars for i, p in enumerate(pred_vars)), reverse=True)
    used: set[str] = set()
    for sc, _, g, p in pairs:
        if sc <= 0:
            break
        if g not in best and p not in used:
            best[g] = p
            used.add(p)
    return best


def build_value_mapping(
    gold_answer_vars: dict,
    gold_results: list[dict],
    predicted_results: list[dict],
    predicted_sparql: str,
    name_mapping: dict[str, str],
) -> dict[str, str]:
    """Map gold answer columns to predicted columns by their values; fall back to name_mapping.

    Primary (tuple) columns: score = Dice overlap of the normalised value sets. For a gold IRI column
    with a literal equivalent, the literal column is used when the predicted column holds literals,
    as in compute_f1. Aggregate columns (compared on the first row by compute_f1): score 1 when the
    first-row values match (averages within AVG_REL_TOLERANCE), else 0.
    """
    pred_vars = [v for v in _parse_select_variables(predicted_sparql) if any(r.get(v) for r in predicted_results)]
    for r in predicted_results[:1]:  # variables bound in the results but not parsed from SELECT (e.g. SELECT *)
        pred_vars += [v for v in r if v not in pred_vars]
    numeric = set(gold_answer_vars.get("numeric", [])) | set(gold_answer_vars.get("aggregates", []))
    equivalents = gold_answer_vars.get("equivalents", {}) or {}
    primary = list(gold_answer_vars.get("primary", []))
    aggregates = [v for v in gold_answer_vars.get("aggregates", []) if v not in primary]

    gold_sets: dict[str, set] = {}
    pred_sets: dict[str, set] = {}

    def gset(var: str, num: bool) -> set:
        key = f"{var}|{num}"
        if key not in gold_sets:
            gold_sets[key] = _value_set(gold_results, var, num)
        return gold_sets[key]

    def pset(var: str, num: bool) -> set:
        key = f"{var}|{num}"
        if key not in pred_sets:
            pred_sets[key] = _value_set(predicted_results, var, num)
        return pred_sets[key]

    def primary_score(g: str, p: str) -> float:
        num = g in numeric
        gcol = g
        if equivalents.get(g) and _column_kind(predicted_results, p) == "literal":
            gcol = equivalents[g]
        sc = _dice(gset(gcol, num), pset(p, num))
        if sc < VALUE_MAPPING_MIN_OVERLAP:
            return 0.0
        return sc + (1e-6 if sc > 0 and name_mapping.get(g) == p else 0.0)  # tie-break: the name-based choice

    mapping = _best_assignment(primary, pred_vars, primary_score)

    if aggregates and gold_results and predicted_results:
        g0, p0 = gold_results[0], predicted_results[0]
        free = [p for p in pred_vars if p not in mapping.values()]

        def agg_score(g: str, p: str) -> float:
            gv = normalize_value(g0.get(g, ""), is_numeric=True)
            pv = normalize_value(p0.get(p, ""), is_numeric=True)
            if not gv or not pv:
                return 0.0
            ok = gv == pv
            if not ok and g.lower().startswith(("avg", "average", "mean")):
                try:
                    ok = math.isclose(float(gv), float(pv), rel_tol=AVG_REL_TOLERANCE)
                except ValueError:
                    pass
            return (1.0 + (1e-6 if name_mapping.get(g) == p else 0.0)) if ok else 0.0

        mapping.update(_best_assignment(aggregates, free, agg_score))

    used = set(mapping.values())
    for g, p in name_mapping.items():  # unmatched by value: keep the name-based choice if still free
        if g not in mapping and p not in used:
            mapping[g] = p
            used.add(p)
    return mapping


def echo_columns(gold_answer_vars: dict, gold_sparql: str, gold_results: list[dict]) -> list[str]:
    """Gold primary columns that only repeat the seed word of the question.

    A primary column is an echo column when it holds one and the same value on every gold row and
    that value is written as a string literal in the gold query (e.g. ?italianWord = "donna" for
    FILTER(STR(?italianWord) = "donna")). An answer that leaves it out is not wrong, so these columns
    are not required. Nothing is dropped when every primary column is an echo column.
    """
    primary = list(gold_answer_vars.get("primary", []))
    if len(primary) < 2 or not gold_results:
        return []
    literals = {m.strip().strip("^$").lower() for m in re.findall(r'"((?:[^"\\]|\\.)*)"', gold_sparql)}
    out = []
    for var in primary:
        values = {normalize_value(r.get(var, "")).lower() for r in gold_results}
        if len(values) == 1:
            v = next(iter(values))
            if v and v in literals:
                out.append(var)
    return [] if len(out) == len(primary) else out


def _column_kind(rows: list[dict], var: str, sample: int = 50) -> Optional[str]:
    """'iri' if the bound values of var (in a sample of rows) are all IRIs, 'literal' otherwise, None if unbound."""
    values = [r[var] for r in rows[:sample] if r.get(var)]
    if not values:
        return None
    return "iri" if all(v.startswith(("http://", "https://", "urn:", "_:")) for v in values) else "literal"


def _is_literal_column(rows: list[dict], var: str, sample: int = 50) -> bool:
    """True if the bound values of var (in a sample of rows) are non-IRI, non-numeric literals."""
    values = [r[var] for r in rows[:sample] if r.get(var)]
    if not values:
        return False
    for v in values:
        if v.startswith(("http://", "https://", "urn:", "_:")):
            return False
        try:
            float(v)
            return False
        except ValueError:
            pass
    return True


def compute_f1(
    gold_results: list[dict[str, str]],
    predicted_results: list[dict[str, str]],
    answer_vars: dict,
    variable_mapping: dict[str, str],
    numeric_vars: Optional[set[str]] = None,
) -> F1Result:
    """Compute F1 score between gold and predicted result sets.

    Args:
        gold_results: Gold standard result rows
        predicted_results: Predicted result rows
        answer_vars: Dict with primary, secondary, aggregates, numeric lists
        variable_mapping: Mapping from gold var names to predicted var names
        numeric_vars: Set of variable names that should be compared numerically

    Returns:
        F1Result with precision, recall, f1, and details
    """
    if numeric_vars is None:
        numeric_vars = set(answer_vars.get("numeric", []))

    tuple_vars = answer_vars.get("primary", [])
    aggregate_vars = answer_vars.get("aggregates", [])

    # IRI/literal equivalents: compare a literal prediction with the gold's literal column
    gold_cols = list(tuple_vars)
    required_gold_cols = []
    for k, var in enumerate(tuple_vars):
        alt = answer_vars.get("equivalents", {}).get(var)
        if alt and _column_kind(predicted_results, variable_mapping.get(var, var)) == "literal":
            gold_cols[k] = alt
            required_gold_cols.append(alt)
    if required_gold_cols:
        gold_results = [r for r in gold_results if all(r.get(c) for c in required_gold_cols)]

    def extract_tuple(row: dict, var_list: list[str], var_map: dict, is_gold: bool) -> tuple:
        values = []
        for k, var in enumerate(var_list):
            if is_gold:
                raw = row.get(gold_cols[k], "")
            else:
                mapped = var_map.get(var, var)
                raw = row.get(mapped, "")
            is_num = var in numeric_vars
            values.append(normalize_value(raw, is_numeric=is_num))
        return tuple(values)

    if tuple_vars:
        gold_tuples = Counter(
            extract_tuple(row, tuple_vars, variable_mapping, is_gold=True)
            for row in gold_results
        )
        pred_tuples = Counter(
            extract_tuple(row, tuple_vars, variable_mapping, is_gold=False)
            for row in predicted_results
        )

        true_positives = 0
        for t in gold_tuples:
            if t in pred_tuples:
                true_positives += min(gold_tuples[t], pred_tuples[t])

        total_gold = sum(gold_tuples.values())
        total_predicted = sum(pred_tuples.values())
    else:
        true_positives = min(len(gold_results), len(predicted_results))
        total_gold = len(gold_results)
        total_predicted = len(predicted_results)

    precision = true_positives / total_predicted if total_predicted > 0 else 0.0
    recall = true_positives / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    aggregate_score = None
    aggregate_details: dict = {}

    if aggregate_vars and gold_results and predicted_results:
        gold_row = gold_results[0]
        pred_row = predicted_results[0]
        matches = 0
        total_aggs = len(aggregate_vars)

        for agg_var in aggregate_vars:
            gold_val = normalize_value(gold_row.get(agg_var, ""), is_numeric=True)
            pred_mapped = variable_mapping.get(agg_var, agg_var)
            pred_val = normalize_value(pred_row.get(pred_mapped, ""), is_numeric=True)

            is_match = gold_val == pred_val
            if not is_match and agg_var.lower().startswith(("avg", "average", "mean")):
                try:
                    is_match = math.isclose(float(gold_val), float(pred_val), rel_tol=AVG_REL_TOLERANCE)
                except ValueError:
                    pass
            aggregate_details[agg_var] = {
                "gold": gold_val,
                "predicted": pred_val,
                "match": is_match,
            }
            if is_match:
                matches += 1

        aggregate_score = matches / total_aggs if total_aggs > 0 else None

        if not tuple_vars and aggregate_score is not None:
            f1 = aggregate_score
            precision = aggregate_score
            recall = aggregate_score

    return F1Result(
        test_id=0,  # caller sets this
        precision=precision,
        recall=recall,
        f1=f1,
        gold_count=total_gold,
        predicted_count=total_predicted,
        true_positives=true_positives,
        aggregate_score=aggregate_score,
        aggregate_details=aggregate_details,
        variable_mapping=variable_mapping,
    )


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class F1Evaluator:
    """F1 Score on Answers evaluator.

    Executes gold and predicted SPARQL queries against the endpoint and
    compares their result sets.

    Usage:
        evaluator = F1Evaluator()
        result = evaluator.evaluate_single(gold_sparql, predicted_sparql, answer_vars)
        print(f"F1: {result.f1:.3f}")

    Args:
        endpoint: SPARQL endpoint URL
        timeout: Query timeout in seconds
        max_results: Maximum results to fetch per query
        cache_gold_results: Whether to cache gold query results
    """

    def __init__(
        self,
        endpoint: str = LIITA_ENDPOINT,
        timeout: int = 60,
        max_results: int = 10000,
        cache_gold_results: bool = True,
        strip_limit: bool = True,
        strip_predicted_limit: Optional[bool] = None,  # NL2SPARQL's name for strip_limit
        column_mapping: str = "value",  # "value" (default since 2026-11-13) or "name" (previous behaviour)
        ignore_echo_columns: bool = True,  # default since 2026-11-14; False reproduces the 2026-11-13 scores
    ):
        self.ignore_echo_columns = ignore_echo_columns
        if column_mapping not in ("value", "name"):
            raise ValueError(f"column_mapping must be 'value' or 'name', not {column_mapping!r}")
        self.column_mapping = column_mapping
        if strip_predicted_limit is not None:
            strip_limit = strip_predicted_limit
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_results = max_results
        self.cache_gold_results = cache_gold_results
        self.strip_limit = strip_limit
        self._gold_cache: dict[int, QueryExecutionResult] = {}

    def evaluate_single(
        self,
        gold_sparql: str,
        predicted_sparql: str,
        answer_variables: dict,
        test_id: int = 0,
    ) -> F1Result:
        """Evaluate a single test case by comparing gold and predicted results.

        Args:
            gold_sparql: Gold standard SPARQL query
            predicted_sparql: Predicted SPARQL query
            answer_variables: Dict with primary, secondary, aggregates, numeric
            test_id: Identifier for this test case

        Returns:
            F1Result with scores and details
        """
        if self.strip_limit:
            gold_sparql = strip_limit_offset(gold_sparql)
            predicted_sparql = strip_limit_offset(predicted_sparql)

        if self.cache_gold_results and test_id in self._gold_cache:
            gold_exec = self._gold_cache[test_id]
        else:
            gold_exec = execute_query_full(
                gold_sparql, self.endpoint, self.timeout, self.max_results,
            )
            if self.cache_gold_results:
                self._gold_cache[test_id] = gold_exec

        if not gold_exec.success:
            return F1Result(
                test_id=test_id,
                precision=0.0, recall=0.0, f1=0.0,
                gold_count=0, predicted_count=0, true_positives=0,
                gold_error=gold_exec.error,
            )

        pred_exec = execute_query_full(
            predicted_sparql, self.endpoint, self.timeout, self.max_results,
        )

        if not pred_exec.success:
            return F1Result(
                test_id=test_id,
                precision=0.0, recall=0.0, f1=0.0,
                gold_count=gold_exec.result_count, predicted_count=0,
                true_positives=0,
                predicted_error=pred_exec.error,
            )

        if gold_exec.result_count == 0 and pred_exec.result_count == 0:
            return F1Result(
                test_id=test_id,
                precision=1.0, recall=1.0, f1=1.0,
                gold_count=0, predicted_count=0, true_positives=0,
            )

        ignored = echo_columns(answer_variables, gold_sparql, gold_exec.results) if self.ignore_echo_columns else []
        if ignored:
            answer_variables = {**answer_variables,
                                "primary": [v for v in answer_variables.get("primary", []) if v not in ignored]}

        var_mapping = build_variable_mapping(answer_variables, predicted_sparql, pred_exec.results)
        if self.column_mapping == "value":
            var_mapping = build_value_mapping(
                answer_variables, gold_exec.results, pred_exec.results, predicted_sparql, var_mapping,
            )

        numeric_vars = set(answer_variables.get("numeric", []))
        numeric_vars.update(answer_variables.get("aggregates", []))

        result = compute_f1(
            gold_results=gold_exec.results,
            predicted_results=pred_exec.results,
            answer_vars=answer_variables,
            variable_mapping=var_mapping,
            numeric_vars=numeric_vars,
        )

        result.test_id = test_id
        result.ignored_columns = ignored
        return result

    def evaluate_dataset(
        self,
        test_data: dict,
        translator=None,
        language: str = "en",
    ) -> F1Report:
        """Evaluate a full test dataset.

        Args:
            test_data: Loaded test dataset dict with "test_cases" key
            translator: Optional translator with translate() method
            language: Language key for NL questions ("en", "it")

        Returns:
            F1Report with aggregate metrics
        """
        test_cases = test_data["test_cases"]

        if self.cache_gold_results:
            self.prefetch_gold_results(test_data)

        results: list[F1Result] = []
        skipped = 0
        f1_by_category: dict[str, list[float]] = {}
        f1_by_pattern: dict[str, list[float]] = {}

        for tc in test_cases:
            gold_sparql = tc.get("sparql")
            answer_vars = tc.get("answer_variables", {})
            test_id = tc["id"]

            if not gold_sparql:
                skipped += 1
                continue

            if translator is not None:
                question = tc.get(f"nl_{language}", tc.get("nl_en", ""))
                try:
                    translation = translator.translate(question)
                    predicted_sparql = translation.sparql
                except Exception as e:
                    results.append(F1Result(
                        test_id=test_id,
                        precision=0.0, recall=0.0, f1=0.0,
                        gold_count=0, predicted_count=0, true_positives=0,
                        predicted_error=str(e),
                    ))
                    continue
            else:
                predicted_sparql = gold_sparql

            if not predicted_sparql:
                results.append(F1Result(
                    test_id=test_id,
                    precision=0.0, recall=0.0, f1=0.0,
                    gold_count=0, predicted_count=0, true_positives=0,
                    predicted_error="Empty predicted SPARQL",
                ))
                continue

            stored_sparql = strip_limit_offset(predicted_sparql) if self.strip_limit else predicted_sparql

            f1_result = self.evaluate_single(
                gold_sparql, predicted_sparql, answer_vars, test_id,
            )
            f1_result.predicted_sparql = stored_sparql

            if f1_result.gold_error:
                skipped += 1
                continue

            results.append(f1_result)

            category = tc.get("category", "unknown")
            f1_by_category.setdefault(category, []).append(f1_result.f1)

            for pattern in tc.get("patterns", []):
                f1_by_pattern.setdefault(pattern, []).append(f1_result.f1)

        evaluated = len(results)
        if evaluated > 0:
            avg_precision = sum(r.precision for r in results) / evaluated
            avg_recall = sum(r.recall for r in results) / evaluated
            avg_f1 = sum(r.f1 for r in results) / evaluated
        else:
            avg_precision = avg_recall = avg_f1 = 0.0

        cat_avgs = []
        cat_report = {}
        for cat, scores in f1_by_category.items():
            cat_avg = sum(scores) / len(scores) if scores else 0.0
            cat_report[cat] = {
                "avg_f1": cat_avg,
                "count": len(scores),
            }
            cat_avgs.append(cat_avg)

        macro_f1 = sum(cat_avgs) / len(cat_avgs) if cat_avgs else 0.0

        pattern_report = {}
        for pat, scores in f1_by_pattern.items():
            pattern_report[pat] = {
                "avg_f1": sum(scores) / len(scores) if scores else 0.0,
                "count": len(scores),
            }

        return F1Report(
            total_evaluated=evaluated,
            total_skipped=skipped,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            macro_f1=macro_f1,
            f1_by_category=cat_report,
            f1_by_pattern=pattern_report,
            results=results,
        )

    def prefetch_gold_results(self, test_data: dict) -> None:
        """Pre-execute all gold queries and store in cache.

        Args:
            test_data: Loaded test dataset dict with "test_cases" key
        """
        for tc in test_data["test_cases"]:
            test_id = tc["id"]
            gold_sparql = tc.get("sparql")

            if test_id in self._gold_cache or not gold_sparql:
                continue

            if self.strip_limit:
                gold_sparql = strip_limit_offset(gold_sparql)

            result = execute_query_full(
                gold_sparql, self.endpoint, self.timeout, self.max_results,
            )
            self._gold_cache[test_id] = result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_f1_report(report: F1Report, path: str) -> None:
    """Save an F1Report to a JSON file.

    Args:
        report: F1Report returned by F1Evaluator.evaluate_dataset()
        path:   Output file path (JSON)
    """
    import json

    data = {
        "summary": {
            "total_evaluated": report.total_evaluated,
            "total_skipped": report.total_skipped,
            "avg_precision": report.avg_precision,
            "avg_recall": report.avg_recall,
            "avg_f1": report.avg_f1,
            "macro_f1": report.macro_f1,
        },
        "by_category": report.f1_by_category,
        "by_pattern": report.f1_by_pattern,
        "results": [
            {
                "test_id": r.test_id,
                "f1": r.f1,
                "precision": r.precision,
                "recall": r.recall,
                "gold_count": r.gold_count,
                "predicted_count": r.predicted_count,
                "true_positives": r.true_positives,
                "aggregate_score": r.aggregate_score,
                "aggregate_details": r.aggregate_details,
                "variable_mapping": r.variable_mapping,
                "predicted_sparql": r.predicted_sparql,
                "gold_error": r.gold_error,
                "predicted_error": r.predicted_error,
            }
            for r in report.results
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
