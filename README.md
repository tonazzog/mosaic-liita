# MoSAIC-LiITA

**MoSAIC** — **Mo**dular **S**parql **A**ssembler for **I**nterlinked **C**orpora

A modular Natural Language to SPARQL translator for the [LiITA](https://liita.it) (Linking Italian) knowledge base.

## Overview

MoSAIC-LiITA translates natural language queries about Italian linguistics into SPARQL queries. It uses a **block-based architecture** where reusable SPARQL patterns (blocks) are assembled to form complete queries — like tiles in a mosaic.

The system offers two translation modes:

| Mode | Description | LLM Required |
|------|-------------|--------------|
| **Deterministic** | Rule-based planner using keyword matching and patterns | No |
| **Agentic** | LLM decomposes queries into tool operations | Yes |

Both modes use the same underlying block system, ensuring consistent and valid SPARQL output.

## Thesis Version and Related Systems

MoSAIC is one of three NL-to-SPARQL systems for LiITA developed for the master's thesis *Querying LiITA in Natural Language: Three Neuro-Symbolic Approaches to NL-to-SPARQL Translation*:

- [PRISMA](https://github.com/tonazzog/prisma-per-liita) — the LLM classifies the intent, SPARQL is assembled by rules;
- **MoSAIC** (this repository) — the LLM composes typed blocks, SPARQL is assembled deterministically;
- [NL2SPARQL](https://github.com/tonazzog/nl2sparql) — the LLM generates SPARQL, grounded by retrieved examples and constraints.

The version evaluated in the thesis is tagged **`thesis-final-2026-11-15`**. It includes the *revised agentic mode* described below (tree-structured plans); the deterministic mode is the original one.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Natural Language Query                       │
└─────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌──────────────────────┐            ┌──────────────────────┐
│   DETERMINISTIC      │            │      AGENTIC         │
│                      │            │                      │
│  Planner (rules)     │            │  QueryAgent (LLM)    │
│  - Keyword matching  │            │  - Tool selection    │
│  - Pattern detection │            │  - Parameter mapping │
│  - Relation resolver │            │  - Multi-step plans  │
└──────────────────────┘            └──────────────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    ┌──────────────────┐
                    │    QuerySpec     │
                    │  (Block calls +  │
                    │   output config) │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  Block Registry  │
                    │  (SPARQL blocks) │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │    QueryPlan     │
                    │ (Ordered blocks) │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │    Assembler     │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   SPARQL Query   │
                    └──────────────────┘
```

## Block System

### What is a Block?

A **Block** is a reusable SPARQL pattern with:
- **ID**: Unique identifier (e.g., `COMPLIT_SEMREL_OF_SEED_LEMMA`)
- **Requires**: Variables that must be available (e.g., `{?liitaLemma}`)
- **Provides**: Variables this block produces (e.g., `{?sicWR, ?sicLemma}`)
- **Prefixes**: Required namespace prefixes
- **Where**: SPARQL triple patterns with optional `{slot}` placeholders
- **Service IRI**: Optional federated query endpoint

### Example Block

```python
Block(
    id="TRANSLATE_TO_SICILIANO",
    requires={"?liitaLemma"},
    provides={"?sicLemma", "?sicWR"},
    prefixes={"ontolex", "vartrans", "dcterms"},
    where=[
        "?itToSicEntry ontolex:canonicalForm ?liitaLemma ;",
        "             vartrans:translatableAs ?sicEntry .",
        "?sicEntry ontolex:canonicalForm ?sicLemma .",
        "?sicLemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> ;",
        "         ontolex:writtenRep ?sicWR .",
    ],
)
```

### Block Dependencies

Blocks declare what they **require** and **provide**. The system uses topological sorting to order blocks so that dependencies are satisfied:

```
COMPLIT_SEMREL_OF_SEED_LEMMA  →  provides: ?wordRel, ?itLemmaString
         ↓
JOIN_WORDREL_TO_LIITA         →  requires: ?wordRel, provides: ?liitaLemma
         ↓
TRANSLATE_TO_SICILIANO        →  requires: ?liitaLemma, provides: ?sicWR
```

## Deterministic Mode

The deterministic planner uses **rule-based pattern matching** to select blocks.

### How It Works

1. **Normalize** the input query (lowercase, collapse whitespace)
2. **Extract** quoted strings, patterns (starts with, ends with, contains)
3. **Detect** intent through keyword matching:
   - Semantic relations: "synonyms", "hyponyms", "antonyms", etc.
   - Definitions: "definition", "meaning"
   - Translations: "Sicilian", "Parmigiano"
   - Sentiment/Emotions: "polarity", "sentiment", "emotions"
4. **Select blocks** based on detected intents
5. **Fill slots** with extracted values (seed words, patterns, filters)
6. **Build QuerySpec** with appropriate SELECT variables and aggregations

### Example

Query: `"Find synonyms of 'antico' with Sicilian translations"`

Detection:
- "synonyms of" → semantic relation query
- `'antico'` → seed word
- "Sicilian translations" → needs translation block

Selected blocks:
1. `COMPLIT_SEMREL_OF_SEED_LEMMA` (slots: seed_lemma="antico", rel_triple for synonyms)
2. `JOIN_WORDREL_TO_LIITA`
3. `TRANSLATE_TO_SICILIANO`

### Strengths & Limitations

**Strengths:**
- Fast (no API calls)
- Predictable output
- No cost

**Limitations:**
- Limited to patterns the rules recognize
- Can't handle novel query formulations
- May miss nuances in complex queries

## Agentic Mode

The agentic translator uses an **LLM to decompose** queries into tool operations. The LLM never writes SPARQL: it produces a JSON plan, and the SPARQL is built deterministically from the blocks.

### How It Works

1. **Present tools** to the LLM with descriptions, parameters, and the variables each tool requires and provides
2. **LLM analyzes** the query and produces a structured plan (JSON)
3. **Build** the query from the plan: steps become blocks, each step's variables are renamed with the step id, operators become `FILTER NOT EXISTS`, `UNION`, `FILTER` and sub-queries
4. **Check** the query on the endpoint; if the plan cannot be built, or the query fails, returns no rows or only zero aggregates, the problem is returned to the LLM for **one round of repair**

### Plan Language

A plan is a tree of steps. Every step has a short `id`, may declare as `input` the step it builds on, and exposes its variables as `<id>.<var>`; a parameter can be a reference (`{"ref": "a.wordRel"}`) instead of a literal, for instance to follow a chain of relations. Five operators can be nested:

| Operator | Meaning |
|----------|---------|
| `not_exists` | keep the results for which the inner steps have no match (negation) |
| `union` | results matching at least one of several branches |
| `compare` | comparison between two variables, or between a variable and a value |
| `reference` | a single value computed separately (e.g. the polarity of a reference word), usable in `compare` |
| `aggregate` | one value per group (e.g. the number of emotions of each word), with optional `HAVING` |

The planner prompt is in `mosaic_liita/prompt_v2.py`. Plans in the original flat format (a list of tool calls without ids) are still accepted.

### Available Tools

The main tools are listed below; in total the agentic mode offers 28 tools, including grammatical features (gender, degree, inflection), hypolemmas and lemma variants, CompL-it semantic types and traits, functional and derivational relations, aggregation tools and metadata (`mosaic_liita/agent.py`, `mosaic_liita/v2_tools.py`).

| Tool | Description | Provides |
|------|-------------|----------|
| `find_semantic_relations` | Find hyponyms, synonyms, etc. of a word | ?wordRel, ?itLemmaString |
| `find_definitions_by_pattern` | Search definitions by pattern | ?word, ?definition |
| `find_liita_lemmas_by_pattern` | Search Italian lemmas | ?lemma, ?wr |
| `find_sicilian_lemmas_by_pattern` | Search Sicilian lemmas | ?sicLemma, ?sicWR |
| `find_parmigiano_lemmas_by_pattern` | Search Parmigiano lemmas | ?parLemma, ?parWR |
| `join_to_liita` | Connect CompL-IT to LiITA | ?liitaLemma |
| `translate_to_sicilian` | Get Sicilian translations | ?sicWR |
| `translate_to_parmigiano` | Get Parmigiano translations | ?parWR |
| `get_sentiment` | Get Sentix polarity | ?polarityLabel |
| `get_emotions` | Get ELIta emotions | ?emotionLabel |
| `filter_variable` | Add pattern filter | - |
| `count_results` | Count instead of list | ?count |

### Example

Query: `"Which synonyms of 'origine' have no translation into Sicilian?"`

LLM Plan:
```json
{
  "steps": [
    {"id": "a", "tool": "find_semantic_relations",
     "params": {"seed_word": "origine", "relation_type": "synonym"}},
    {"id": "j", "tool": "join_to_liita", "input": "a", "params": {"source_var": "?wordRel"}},
    {"id": "n", "op": "not_exists", "steps": [
      {"id": "t", "tool": "translate_to_sicilian", "input": "j", "params": {}}]}
  ],
  "output_vars": ["a.wordRel", "a.itLemmaString"],
  "aggregation": null
}
```

The builder renders the translation step inside `FILTER NOT EXISTS { ... }`; on the live endpoint the query returns the four synonyms without a Sicilian translation (*causa*, *filiazione*, *genesi*, *germe*).

### Strengths & Limitations

**Strengths:**
- Handles complex, multi-step queries
- Understands natural language nuances
- Can combine operations in novel ways

**Limitations:**
- Requires LLM access (cost, latency)
- Output depends on LLM quality
- May occasionally produce invalid plans; one round of repair is attempted
- Computations over aggregates (per-group statistics, nested aggregation, several statistics in one answer) are its weakest point

## Installation

### Requirements

- Python 3.9+
- Dependencies (install via pip):

```bash
pip install gradio requests
```

For LLM support (agentic mode), you'll also need the appropriate SDK:
```bash
# For Mistral
pip install mistralai

# For OpenAI
pip install openai

# For Anthropic
pip install anthropic

# For Google Gemini
pip install google-genai

# For Ollama (local, no pip install needed - just run Ollama)

# For Claude Code (provider "claudecode"): no SDK and no API key; requires a local
# installation of the Claude Code CLI, logged in (https://docs.claude.com/en/docs/claude-code)
```

### Data Files

The system requires an ontology catalog file:
- `data/ontology_filtered.json` - Contains property metadata for relation resolution

## Running the Gradio App

### Basic Usage

```bash
python gradio_app.py
```

This starts the web UI at `http://localhost:7860`.

### Options

```bash
# Create a public shareable link
python gradio_app.py --share

# Use a different port
python gradio_app.py --port 8080
```

### Using the UI

1. **Translate Tab**
   - Enter your natural language query
   - Select mode: **Deterministic** or **Agentic**
   - Click **Translate**
   - View the generated SPARQL and execution plan

2. **Execute SPARQL Tab**
   - Paste or write SPARQL queries
   - Execute against the LiITA endpoint
   - View results in a table

3. **Settings Tab** (for Agentic mode)
   - Select LLM provider (Mistral, OpenAI, Anthropic, Gemini, Ollama)
   - Enter model name (optional, uses defaults)
   - Enter API key (not needed for Ollama)

### Example Queries

```
Find synonyms of 'antico' with their senses.
Find definitions of words starting with 'ante'
Find Sicilian lemmas ending with 'u' and translate them into Italian.
Find Italian words that express positive emotions ('gioia') and are hyponyms of 'sentimento'.
How many lemmas are present in Sicilian lexicon?
Find synonyms of 'origine' whose Sicilian translation starts with 'm'
```

## Programmatic Usage

### Deterministic Translation

```python
from mosaic_liita import Planner, Assembler, make_registry
import json

# Load catalog
with open("data/ontology_filtered.json") as f:
    catalog = json.load(f)["documents"]

# Initialize
registry = make_registry()
planner = Planner(registry, catalog)
assembler = Assembler()

# Translate
spec = planner.plan("Find synonyms of 'antico'")
plan = spec.compile(registry)
sparql = assembler.assemble(plan)

print(sparql)
```

### Agentic Translation

```python
from mosaic_liita import QueryAgent, make_registry
from shared.llm import create_llm_client
import json

# Load catalog
with open("data/ontology_filtered.json") as f:
    catalog = json.load(f)["documents"]

# Initialize
registry = make_registry()
llm = create_llm_client(provider="mistral", api_key="your-key", model="mistral-large-4-0")
agent = QueryAgent(registry, catalog, llm)

# Translate
sparql, plan, spec = agent.translate("Find synonyms of 'origine' with Sicilian translations")

print("Plan:", plan.reasoning)
print("SPARQL:", sparql)
```

## Project Structure

```
mosaic-liita/
├── gradio_app.py          # Web UI
├── demo.ipynb             # Jupyter notebook demo
├── requirements.txt       # Python dependencies
├── data/
│   ├── ontology_filtered.json   # Property metadata for relation resolution
│   └── test_dataset.json        # 100-case NL→SPARQL benchmark
├── evaluation/
│   ├── __init__.py              # Package exports (F1Evaluator, F1Report, …)
│   ├── f1_evaluator.py          # F1-on-answers evaluator
│   └── mosaic_translator.py     # MoSAIC adapter + SPARQL de-aggregation
├── scripts/
│   └── run_f1_evaluation.py     # CLI evaluation runner
├── reports/                     # Saved evaluation JSON reports
├── shared/
│   └── llm.py                   # LLM client abstraction (multi-provider)
└── mosaic_liita/
    ├── __init__.py              # Public API exports
    ├── blocks.py                # Block, BlockRegistry, make_registry()
    ├── query.py                 # QuerySpec, QueryPlan, validation
    ├── planner.py               # Deterministic Planner
    ├── assembler.py             # SPARQL Assembler
    ├── agent.py                 # QueryAgent, tools, agentic translation
    ├── relations.py             # Semantic relation resolution
    ├── llm_refinement.py        # Optional LLM-based SPARQL refinement
    ├── constants.py             # PREFIXES, SERVICE URIs
    └── utils.py                 # Helper functions
```

## Evaluation

MoSAIC-LiITA includes an evaluation pipeline that executes generated and gold queries on the LiITA endpoint and compares their answers.

### About the Evaluation

The evaluator (`evaluation/f1_evaluator.py`) is the one used in the thesis; an identical copy is used by PRISMA, MoSAIC and NL2SPARQL, so that their scores are comparable. It executes the gold and the predicted query on the LiITA endpoint and computes precision, recall and F1 on the answers. Columns are matched **by value** (the predicted column whose values overlap most with each gold column), `LIMIT`/`OFFSET` are removed from both queries, a gold column that only repeats the seed word of the question (an *echo* column) is not required, and IRI and literal answers are accepted when the gold names both.

`data/test_dataset.json` is the 100-question set of the first evaluation. In the thesis it serves as a **regression set**; the final evaluation used a separate test set of 86 questions, frozen before the systems were revised. Both are described in Chapter 6 of the thesis.

### Regression Set

The regression set (`data/test_dataset.json`) contains **100 NL→SPARQL pairs**:

| Category | Count | Description |
|----------|-------|-------------|
| `complex` | 56 | Multi-feature queries (emotion + translation, semantic relation + translation, etc.) |
| `emotion` | 9 | Queries using the ELIta emotion lexicon |
| `semantic_combined` | 29 | Semantic relation queries combined with other features |
| `translation` | 6 | Pure dialect translation queries |

Each test case also carries one or more **pattern tags** that describe the SPARQL features required:

| Pattern | Count | Meaning |
|---------|-------|---------|
| `EMOTION_LEXICON` | 36 | Requires ELIta emotion annotations |
| `TRANSLATION` | 29 | Requires `vartrans:translatableAs` dialect links |
| `MULTI_TRANSLATION` | 28 | Requires both Sicilian and Parmigiano translations |
| `SERVICE_INTEGRATION` | 48 | Requires a federated `SERVICE` call to CompL-IT |
| `SEMANTIC_RELATION` | 17 | Requires lexinfo hyponym/hypernym/meronym triples |
| `POS_FILTER` | 21 | Requires `lila:hasPOS` filtering |
| `MORPHO_REGEX` | 13 | Requires morphological pattern filtering |
| `COUNT_ENTITIES` | 15 | Requires `COUNT`/`AVG` aggregation |
| `SENSE_DEFINITION` | 18 | Requires `skos:definition` lookup |
| `LEXICAL_FORM` | 7 | Requires `ontolex:writtenRep` enumeration |
| `META_GRAPH` | 23 | Queries directly over the LiITA GRAPH |

Each case specifies the **expected answer variables** (classified as primary/secondary/aggregates/numeric) and has a **gold SPARQL query** verified to execute correctly against the LiITA endpoint.

### Running the Evaluation

#### Deterministic mode (no LLM required)

```bash
python scripts/run_f1_evaluation.py --mode deterministic
```

#### Agentic mode

```bash
# Mistral AI (cloud)
python scripts/run_f1_evaluation.py \
    --mode agentic \
    --provider mistral \
    --model mistral-large-4-0 \
    --api-key YOUR_KEY

# Anthropic
python scripts/run_f1_evaluation.py \
    --mode agentic \
    --provider anthropic \
    --model claude-haiku-4-5-20251001 \
    --api-key YOUR_KEY

# Ollama (local, no key needed)
python scripts/run_f1_evaluation.py \
    --mode agentic \
    --provider ollama \
    --model mistral-large-3:675b-cloud
```

#### All parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `deterministic` | `deterministic` (rule-based planner) or `agentic` (LLM) |
| `--provider` | — | LLM provider: `mistral`, `anthropic`, `openai`, `gemini`, `ollama`, `claudecode` (Claude Code CLI, no API key). Required for agentic mode. |
| `--dataset` | `data/test_dataset.json` | Dataset to evaluate (same JSON format) |
| `--model` | provider default | Model identifier (e.g. `mistral-large-4-0`, `claude-haiku-4-5-20251001`) |
| `--api-key` | env variable | API key; falls back to the provider's standard env variable if omitted |
| `--language` | `en` | NL question language to use: `en` or `it` |
| `-o` / `--output` | auto | Output JSON path. Defaults to `reports/f1_report_mosaic_<mode>[_<provider>_<model>].json` |
| `--timeout` | `60` | SPARQL endpoint timeout in seconds per query |
| `--delay` | `0` | Minimum seconds between LLM calls (hard RPS cap) |
| `--tpm-limit` | `0` | Token-per-minute budget cap. The evaluator maintains a 60-second sliding window and sleeps automatically when the budget is exhausted. Set this to your API tier's TPM limit. |
| `--tokens-per-call` | auto | Override the per-call token estimate used for TPM accounting (default: system prompt size + 40 tokens) |
| `--no-prefetch` | — | Disable pre-fetching of all gold query results before translation starts |
| `--no-strip-limit` | — | Disable stripping of `LIMIT`/`OFFSET` clauses before result-set comparison |
| `--no-deaggregate` | — | Disable de-aggregation post-processing (ablation study) |
| `--baseline` | — | Use gold SPARQL as prediction — F1 should be ≈1.0. Tests the evaluator infrastructure. |
| `--rerun-errors` | — | Re-evaluate only test cases that have a `predicted_error` in the existing report, then merge results back |
| `--error-filter SUBSTR` | — | Combined with `--rerun-errors`: only re-run cases whose error message contains `SUBSTR` |
| `--test-ids ID[,ID...]` | — | Evaluate only the specified test IDs (comma-separated). If the output file already exists, results are merged back in — useful to recover previously skipped cases. |

#### Rate-limiting example (Mistral free tier: 500 000 TPM)

```bash
python scripts/run_f1_evaluation.py \
    --mode agentic \
    --provider mistral \
    --model mistral-large-4-0 \
    --tpm-limit 500000 \
    --delay 1.2
```

The `--tpm-limit` and `--delay` constraints are independent and both enforced simultaneously. Using both provides a safety margin against burst usage.

#### Patching skipped cases

If a gold query times out during a run, the case is skipped. To recover it without re-running everything:

```bash
python scripts/run_f1_evaluation.py \
    --mode agentic --provider ollama --model mistral-large-3:675b-cloud \
    --test-ids 2149,2150 --timeout 120 \
    -o reports/f1_report_mosaic_agentic_ollama_mistral-large-3-675b-cloud.json
```

### Report Format

Reports are saved as JSON files in the `reports/` directory, with a `summary` (evaluated and skipped cases, average precision, recall and F1, macro-F1), breakdowns `by_category` and `by_pattern`, and one entry per question in `results`:

```json
{
  "test_id": 2054,
  "f1": 1.0, "precision": 1.0, "recall": 1.0,
  "gold_count": 1, "predicted_count": 1, "true_positives": 1,
  "aggregate_score": 1.0,
  "aggregate_details": {"count": {"gold": "78442.0", "predicted": "78442.0", "match": true}},
  "variable_mapping": {"count": "count"},
  "predicted_sparql": "SELECT (COUNT(DISTINCT ?lemma) AS ?count) WHERE { ... }",
  "gold_error": null,
  "predicted_error": null
}
```

Each result entry records the gold and predicted result-set sizes, true positives, the inferred variable mapping, and the generated SPARQL for inspection. Cases where the gold query itself fails (e.g., endpoint timeout) are counted in `total_skipped` and omitted from the `results` array. Cases where MoSAIC fails to generate a plan are included with `f1: 0.0` and a non-null `predicted_error`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


