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

The agentic translator uses an **LLM to decompose** queries into tool operations.

### How It Works

1. **Present tools** to the LLM with descriptions and parameters
2. **LLM analyzes** the query and produces a structured plan (JSON)
3. **Validate** the plan (check tool existence, dependencies, parameters)
4. **Convert** tool calls to BlockCalls with appropriate slots
5. **Build QuerySpec** from the plan

### Available Tools

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

Query: `"Find synonyms of 'origine' whose Sicilian translation starts with 'm'"`

LLM Plan:
```json
{
  "reasoning": "Need to find synonyms, join to LiITA, translate to Sicilian, then filter",
  "steps": [
    {"tool": "find_semantic_relations", "params": {"seed_word": "origine", "relation_type": "synonym"}},
    {"tool": "join_to_liita", "params": {"source_var": "?wordRel"}},
    {"tool": "translate_to_sicilian", "params": {}},
    {"tool": "filter_variable", "params": {"variable": "?sicWR", "pattern_type": "prefix", "pattern_text": "m"}}
  ],
  "output_vars": ["?itLemmaString", "?sicWR"]
}
```

### Strengths & Limitations

**Strengths:**
- Handles complex, multi-step queries
- Understands natural language nuances
- Can combine operations in novel ways

**Limitations:**
- Requires LLM API access (cost, latency)
- Output depends on LLM quality
- May occasionally produce invalid plans (validated and rejected)

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
pip install google-generativeai

# For Ollama (local, no pip install needed - just run Ollama)
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
llm = create_llm_client(provider="mistral", api_key="your-key", model="mistral-large-latest")
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
├── data/
│   └── ontology_filtered.json
├── shared/
│   └── llm.py             # LLM client abstraction
└── mosaic_liita/
    ├── __init__.py        # Public API exports
    ├── blocks.py          # Block, BlockRegistry, make_registry()
    ├── query.py           # QuerySpec, QueryPlan, validation
    ├── planner.py         # Deterministic Planner
    ├── assembler.py       # SPARQL Assembler
    ├── agent.py           # QueryAgent, tools, agentic translation
    ├── relations.py       # Semantic relation resolution
    ├── constants.py       # PREFIXES, SERVICE URIs
    └── utils.py           # Helper functions
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LiITA Project](https://liita.it) - Interlinking linguistic resources for Italian via Linked Data
- [CompL-IT](https://klab.ilc.cnr.it/graphdb-compl-it/) - Computational Lexicon for Italian
