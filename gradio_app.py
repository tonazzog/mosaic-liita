#!/usr/bin/env python3
"""
MoSAIC-LiITA - Gradio Web UI
============================

User-friendly interface for the deterministic NL -> SPARQL pipeline.
Shows the assembled query and the execution plan built by the planner.

Usage:
    python gradio_app.py
    python gradio_app.py --share
"""

import argparse
import json
import traceback
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import requests

from mosaic_liita import Assembler, Planner, make_registry, BlockRegistry, QueryAgent
from shared.llm import create_llm_client


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent / "data"
ONTOLOGY_PATH = DATA_DIR / "ontology_filtered.json"
LIITA_ENDPOINT = "https://liita.it/sparql"
DEFAULT_TIMEOUT = 30

EXAMPLE_QUERIES = [
    "Find CompL-IT definitions starting with 'uccello' and show Parmigiano and Sicilian translation.",
    "Find definitions of words starting with 'ante'",
    "Find Italian words that express positive emotions ('gioia') and are hyponyms of 'sentimento'.",
    "Find Sicilian lemmas ending with 'u' and translate them into Italian.",
    "Find synonyms of 'antico' with their senses.",
]

LLM_PROVIDERS = ["mistral", "anthropic", "openai", "gemini", "ollama"]

DEFAULT_MODELS = {
    "mistral": "mistral-large-latest",
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "gemini": "gemini-1.5-pro",
    "ollama": "llama3.1",
}


# ============================================================================
# Global State
# ============================================================================

registry: Optional[BlockRegistry] = None
planner: Optional[Planner] = None
assembler: Optional[Assembler] = None
catalog: Optional[list] = None  # For QueryAgent


# ============================================================================
# Initialization
# ============================================================================

def init_pipeline() -> str:
    """Initialize planner and assembler."""
    global registry, planner, assembler, catalog

    try:
        with open(ONTOLOGY_PATH, "r", encoding="utf-8") as file:
            catalog_data = json.load(file)

        catalog = catalog_data["documents"]
        registry = make_registry()
        planner = Planner(registry, catalog)
        assembler = Assembler()

        return "Pipeline ready"
    except Exception as e:
        registry = None
        planner = None
        assembler = None
        catalog = None
        return f"Initialization failed: {e}"


# ============================================================================
# Formatting
# ============================================================================

def format_plan(plan) -> str:
    lines = []

    lines.append(f"### Blocks: {len(plan.blocks)}\n")

    for i, bi in enumerate(plan.blocks, start=1):
        lines.append(f"**{i}. {bi.block.id}**")

        if bi.block.service_iri:
            lines.append(f"- Service: {bi.block.service_iri}")

        if bi.block.requires:
            req = ", ".join(sorted(bi.block.requires))
            lines.append(f"- Requires: {req}")

        if bi.block.provides:
            prov = ", ".join(sorted(bi.block.provides))
            lines.append(f"- Provides: {prov}")

        if bi.slots:
            slot_items = []
            for k, v in bi.slots.items():
                if v:
                    slot_items.append(f"{k}={v}")
            if slot_items:
                lines.append(f"- Slots: {', '.join(slot_items)}")

        lines.append("")

    # Plan metadata
    lines.append("### Plan Metadata\n")
    if plan.select_vars:
        lines.append(f"- Select: {' '.join(plan.select_vars)}")
    if plan.aggregates:
        agg_items = [f"{k}={v}" for k, v in plan.aggregates.items()]
        lines.append(f"- Aggregates: {', '.join(agg_items)}")
    if plan.group_by:
        lines.append(f"- Group by: {' '.join(plan.group_by)}")
    if plan.having:
        lines.append(f"- Having: {plan.having}")
    if plan.order_by:
        lines.append(f"- Order by: {plan.order_by}")
    if plan.limit:
        lines.append(f"- Limit: {plan.limit}")

    return "\n".join(lines)


def format_agent_plan(agent_plan) -> str:
    """Format an AgentPlan for display."""
    lines = []

    lines.append("### Agent Reasoning\n")
    lines.append(f"{agent_plan.reasoning}\n")

    lines.append(f"### Tool Calls: {len(agent_plan.steps)}\n")

    for i, step in enumerate(agent_plan.steps, start=1):
        lines.append(f"**{i}. {step.tool}**")
        if step.params:
            for k, v in step.params.items():
                lines.append(f"  - {k}: `{v}`")
        lines.append("")

    lines.append("### Output Variables\n")
    lines.append(f"{', '.join(agent_plan.output_vars)}\n")

    if agent_plan.aggregation:
        lines.append("### Aggregation\n")
        lines.append(f"- Count: {agent_plan.aggregation.get('count_variable', '?')}")
        if agent_plan.aggregation.get('group_by'):
            lines.append(f"- Group by: {', '.join(agent_plan.aggregation['group_by'])}")

    return "\n".join(lines)


# ============================================================================
# SPARQL Execution
# ============================================================================

def execute_sparql(sparql: str, limit: int = 20) -> str:
    """Execute a SPARQL query against the LiITA endpoint."""
    if not sparql.strip():
        return "Please enter a SPARQL query"

    try:
        if "LIMIT" not in sparql.upper():
            sparql = sparql.rstrip().rstrip(";") + f"\nLIMIT {limit}"

        response = requests.post(
            LIITA_ENDPOINT,
            data={"query": sparql},
            headers={
                "Accept": "application/sparql-results+json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=DEFAULT_TIMEOUT,
        )

        if response.status_code != 200:
            return f"**Error:** HTTP {response.status_code}\n\n```\n{response.text}\n```"

        data = response.json()
        variables = data.get("head", {}).get("vars", [])
        bindings = data.get("results", {}).get("bindings", [])

        output = f"### Results: {len(bindings)} rows\n\n"
        output += f"**Variables:** {', '.join(variables)}\n\n"

        if bindings:
            output += "| " + " | ".join(variables) + " |\n"
            output += "| " + " | ".join(["---"] * len(variables)) + " |\n"

            for row in bindings[:limit]:
                values = []
                for var in variables:
                    if var in row:
                        val = row[var].get("value", "")
                        if len(val) > 50:
                            val = val[:47] + "..."
                        values.append(val)
                    else:
                        values.append("")
                output += "| " + " | ".join(values) + " |\n"

            if len(bindings) > limit:
                output += f"\n*Showing first {limit} of {len(bindings)} results*"
        else:
            output += "*No results found*"

        return output

    except requests.exceptions.Timeout:
        return "**Error:** Query timed out"
    except requests.exceptions.RequestException as e:
        return f"**Error:** Request failed: {str(e)}"
    except json.JSONDecodeError as e:
        return f"**Error:** Invalid JSON response: {str(e)}"
    except Exception as e:
        return f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"


# ============================================================================
# Translation
# ============================================================================

def translate_deterministic(question: str) -> Tuple[str, str, str]:
    """
    Translate using the deterministic planner (no LLM).

    Returns: (sparql, plan_md, status)
    """
    if not planner or not assembler or not registry:
        return "", "", "Pipeline not initialized"

    if not question.strip():
        return "", "", "Please enter a question"

    try:
        spec = planner.plan(question)
        plan = spec.compile(registry)
        sparql = assembler.assemble(plan)
        plan_md = format_plan(plan)

        return sparql, plan_md, "Translation successful (deterministic)"
    except Exception as e:
        error_md = f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return "", error_md, "Translation failed"


def translate_agentic(
    question: str,
    llm_provider: str = "mistral",
    llm_api_key: str = "",
    llm_model: str = "",
) -> Tuple[str, str, str]:
    """
    Translate using the LLM-powered QueryAgent for decomposition.

    Returns: (sparql, plan_md, status)
    """
    if not registry or not catalog:
        return "", "", "Pipeline not initialized"

    if not question.strip():
        return "", "", "Please enter a question"

    # Agentic mode requires an LLM
    if llm_provider != "ollama" and not llm_api_key.strip():
        return "", "", "Agentic mode requires an API key (configure in Settings tab, or use Ollama)"

    try:
        model = llm_model.strip() or DEFAULT_MODELS.get(llm_provider, "")
        client = create_llm_client(
            provider=llm_provider,
            api_key=llm_api_key.strip() if llm_api_key.strip() else None,
            model=model,
        )

        agent = QueryAgent(registry, catalog, client)
        sparql, agent_plan, query_spec = agent.translate(question)

        # Format the agent plan for display
        plan_md = format_agent_plan(agent_plan)

        # Add QuerySpec details
        plan_md += "\n\n---\n\n### Generated QuerySpec\n"
        plan_md += f"- Blocks: {len(query_spec.blocks)}\n"
        for bc in query_spec.blocks:
            plan_md += f"  - {bc.block_id}\n"
        if query_spec.select_vars:
            plan_md += f"- Select: {', '.join(query_spec.select_vars)}\n"
        if query_spec.aggregates:
            plan_md += f"- Aggregates: {query_spec.aggregates}\n"

        return sparql, plan_md, f"Translation successful (agentic via {llm_provider})"

    except Exception as e:
        error_md = f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return "", error_md, "Translation failed"


def translate_query(
    question: str,
    mode: str = "Deterministic",
    llm_provider: str = "mistral",
    llm_api_key: str = "",
    llm_model: str = "",
) -> Tuple[str, str, str]:
    """
    Translate a natural language question to SPARQL.

    Args:
        question: Natural language query
        mode: "Deterministic" or "Agentic"
        llm_provider: LLM provider name (for Agentic mode)
        llm_api_key: API key for the provider (for Agentic mode)
        llm_model: Model name (uses default if empty)

    Returns: (sparql, plan_md, status)
    """
    if mode == "Agentic":
        return translate_agentic(question, llm_provider, llm_api_key, llm_model)
    else:
        return translate_deterministic(question)


# ============================================================================
# Gradio UI
# ============================================================================

def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    with gr.Blocks(title="MoSAIC-LiITA - NL to SPARQL") as app:
        gr.Markdown("""
        # MoSAIC-LiITA

        NL -> SPARQL translation using reusable blocks.
        Choose between **Deterministic** (rule-based) or **Agentic** (LLM-powered) mode.
        """)

        with gr.Tabs():
            # ================================================================
            # Translate Tab
            # ================================================================
            with gr.TabItem("Translate"):
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Natural Language Question",
                        placeholder="e.g., Find definitions of words starting with 'ante'",
                        lines=2,
                        scale=4,
                    )
                    translate_btn = gr.Button("Translate", variant="primary", scale=1)

                mode_selector = gr.Radio(
                    choices=["Deterministic", "Agentic"],
                    value="Deterministic",
                    label="Translation Mode",
                    info="Deterministic: Rule-based planner (no LLM). Agentic: LLM decomposes query into tool calls (configure LLM in Settings tab).",
                )

                status_output = gr.Markdown(label="Status")

                with gr.Row():
                    with gr.Column(scale=2):
                        sparql_output = gr.Code(
                            label="Generated SPARQL",
                            language="sql",
                            lines=16,
                        )
                        with gr.Row():
                            copy_btn = gr.Button("Copy to Execute Tab", size="sm")
                    with gr.Column(scale=1):
                        plan_output = gr.Markdown(label="Execution Plan / Agent Reasoning")

                gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUERIES],
                    inputs=[question_input],
                    label="Example Queries",
                )

            # ================================================================
            # Execute SPARQL Tab
            # ================================================================
            with gr.TabItem("Execute SPARQL"):
                gr.Markdown("""
                ### Execute SPARQL Query

                Paste queries from the Translate tab or write your own.
                """)

                exec_sparql_input = gr.Code(
                    label="SPARQL Query",
                    language="sql",
                    lines=12,
                    value="""PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX lila: <http://lila-erc.eu/ontologies/lila/>
PREFIX lexinfo: <http://www.lexinfo.net/ontology/3.0/lexinfo#>

SELECT ?lemma ?writtenRep
WHERE {
    ?lemma a ontolex:LexicalEntry ;
           ontolex:canonicalForm ?form ;
           lexinfo:partOfSpeech lexinfo:noun .
    ?form ontolex:writtenRep ?writtenRep .
}
LIMIT 10""",
                )

                with gr.Row():
                    exec_limit = gr.Slider(
                        minimum=5, maximum=100, value=20, step=5,
                        label="Result Limit",
                    )
                    exec_btn = gr.Button("Execute Query", variant="primary")

                exec_output = gr.Markdown(label="Results")

                exec_btn.click(
                    execute_sparql,
                    inputs=[exec_sparql_input, exec_limit],
                    outputs=[exec_output],
                )

            # ================================================================
            # Settings Tab
            # ================================================================
            with gr.TabItem("Settings"):
                gr.Markdown("""
                ### LLM Settings

                Configure the LLM provider for **Agentic mode**. These settings are only used
                when Agentic mode is selected in the Translate tab.
                """)

                llm_provider = gr.Dropdown(
                    choices=LLM_PROVIDERS,
                    value="mistral",
                    label="LLM Provider",
                )

                llm_model = gr.Textbox(
                    label="Model (optional)",
                    placeholder="Leave empty to use default model for the provider",
                    info=f"Defaults: {', '.join(f'{k}={v}' for k, v in DEFAULT_MODELS.items())}",
                )

                llm_api_key = gr.Textbox(
                    label="API Key",
                    placeholder="Enter your API key (not needed for Ollama)",
                    type="password",
                )

                gr.Markdown("""
                ---
                **Note:** API keys are not stored and must be re-entered each session.
                For Ollama, no API key is required (runs locally).
                """)

            # ================================================================
            # Event Handlers
            # ================================================================
            translate_btn.click(
                translate_query,
                inputs=[question_input, mode_selector, llm_provider, llm_api_key, llm_model],
                outputs=[sparql_output, plan_output, status_output],
            )

            question_input.submit(
                translate_query,
                inputs=[question_input, mode_selector, llm_provider, llm_api_key, llm_model],
                outputs=[sparql_output, plan_output, status_output],
            )

            copy_btn.click(
                lambda x: x,
                inputs=[sparql_output],
                outputs=[exec_sparql_input],
            )

    return app


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="MoSAIC-LiITA - Gradio Web UI")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MoSAIC-LiITA - Gradio Web UI")
    print("=" * 60)
    print("Initializing pipeline...")
    print(init_pipeline())
    print("Starting web UI...")

    app = create_ui()
    app.launch(
        share=args.share,
        server_port=args.port,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
