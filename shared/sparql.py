"""
SPARQL execution utilities shared across the Gradio app and notebooks.
"""

import requests

LIITA_ENDPOINT = "https://liita.it/sparql"
DEFAULT_TIMEOUT = 90


def fetch_sparql(
    sparql: str,
    limit: int = 20,
    endpoint: str = LIITA_ENDPOINT,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """
    Execute a SPARQL SELECT query and return the raw results dict.

    Appends LIMIT if not already present.
    Raises RuntimeError on HTTP errors (includes server response body).
    """
    if "LIMIT" not in sparql.upper():
        sparql = sparql.rstrip().rstrip(";") + f"\nLIMIT {limit}"

    response = requests.get(
        endpoint,
        params={"query": sparql, "format": "application/sparql-results+json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=timeout,
    )

    if not response.ok:
        body = response.text[:500] if response.text else "(empty body)"
        raise RuntimeError(f"SPARQL endpoint returned {response.status_code}: {body}")

    return response.json()
