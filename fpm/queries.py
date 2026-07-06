"""Example LTL queries for filtering Chinook sensor clients."""

from __future__ import annotations

import re

EXAMPLE_QUERIES = {
    "all_clients": "",
    "half_clients_i03": "F(I03_PRESENT)",
    "m07_then_m23": "F(M07_ON & X(F M23_ON))",
    "m01_occurs": "F(M01_ON)",
    "m08_then_m09": "F(M08_ON & X(F M09_ON))",
    "no_m14": "G(!M14_ON)",
}


def query_slug(query_text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", query_text).strip("_")
    return slug[:80] or "query"
