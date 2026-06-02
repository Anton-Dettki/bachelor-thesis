"""Shared LTL scenario queries and helpers for the FPM pipeline."""

from __future__ import annotations

import re

SCENARIO_QUERIES = {
    "scenario1_shopping_mealprep": "F(Shopping & X(F Mealpreparation))",
    "scenario2_no_sport": "G(!Sport)",
    "scenario3_movement_transportation": "F(Movement & X(F Transportation))",
    "scenario4_social_eat_transport": (
        "F(Socializing & X(F(EatingDrinking & X(F Transportation))))"
    ),
    "scenario5_no_eat_no_social": "G(!EatingDrinking) & G(!Socializing)",
}


def query_slug(query_text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", query_text).strip("_")
    return slug[:80] or "query"
