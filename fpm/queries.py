"""Shared LTL scenario queries and helpers for the FPM pipeline."""

from __future__ import annotations

import re

# Paper Section 7 notation (SOWCompact):
#   r = Finally, □ = Next, □ = Globally, : = negation, ! = material implication,
#   _ = or, ^ = and
PAPER_SCENARIO_QUERIES = {
    "scenario1_shopping_mealprep": "r(Shopping!□Mealpreparation)",
    "scenario2_no_sport": "r(:Sport)",
    "scenario3_movement_transportation": "r(Movement!Transportation)",
    "scenario4_social_eat_transport": "r(Socializing!□EatingDrinking!Transportation)",
    "scenario5_no_eat_no_social": "r(:EatingDrinking!□:Socializing)",
}

# Trace-level LTL used by the resolver. These follow the paper's *intent* while
# using ASCII operators our engine understands. See README "Paper ↔ implementation".
SCENARIO_QUERIES = {
    "scenario1_shopping_mealprep": "F(Shopping & X(Mealpreparation))",
    "scenario2_no_sport": "G(!Sport)",
    # r(Movement!Transportation): sequence — Movement then later Transportation
    "scenario3_movement_transportation": "F(Movement & X(F Transportation))",
    "scenario4_social_eat_transport": (
        "F(Socializing & X(F(EatingDrinking & X(F Transportation))))"
    ),
    "scenario5_no_eat_no_social": "G(!EatingDrinking) & G(!Socializing)",
}

# Direct symbol mapping only (naive translation — matches almost all traces).
PAPER_SYMBOL_MAP = {
    "r": "F",
    "□": "X",
    "◻": "X",
    "☐": "X",
    ":": "!",
    "_": "|",
    "^": "&",
    "$": "<->",
}


def paper_to_ascii_literal(paper_query: str) -> str:
    """Map paper symbols to ASCII LTL without changing semantics.

    Example: ``r(:Sport)`` → ``F(!Sport)``. Material implication ``!`` becomes ``->``.
    """
    text = paper_query.strip()
    if text.startswith("r(") and text.endswith(")"):
        text = "F(" + text[2:-1] + ")"

    # Paper ! is implication; protect ASCII -> first, then swap remaining !.
    text = text.replace("->", "\x00")
    text = text.replace("!", "->")
    text = text.replace("\x00", "->")

    for paper_sym, ascii_sym in PAPER_SYMBOL_MAP.items():
        text = text.replace(paper_sym, ascii_sym)
    return text


def resolve_paper_scenario(scenario_name: str) -> str:
    """Return the trace-level ASCII query for a named paper scenario."""
    try:
        return SCENARIO_QUERIES[scenario_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown scenario {scenario_name!r}. "
            f"Choose from: {sorted(SCENARIO_QUERIES)}"
        ) from exc


def query_slug(query_text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", query_text).strip("_")
    return slug[:80] or "query"
