"""Example LTL queries for CASAS2 grouped prediction.

Atoms use the M07_ON token form (underscore). The grouped pipeline maps
CASAS2 event labels like M07=ON to this form automatically.
"""

from __future__ import annotations

EXAMPLE_QUERIES = {
    "all_clients": "",
    "m07_then_m23": "F(M07_ON & X(F M23_ON))",
    "m01_occurs": "F(M01_ON)",
    "m08_then_m09": "F(M08_ON & X(F M09_ON))",
    "no_m14": "G(!M14_ON)",
    "kitchen_flow": "F(M07_ON & X(F M08_ON))",
}
