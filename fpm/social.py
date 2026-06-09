"""Social Process Mining — Heuristic Miner on integrated federated logs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pm4py

from fpm.loader import ACTIVITY, CASE_ID, TIMESTAMP


@dataclass
class DiscoveryResult:
    net: Any
    initial_marking: Any
    final_marking: Any
    heuristics_net: Any
    discovery_time_s: float
    stats: dict[str, Any]


class SocialProcessMiner:
    """Discover SOW models from an integrated event log."""

    def discover_heuristics_net(self, integrated_log: pd.DataFrame):
        return pm4py.discover_heuristics_net(
            integrated_log,
            activity_key=ACTIVITY,
            case_id_key=CASE_ID,
            timestamp_key=TIMESTAMP,
        )

    def discover(self, integrated_log: pd.DataFrame) -> tuple[Any, Any, Any]:
        return pm4py.discover_petri_net_heuristics(
            integrated_log,
            activity_key=ACTIVITY,
            case_id_key=CASE_ID,
            timestamp_key=TIMESTAMP,
        )

    @staticmethod
    def sum_arc_weights(heuristics_net) -> int:
        total = 0
        for targets in heuristics_net.dfg_matrix.values():
            total += sum(targets.values())
        return total

    @staticmethod
    def dfg_arc_count(heuristics_net) -> int:
        return sum(len(targets) for targets in heuristics_net.dfg_matrix.values())

    def model_stats(
        self,
        net,
        heuristics_net,
        integrated_log: pd.DataFrame,
    ) -> dict[str, Any]:
        labeled_transitions = [
            transition.label for transition in net.transitions if transition.label
        ]
        trace_count = len(pm4py.get_event_attribute_values(integrated_log, CASE_ID))

        return {
            "activities": len(labeled_transitions),
            "heuristics_net_nodes": len(heuristics_net.nodes),
            "arcs": len(net.arcs),
            "dfg_arcs": self.dfg_arc_count(heuristics_net),
            "sum_arc_weights": self.sum_arc_weights(heuristics_net),
            "places": len(net.places),
            "transitions": len(net.transitions),
            "traces": trace_count,
            "events": len(integrated_log),
        }

    def discover_with_stats(self, integrated_log: pd.DataFrame) -> DiscoveryResult:
        if integrated_log.empty:
            raise ValueError("Cannot discover a SOW model from an empty integrated log.")

        start = time.perf_counter()
        heuristics_net = self.discover_heuristics_net(integrated_log)
        net, initial_marking, final_marking = self.discover(integrated_log)
        discovery_time_s = time.perf_counter() - start

        stats = self.model_stats(net, heuristics_net, integrated_log)
        stats["discovery_time_s"] = round(discovery_time_s, 6)

        return DiscoveryResult(
            net=net,
            initial_marking=initial_marking,
            final_marking=final_marking,
            heuristics_net=heuristics_net,
            discovery_time_s=discovery_time_s,
            stats=stats,
        )

    @staticmethod
    def write_artifacts(
        net,
        initial_marking,
        final_marking,
        output_dir: Path,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        pnml_path = output_dir / "model.pnml"
        png_path = output_dir / "model.png"

        pm4py.write_pnml(net, initial_marking, final_marking, str(pnml_path))
        pm4py.save_vis_petri_net(
            net,
            initial_marking,
            final_marking,
            str(png_path),
        )

        return {"pnml": pnml_path, "png": png_path}
