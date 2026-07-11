"""Regression coverage for scripts/run_bench.py's `--check` blind spot.

Before this fix, `--check` only asserted "fresh recompute == the committed
bench_data/leaderboard.json" (self-consistency). That means a `perfect` card
that had drifted from optimality=1.0 could, in principle, be "fixed" by
simply re-running `run_bench.py` (no --check) and committing the new
(self-consistent but wrong) leaderboard.json -- nothing ever asserted the
absolute invariant "the reference/ceiling agent must actually be optimal".

`_check_reference_invariants` closes that gap: it is evaluated against the
FRESH recompute, independent of whatever the committed file says, so a
regressed `perfect` fails both `run_bench.py` (can't even be written) and
`run_bench.py --check` (can't slip through as "self-consistent").
"""
from __future__ import annotations

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

run_bench = importlib.import_module("run_bench")


def _fake_records(perfect_optimality=1.0, perfect_blunder=0.0):
    return [
        {"name": "cheapnet", "kind": "nn", "optimality": 0.92, "blunder_rate": 0.05,
         "soundness": 0.95, "per_outcome": {}, "size_bytes": 6000, "params": 2500,
         "flops_per_move": 5000, "flops_plausible": True, "latency_ms": 1.0,
         "over_budget": False, "elo": 500},
        {"name": "perfect", "kind": "search", "optimality": perfect_optimality,
         "blunder_rate": perfect_blunder, "soundness": 1.0 - perfect_blunder,
         "per_outcome": {}, "size_bytes": 0, "params": 0,
         "flops_per_move": 50_000_000, "flops_plausible": True, "latency_ms": 80.0,
         "over_budget": True, "elo": 1200},
    ]


def test_reference_invariant_passes_when_perfect_is_genuinely_perfect():
    problems = run_bench._check_reference_invariants(_fake_records())
    assert problems == []


def test_reference_invariant_fails_when_perfect_card_is_wrong():
    problems = run_bench._check_reference_invariants(
        _fake_records(perfect_optimality=0.9833, perfect_blunder=0.0033))
    assert problems
    assert "perfect" in problems[0]
    assert "0.9833" in problems[0]


def test_reference_invariant_fails_on_nonzero_blunder_rate_even_if_optimality_ok():
    # a degenerate/rounding case: optimality could round to 1.0 while
    # blunder_rate is still nonzero (different aggregation) -- both must hold.
    problems = run_bench._check_reference_invariants(
        _fake_records(perfect_optimality=1.0, perfect_blunder=0.001))
    assert problems


def test_reference_invariant_ignores_agents_not_present():
    # a build without `perfect` registered must not spuriously fail
    records = [r for r in _fake_records() if r["name"] != "perfect"]
    assert run_bench._check_reference_invariants(records) == []


def test_strip_latency_flip_sanity_still_trips_the_consistency_guard():
    """Sanity check requested by the audit: corrupting a published number
    must make the committed-vs-recomputed comparison disagree."""
    committed = {"headline": {"value": 0.956667, "agent": "neurofour-net2"},
                 "agents": [{"name": "perfect", "optimality": 1.0,
                             "blunder_rate": 0.0, "latency_ms": 0.003}]}
    recomputed_same = {"headline": {"value": 0.956667, "agent": "neurofour-net2"},
                        "agents": [{"name": "perfect", "optimality": 1.0,
                                    "blunder_rate": 0.0, "latency_ms": 0.09}]}
    # latency differs but is stripped -> still equal
    assert run_bench._strip_latency(committed) == run_bench._strip_latency(recomputed_same)

    corrupted = {"headline": {"value": 0.956667, "agent": "neurofour-net2"},
                 "agents": [{"name": "perfect", "optimality": 0.9833,
                             "blunder_rate": 0.0033, "latency_ms": 0.003}]}
    assert run_bench._strip_latency(committed) != run_bench._strip_latency(corrupted)
