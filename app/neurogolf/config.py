"""Benchmark constants from METRIC.md."""
from __future__ import annotations

import os

LATENCY_CAP_MS = 50.0
# gen-14 raise-flop-cap frame: overridable via NEUROFOUR_FLOP_CAP so a sweep
# script can probe higher compute ceilings without touching the committed
# default (still 5,000,000 -- byte-identical to every prior generation
# unless the env var is explicitly set). LATENCY_CAP_MS is intentionally
# left a hard constant (never env-overridable) -- the whole point of the
# sweep is to test whether the SAME latency ceiling still binds a deeper
# search, not to move the goalposts.
FLOP_CAP = int(os.environ.get("NEUROFOUR_FLOP_CAP", "5000000"))

# METRIC.md sec.9: "latency_ms ... is reported but never used in the
# pass/fail gate -- only in intra-tier tiebreaks, and there rounded to a
# stable bucket". Every agent measured on this reference sandbox sits well
# under 1ms (single numpy forward passes / small alpha-beta searches over a
# 194-dim feature vector), while process-scheduling / concurrent-load noise
# on a shared machine can easily swing a sub-millisecond wall-clock
# measurement by 2-10x run-to-run. A 1ms bucket comfortably absorbs that
# noise for every agent shipped so far while still discriminating a genuinely
# slower agent (one that's actually >=1ms slower, not just noisier) in a
# tiebreak.
LATENCY_BUCKET_MS = 1.0

# Single source of truth for "how many stones must already be on the board
# before an exact (mate-distance, scored-mode) pure-Python solve is fast
# enough to call inline". bench_data/dev.jsonl and bench_data/sealed.jsonl
# never contain a position with fewer stones than this (see
# tests/test_perfect_agent.py::test_exact_solve_min_ply_covers_all_labelled_positions
# and app/neurogolf/positions.py's default min_ply), so any agent -- in
# particular the `perfect` reference agent -- that falls through to an exact
# solve at this ply (instead of a non-exact heuristic) is guaranteed to be
# able to answer every strength-scored position exactly, even if a cache /
# opening-book lookup misses. `/analyze` (app/main.py) uses the same
# constant for the same reason: below it, a full solve risks a multi-minute
# pure-Python search from a near-empty board.
EXACT_SOLVE_MIN_PLY = 14

# budget tiers by size_bytes (name, cap)  -- ordered small -> large
TIERS = [
    ("nano", 4_096),
    ("micro", 32_768),
    ("mini", 262_144),
    ("small", 2_097_152),
    ("open", float("inf")),
]

# opening books for the ladder: empty board + a fixed diverse set of short openings
OPENING_BOOKS = [
    [],
    [3],
    [3, 3],
    [2],
    [4],
    [3, 2],
    [3, 4],
    [2, 3],
    [0],
    [3, 3, 2],
]


def seed() -> int:
    try:
        return int(os.environ.get("NEUROFOUR_SEED", "4"))
    except ValueError:
        return 4


def tier_for(size_bytes: int) -> str:
    for name, cap in TIERS:
        if size_bytes <= cap:
            return name
    return "open"


# rank index for tie-breaking (smaller tier = better)
TIER_RANK = {name: i for i, (name, _cap) in enumerate(TIERS)}
