"""Regression coverage for the `perfect` reference agent's benchmark integrity.

Background: PerfectAgent.select_move looks up a memoised transposition cache
(app/data/solver_cache.json) before ever calling the live solver, purely as a
speed optimisation. Before this fix, the cache-miss fallback for positions
with fewer than FAST_PLY=16 stones was a non-exact, depth-6, static-eval
search (MinimaxAgent) -- NOT the exact mate-distance solver. Because
bench_data/dev.jsonl and bench_data/sealed.jsonl can contain positions with as
few as 14 stones, a cache miss on one of those (stale/corrupted/incomplete
solver_cache.json) could silently make `perfect` prefer a losing move over a
drawing one (a genuine horizon effect in the depth-6 heuristic), while the
published leaderboard card kept claiming optimality=1.0 / blunder_rate=0.0.

The fix aligns PerfectAgent.FAST_PLY with app.neurogolf.config.EXACT_SOLVE_MIN_PLY
(14), the same floor `/analyze` already used, which is <= the minimum stone
count of every labelled dev/sealed position -- so a cache miss on any position
that can ever be strength-scored now always takes the exact-solve path.
"""
from __future__ import annotations

import json
import os

import pytest

from app.engine.board import Board
from app.agents.registry import make_agent
from app.agents.baselines import PerfectAgent, MinimaxAgent
from app.neurogolf import strength
from app.neurogolf.config import EXACT_SOLVE_MIN_PLY
from app.neurogolf.positions import load_set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEALED = os.path.join(ROOT, "bench_data", "sealed.jsonl")
DEV = os.path.join(ROOT, "bench_data", "dev.jsonl")
LEADERBOARD = os.path.join(ROOT, "bench_data", "leaderboard.json")

# The exact repro board from the benchmark-integrity audit: a draw is
# available (col 4, value 0) but a naive depth-limited heuristic prefers a
# losing move (col 5, value -1) on a genuine horizon effect.
REPRO_MOVES = "1,4,3,1,5,1,2,0,6,5,1,6,0,6"


def test_repro_board_plays_the_drawing_move():
    b = Board.from_moves(REPRO_MOVES)
    assert b.n == 14
    agent = make_agent("perfect")
    assert agent.select_move(b) == 4


def test_repro_board_off_cache_still_plays_the_drawing_move():
    """Even with the memoised cache entry removed (simulating a stale/missing
    cache), `perfect` must still land on the exact answer -- this is the
    actual regression: it used to fall through to a non-exact heuristic that
    played col 5 (a loss) instead."""
    b = Board.from_moves(REPRO_MOVES)
    agent = PerfectAgent()
    key = b.to_key()
    agent._cache.pop(key, None)
    agent._book.pop(key, None)
    assert agent.select_move(b) == 4


def test_naive_depth_limited_fallback_would_have_blundered_here():
    """Documents the actual root cause: MinimaxAgent(6)'s static-eval search
    (the OLD off-cache fallback) genuinely prefers the losing move on this
    position -- it's a horizon effect, not a mate-distance/tie-break quirk.
    This is why the fallback must never be reachable for a strength-scored
    position (see test_repro_board_off_cache_still_plays_the_drawing_move)."""
    b = Board.from_moves(REPRO_MOVES)
    assert MinimaxAgent(6).select_move(b) == 5


def test_exact_solve_min_ply_covers_all_labelled_positions():
    """The exact-solve floor must be <= the shallowest position that can ever
    be strength-scored, so a cache miss can never downgrade to the non-exact
    fallback for a position that matters."""
    for path in (DEV, SEALED):
        rows = load_set(path)
        min_n = min(Board.from_moves(r["board"]).n for r in rows)
        assert min_n >= EXACT_SOLVE_MIN_PLY, (
            f"{path} has a position with only {min_n} stones, below "
            f"EXACT_SOLVE_MIN_PLY={EXACT_SOLVE_MIN_PLY}"
        )
    assert PerfectAgent.FAST_PLY == EXACT_SOLVE_MIN_PLY


def test_perfect_is_robust_to_missing_cache_entries_on_sealed_positions():
    """Sample several genuinely shallow (14-15 stone) sealed positions, strip
    their cache entries, and confirm the live exact-solve fallback still
    lands on an optimal move -- not just this one repro board."""
    rows = load_set(SEALED)
    shallow = [r for r in rows if Board.from_moves(r["board"]).n <= 15][:6]
    assert shallow, "expected some shallow (<=15 stone) sealed positions"
    agent = PerfectAgent()
    for r in shallow:
        b = Board.from_moves(r["board"])
        key = b.to_key()
        agent._cache.pop(key, None)
        agent._book.pop(key, None)
        mv = agent.select_move(b)
        assert mv in r["optimal_cols"], (
            f"board={r['board']} played {mv}, optimal={r['optimal_cols']}"
        )


def test_perfect_card_is_genuinely_1_0_0_0_on_sealed():
    """Drive `perfect` through strength.score exactly like eval_agent.py /
    run_bench.py do -- the live, freshly-measured card must be perfect."""
    rows = load_set(SEALED)
    agent = make_agent("perfect")
    card = strength.score(agent, rows)
    assert card.optimality == 1.0
    assert card.blunder_rate == 0.0


def test_perfect_live_card_matches_published_leaderboard_card():
    if not os.path.exists(LEADERBOARD):
        pytest.skip("leaderboard.json not built")
    with open(LEADERBOARD, "r", encoding="utf-8") as f:
        lb = json.load(f)
    published = next((a for a in lb["agents"] if a["name"] == "perfect"), None)
    if published is None:
        pytest.skip("perfect not present in committed leaderboard.json")
    rows = load_set(SEALED)
    agent = make_agent("perfect")
    card = strength.score(agent, rows)
    assert round(card.optimality, 6) == published["optimality"]
    assert round(card.blunder_rate, 6) == published["blunder_rate"]
    assert published["optimality"] == 1.0
    assert published["blunder_rate"] == 0.0
