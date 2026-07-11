"""Tests for `neurofour-net5` -- the direct POLICY classifier (PART 2/P1+P2
of the gen-9 task; see `net5.py`/`train_net5.py` module docstrings, incl. the
correction re: the original `neurofour-net` also being lookahead-free).
Unlike every DEEP-SEARCH net (net0/net0b/net0d/net1/net2/net4), `select_move`
does NOT construct any child boards for a value-lookahead ranking -- it reads
the CURRENT board's policy logits directly (masked to the legal columns)
after an optional 0-param tactical guard.
"""
import ast
import io
import json
import os
import tokenize

import pytest

from tests._oracle_scan import assert_no_oracle_access

from app.engine.board import Board, WIDTH
from app.agents.net5 import Net5Agent, DEFAULT_ARTIFACT
from app.agents import registry
from app.neurogolf import strength
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")
_NET5_SRC = os.path.join(_ROOT, "app", "agents", "net5.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_ARTIFACT),
    reason="neurofour-net5's artifact not trained yet",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- agent behaviour + manifest -------------------------------------------- #

def test_agent_returns_legal_and_deterministic():
    ag = Net5Agent()
    for p in _load(_SEALED)[:80]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


def test_agent_never_misses_win_or_block_when_guard_enabled():
    ag = Net5Agent(use_guard=True)
    for p in _load(_SEALED):
        b = Board.from_moves(p["board"])
        moves = b.legal_moves()
        wins = [c for c in moves if b.winning_move(c)]
        m = ag.select_move(b)
        if wins:
            assert m in wins, f"missed an immediate win at {p['board']}"
            continue
        from app.engine.board import _won, _bottom_mask, _column_mask
        opp = b.mask ^ b.cur
        blocks = [c for c in moves
                  if _won(opp | ((b.mask + _bottom_mask(c)) & _column_mask(c)))]
        if blocks:
            assert m in blocks, f"missed a forced block at {p['board']}"


def test_manifest_is_honest_and_within_caps():
    ag = Net5Agent()
    man = ag.manifest()
    assert man.kind == "nn"
    assert man.size_bytes == os.path.getsize(ag.artifact_path)
    assert man.size_bytes <= 32_768           # micro tier
    assert man.flops_per_move <= FLOP_CAP
    assert man.params > 0


def test_manifest_flops_is_single_forward_pass_not_width_scaled():
    """The whole point of net5: ONE forward pass over the current board, not
    WIDTH child-board forward passes like every 1-ply value-lookahead agent
    (net/net0/net1/net2/net4/net0b/net0d)."""
    from app.agents.encode import FEATURE_DIM
    ag_guard = Net5Agent(use_guard=True)
    ag_noguard = Net5Agent(use_guard=False)
    params = ag_guard.manifest().params
    expected_noguard = 2 * params + FEATURE_DIM
    expected_guard = expected_noguard + 4 * WIDTH
    assert ag_noguard.manifest().flops_per_move == expected_noguard
    assert ag_guard.manifest().flops_per_move == expected_guard
    # sanity: dramatically cheaper than a WIDTH-scaled 1-ply agent of similar params
    assert ag_guard.manifest().flops_per_move < WIDTH * (2 * params + FEATURE_DIM)


def test_qualifies_micro_budget_and_latency():
    import time
    ag = Net5Agent()
    man = ag.manifest()
    assert man.size_bytes <= 32_768
    assert man.flops_per_move < FLOP_CAP
    rows = _load(_SEALED)[:200]
    boards = [Board.from_moves(p["board"]) for p in rows]
    for b in boards[:8]:
        ag.select_move(b)   # warmup
    timings = []
    for b in boards:
        t0 = time.perf_counter()
        ag.select_move(b)
        timings.append((time.perf_counter() - t0) * 1000.0)
    timings.sort()
    p50 = timings[len(timings) // 2]
    assert p50 < LATENCY_CAP_MS


def test_registered_in_registry():
    names = registry.agent_names()
    assert "neurofour-net5" in names
    # unrelated agents untouched
    assert "neurofour-net1" in names
    assert "neurofour-net4" in names


# ---- anti-cheat: no solver / oracle access at inference -------------------- #

def test_source_has_no_solver_or_oracle_access():
    """Transitive scan (see tests/_oracle_scan.py, gen-9 T1): the OLD
    single-file scan (docstrings/comments excluded, imports of THIS file
    only) could be routed around by importing the oracle through a helper
    module -- net5.py imports app.agents.encode, which legitimately
    imports ONLY the two allowlisted pure board-math helpers from
    app.solver.solver; the new scan walks the FULL app.** import graph
    reachable from net5.py, and additionally forbids private Solver
    methods, dynamic solver imports, and label-file reads anywhere in
    that closure. See tests/test_oracle_scan_selftest.py for the proof
    this scan can actually fail on a laundered import."""
    assert_no_oracle_access('app.agents.net5')
