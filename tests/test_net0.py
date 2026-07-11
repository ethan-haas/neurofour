"""Tests for the NANO-tier learned agent `neurofour-net0`.

Covers:
  * the 0-param tactical guard invariants (identical contract to net1/net2);
  * the agent loads, always returns a legal move, is deterministic;
  * manifest honesty: size == real artifact bytes, size fits the NANO cap
    (<=4096 bytes), flops within the caps;
  * it clears the headline bar (sealed optimality > 0.900, i.e. beats the
    0-byte heuristic) and beats minimax-2 (0.9033);
  * registered in the registry alongside net1/net2 (all three coexist);
  * source-level anti-cheat check: net0.py never imports/calls the solver or
    any exact-solve/oracle machinery at inference.
"""
import ast
import io
import json
import os
import tokenize

import pytest

from tests._oracle_scan import assert_no_oracle_access

from app.engine.board import Board
from app.agents.net0 import Net0Agent, DEFAULT_ARTIFACT
from app.agents.net1 import tactical_move
from app.agents import registry
from app.neurogolf import strength
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")
_NET0_SRC = os.path.join(_ROOT, "app", "agents", "net0.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_ARTIFACT),
    reason="neurofour-net0 artifact not trained yet",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- tactical guard invariants (0-param board logic, same contract as net1) #

def test_agent_never_misses_win_or_block_on_sealed():
    ag = Net0Agent()
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


# ---- agent behaviour + manifest ------------------------------------------- #

def test_agent_returns_legal_and_deterministic():
    ag = Net0Agent()
    for p in _load(_SEALED)[:50]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


def test_manifest_is_honest_and_within_nano_cap():
    ag = Net0Agent()
    man = ag.manifest()
    assert man.kind == "search"
    assert man.size_bytes == os.path.getsize(DEFAULT_ARTIFACT)
    assert man.size_bytes <= 4_096, (
        f"neurofour-net0 must fit the NANO tier: size={man.size_bytes}B > 4096B")
    assert man.flops_per_move <= FLOP_CAP
    assert man.params > 0


def test_registered_in_registry():
    names = registry.agent_names()
    assert "neurofour-net0" in names
    # net1/net2 stay registered, unchanged
    assert "neurofour-net1" in names
    assert "neurofour-net2" in names


def test_beats_heuristic_and_minimax2_on_sealed():
    ag = Net0Agent()
    card = strength.score(ag, _load(_SEALED))
    assert card.optimality > 0.900, f"optimality {card.optimality} must beat heuristic 0.900"
    assert card.optimality > 0.9033, f"optimality {card.optimality} must beat minimax-2 0.9033"


def test_qualifies_micro_budget_and_latency():
    """net0 must clear the micro-tier gate (and by construction the nano cap):
    size<=32768 (in fact <=4096), flops<FLOP_CAP, p50 latency<LATENCY_CAP_MS --
    measured, not asserted from a table."""
    import time
    ag = Net0Agent()
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


# ---- anti-cheat: no solver / oracle access at inference -------------------- #

def test_source_has_no_solver_or_oracle_access():
    """Transitive scan (see tests/_oracle_scan.py, gen-9 T1): the OLD
    single-file scan (docstrings/comments excluded, imports of THIS file
    only) could be routed around by importing the oracle through a helper
    module -- net0.py imports app.agents.encode, which legitimately
    imports ONLY the two allowlisted pure board-math helpers from
    app.solver.solver; the new scan walks the FULL app.** import graph
    reachable from net0.py, and additionally forbids private Solver
    methods, dynamic solver imports, and label-file reads anywhere in
    that closure. See tests/test_oracle_scan_selftest.py for the proof
    this scan can actually fail on a laundered import."""
    assert_no_oracle_access('app.agents.net0')
