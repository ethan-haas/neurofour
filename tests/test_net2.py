"""Tests for the deeper learned search agent `neurofour-net2`.

Covers:
  * the 0-param tactical guard invariants (identical contract to net1);
  * the agent loads, always returns a legal move, is deterministic;
  * manifest honesty: size == real artifact bytes, flops within the caps,
    and the declared flops match the WIDTH + WIDTH**depth structural bound
    (never an under-count, regardless of which positions are scored);
  * it beats net1's sealed optimality (the whole point of net2) and clears
    the qualifies_micro budget (size<=32768, flops<5e6, latency<50ms);
  * source-level anti-cheat check: net2.py never imports/calls the solver or
    any exact-solve/oracle machinery at inference.
"""
import ast
import json
import os

import pytest

from tests._oracle_scan import assert_no_oracle_access

from app.engine.board import Board
from app.agents.net1 import tactical_move, DEFAULT_ARTIFACT
from app.agents.net2 import Net2Agent, DEPTH
from app.agents import registry
from app.neurogolf import strength
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")
_NET2_SRC = os.path.join(_ROOT, "app", "agents", "net2.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_ARTIFACT),
    reason="neurofour-net2's leaf artifact (net1's) not trained yet",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- tactical guard invariants (0-param board logic, same contract as net1) #

def test_agent_never_misses_win_or_block_on_sealed():
    ag = Net2Agent()
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
    ag = Net2Agent()
    for p in _load(_SEALED)[:50]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


def test_manifest_is_honest_and_within_caps():
    ag = Net2Agent()
    man = ag.manifest()
    assert man.kind == "search"
    assert man.size_bytes == os.path.getsize(ag.artifact_path)
    assert man.size_bytes <= 32_768           # micro tier
    assert man.flops_per_move <= FLOP_CAP
    assert man.params > 0


def test_manifest_flops_match_structural_bound():
    """flops_per_move must equal the declared WIDTH + WIDTH**depth worst-case
    leaf-call bound times the per-forward-pass cost, plus guard bit-ops --
    i.e. a value derived purely from the algorithm's structure, not tuned to
    any particular eval set."""
    from app.engine.board import WIDTH
    from app.agents.encode import FEATURE_DIM
    ag = Net2Agent()
    man = ag.manifest()
    max_leaf_calls = WIDTH + WIDTH ** ag.depth
    expected = max_leaf_calls * (2 * man.params + FEATURE_DIM) + 4 * WIDTH
    assert man.flops_per_move == expected


def test_registered_in_registry():
    names = registry.agent_names()
    assert "neurofour-net2" in names
    assert "neurofour-net1" in names   # net1 stays registered, unchanged


def test_net2_beats_net1_on_sealed():
    ag1_names = registry.agent_names()
    assert "neurofour-net1" in ag1_names
    from app.agents.net1 import Net1Agent
    rows = _load(_SEALED)
    card1 = strength.score(Net1Agent(), rows)
    card2 = strength.score(Net2Agent(), rows)
    assert card2.optimality > card1.optimality, (
        f"net2 {card2.optimality} must beat net1 {card1.optimality}")
    assert card2.optimality > 0.946667
    assert card2.blunder_rate == 0.0


def test_qualifies_micro_budget():
    """net2 must clear the micro-tier gate: size<=32768, flops<FLOP_CAP,
    p50 latency<LATENCY_CAP_MS -- measured, not asserted from a table."""
    import time
    ag = Net2Agent()
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
    module -- net2.py imports app.agents.encode, which legitimately
    imports ONLY the two allowlisted pure board-math helpers from
    app.solver.solver; the new scan walks the FULL app.** import graph
    reachable from net2.py, and additionally forbids private Solver
    methods, dynamic solver imports, and label-file reads anywhere in
    that closure. See tests/test_oracle_scan_selftest.py for the proof
    this scan can actually fail on a laundered import."""
    assert_no_oracle_access('app.agents.net2')


def test_depth_within_flop_budget():
    """D=4 would exceed FLOP_CAP (documented reason net2 ships D=3)."""
    from app.engine.board import WIDTH
    from app.agents.encode import FEATURE_DIM
    ag = Net2Agent()
    params = ag.manifest().params
    over_cap_at_4 = (WIDTH + WIDTH ** 4) * (2 * params + FEATURE_DIM) + 4 * WIDTH
    assert over_cap_at_4 > FLOP_CAP
    assert DEPTH in (2, 3)
