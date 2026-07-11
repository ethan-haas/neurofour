"""Tests for the NANO-tier deep-search agent `neurofour-net0d`.

Covers:
  * the 0-param tactical guard invariants (identical contract to net0/net1/net2);
  * the agent loads, always returns a legal move, is deterministic;
  * manifest honesty: size == real artifact bytes, size fits the NANO cap
    (<=4096 bytes, net0's own artifact), flops within the caps and matching
    the WIDTH + WIDTH**depth structural bound;
  * it beats net0's sealed optimality (the whole point of net0d: net2's
    depth-3 refutation search applied to net0's nano leaf) and clears the
    qualifies_micro budget (size<=32768, flops<5e6, latency<50ms);
  * registered in the registry alongside net0/net1/net2 (all coexist,
    net0/net1/net2 unaffected);
  * source-level anti-cheat check: net0d.py never imports/calls the solver or
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
from app.agents.net0 import Net0Agent, DEFAULT_ARTIFACT as NET0_ARTIFACT
from app.agents.net0d import Net0dAgent
from app.agents.net2 import DEPTH
from app.agents import registry
from app.neurogolf import strength
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")
_NET0D_SRC = os.path.join(_ROOT, "app", "agents", "net0d.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(NET0_ARTIFACT),
    reason="neurofour-net0d's leaf artifact (net0's) not trained yet",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- tactical guard invariants (0-param board logic, same contract as net1) #

def test_agent_never_misses_win_or_block_on_sealed():
    ag = Net0dAgent()
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
    ag = Net0dAgent()
    for p in _load(_SEALED)[:50]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


def test_manifest_is_honest_and_within_nano_cap():
    ag = Net0dAgent()
    man = ag.manifest()
    assert man.kind == "search"
    assert man.name == "neurofour-net0d"
    assert man.size_bytes == os.path.getsize(ag.artifact_path)
    assert man.size_bytes == os.path.getsize(NET0_ARTIFACT), (
        "net0d must reuse net0's exact on-disk artifact")
    assert man.size_bytes <= 4_096, (
        f"neurofour-net0d must fit the NANO tier: size={man.size_bytes}B > 4096B")
    assert man.flops_per_move <= FLOP_CAP
    assert man.params > 0


def test_manifest_flops_match_structural_bound():
    """flops_per_move must equal the declared WIDTH + WIDTH**depth worst-case
    leaf-call bound times the per-forward-pass cost, plus guard bit-ops --
    same formula as net2 (net0d reuses net2's manifest() verbatim via
    subclassing), just with net0's smaller param count."""
    from app.engine.board import WIDTH
    from app.agents.encode import FEATURE_DIM
    ag = Net0dAgent()
    man = ag.manifest()
    max_leaf_calls = WIDTH + WIDTH ** ag.depth
    expected = max_leaf_calls * (2 * man.params + FEATURE_DIM) + 4 * WIDTH
    assert man.flops_per_move == expected
    assert man.flops_per_move < 5_000_000


def test_registered_in_registry():
    names = registry.agent_names()
    assert "neurofour-net0d" in names
    # net0/net1/net2 stay registered, unchanged
    assert "neurofour-net0" in names
    assert "neurofour-net1" in names
    assert "neurofour-net2" in names


def test_net0d_beats_net0_on_sealed():
    """The whole point of net0d: net2's depth-3 refutation search applied to
    net0's nano leaf must add REAL strength over net0's 1-ply-only search."""
    rows = _load(_SEALED)
    card0 = strength.score(Net0Agent(), rows)
    card0d = strength.score(Net0dAgent(), rows)
    assert card0d.optimality > card0.optimality, (
        f"net0d {card0d.optimality} must beat net0 {card0.optimality}")
    assert card0d.optimality > 0.936667
    assert card0d.blunder_rate == 0.0


def test_qualifies_micro_budget_and_latency():
    """net0d must clear the micro-tier gate (and by construction the nano
    cap): size<=32768 (in fact <=4096), flops<FLOP_CAP, p50 latency<
    LATENCY_CAP_MS -- measured, not asserted from a table."""
    import time
    ag = Net0dAgent()
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


def test_depth_matches_net2():
    """net0d reuses net2's exact depth (subclassing, no depth drift)."""
    ag = Net0dAgent()
    assert ag.depth == DEPTH == 3


# ---- anti-cheat: no solver / oracle access at inference -------------------- #

def test_source_has_no_solver_or_oracle_access():
    """Transitive scan (see tests/_oracle_scan.py, gen-9 T1): the OLD
    single-file scan (docstrings/comments excluded, imports of THIS file
    only) could be routed around by importing the oracle through a helper
    module -- net0d.py imports app.agents.encode, which legitimately
    imports ONLY the two allowlisted pure board-math helpers from
    app.solver.solver; the new scan walks the FULL app.** import graph
    reachable from net0d.py, and additionally forbids private Solver
    methods, dynamic solver imports, and label-file reads anywhere in
    that closure. See tests/test_oracle_scan_selftest.py for the proof
    this scan can actually fail on a laundered import."""
    assert_no_oracle_access('app.agents.net0d')
