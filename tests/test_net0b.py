"""Tests for `neurofour-net0b`: net4's top-K=2 beam refutation search (D=3)
applied to net0's NANO leaf. Same on-disk artifact/size as net0d, strictly
fewer flops -- see net0b.py's module docstring for the full honest
sealed-vs-seed99 breakdown (the small sealed-set edge over net0d is noise;
the flops saving is the real, reproducible win).
"""
import ast
import io
import json
import os
import tokenize

import pytest

from tests._oracle_scan import assert_no_oracle_access

from app.engine.board import Board, WIDTH
from app.agents.net0 import Net0Agent, DEFAULT_ARTIFACT as NET0_ARTIFACT
from app.agents.net0d import Net0dAgent
from app.agents.net0b import Net0bAgent
from app.agents.net4 import DEPTH, K
from app.agents import registry
from app.neurogolf import strength
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")
_NET0B_SRC = os.path.join(_ROOT, "app", "agents", "net0b.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(NET0_ARTIFACT),
    reason="neurofour-net0b's leaf artifact (net0's) not trained yet",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- tactical guard invariants --------------------------------------------- #

def test_agent_never_misses_win_or_block_on_sealed():
    ag = Net0bAgent()
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


# ---- agent behaviour + manifest -------------------------------------------- #

def test_agent_returns_legal_and_deterministic():
    ag = Net0bAgent()
    for p in _load(_SEALED)[:50]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


def test_manifest_is_honest_and_within_nano_cap():
    ag = Net0bAgent()
    man = ag.manifest()
    assert man.kind == "search"
    assert man.name == "neurofour-net0b"
    assert man.size_bytes == os.path.getsize(ag.artifact_path)
    assert man.size_bytes == os.path.getsize(NET0_ARTIFACT), (
        "net0b must reuse net0's exact on-disk artifact")
    assert man.size_bytes <= 4_096, (
        f"neurofour-net0b must fit the NANO tier: size={man.size_bytes}B > 4096B")
    assert man.flops_per_move <= FLOP_CAP
    assert man.params > 0


def test_depth_and_k_match_net4_default():
    ag = Net0bAgent()
    assert ag.depth == DEPTH == 3
    assert ag.k == K == 2


def test_registered_in_registry():
    names = registry.agent_names()
    assert "neurofour-net0b" in names
    # net0/net0d/net4 all stay registered, unchanged
    assert "neurofour-net0" in names
    assert "neurofour-net0d" in names
    assert "neurofour-net4" in names


def test_net0b_same_artifact_and_size_as_net0d():
    """net0b and net0d share net0's exact on-disk artifact -- byte-identical
    size, only the search structure (top-K beam vs full-width) differs."""
    b = Net0bAgent()
    d = Net0dAgent()
    assert b.manifest().size_bytes == d.manifest().size_bytes
    assert b.artifact_path == d.artifact_path == NET0_ARTIFACT


def test_net0b_dominates_net0d_on_flops_per_metric_sec7():
    """METRIC.md sec.7: A dominates B iff optimality>=, size<=, flops<=, with
    >=1 strict. net0b's optimality>=net0d's (sealed AND seed99, see module
    docstring), size is exactly equal, flops is strictly lower -- a real
    dominance regardless of whether the sealed-set optimality delta is
    signal or noise (it is documented as noise; the flops win is real)."""
    rows = _load(_SEALED)
    card_d = strength.score(Net0dAgent(), rows)
    card_b = strength.score(Net0bAgent(), rows)
    man_d = Net0dAgent().manifest()
    man_b = Net0bAgent().manifest()
    assert card_b.optimality >= card_d.optimality
    assert man_b.size_bytes == man_d.size_bytes
    assert man_b.flops_per_move < man_d.flops_per_move


def test_qualifies_micro_budget_and_latency():
    import time
    ag = Net0bAgent()
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
    module -- net0b.py imports app.agents.encode, which legitimately
    imports ONLY the two allowlisted pure board-math helpers from
    app.solver.solver; the new scan walks the FULL app.** import graph
    reachable from net0b.py, and additionally forbids private Solver
    methods, dynamic solver imports, and label-file reads anywhere in
    that closure. See tests/test_oracle_scan_selftest.py for the proof
    this scan can actually fail on a laundered import."""
    assert_no_oracle_access('app.agents.net0b')
