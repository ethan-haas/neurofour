"""Tests for `neurofour-net14` (gen-8 T4: zero-byte, zero-param pure-search
agent -- net13's search machinery, no artifact loaded at all, static
0-param heuristic leaf). See net14.py's module docstring for the full
design rationale and op-cost derivation."""
import ast
import io
import json
import os
import tokenize

import pytest

from tests._oracle_scan import assert_no_oracle_access

from app.engine.board import Board, WIDTH
from app.agents.net14 import Net14Agent, OPS_PER_NODE
from app.neurogolf import strength
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")
_NET14_SRC = os.path.join(_ROOT, "app", "agents", "net14.py")


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- tactical guard invariants --------------------------------------------- #

def test_agent_never_misses_win_or_block_on_sealed_sample():
    ag = Net14Agent()
    for p in _load(_SEALED)[:60]:
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


# ---- determinism + legality ------------------------------------------------- #

def test_agent_returns_legal_and_deterministic():
    ag = Net14Agent()
    for p in _load(_SEALED)[:40]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


# ---- the single hard structural budget -------------------------------------- #

def test_search_node_budget_is_structural():
    for M in [0, 50, 500, 3000]:
        ag = Net14Agent(m_budget=M, max_depth=10)
        for p in _load(_SEALED)[:40]:
            b = Board.from_moves(p["board"])
            ag.select_move(b)
            assert ag._nodes <= M, f"M={M}: actual_nodes={ag._nodes} > M"


def test_zero_node_budget_still_returns_legal_move():
    ag = Net14Agent(m_budget=0, max_depth=5)
    for p in _load(_SEALED)[:20]:
        b = Board.from_moves(p["board"])
        m = ag.select_move(b)
        assert m in b.legal_moves()


# ---- manifest honesty: params=0, size_bytes=0 per METRIC.md sec.3 ---------- #

def test_manifest_is_honest_zero_artifact_and_within_caps():
    ag = Net14Agent()
    man = ag.manifest()
    assert man.kind == "search"
    assert man.params == 0
    assert man.size_bytes == 0
    assert man.artifact_path is None
    assert man.flops_per_move <= FLOP_CAP


def test_manifest_flops_match_single_currency_formula():
    for M in [0, 100, 1428, 5000]:
        ag = Net14Agent(m_budget=M)
        man = ag.manifest()
        expected = M * OPS_PER_NODE + 4 * WIDTH
        assert man.flops_per_move == expected
        assert ag._max_node_calls() == M


def test_qualifies_nano_budget():
    """size_bytes=0 qualifies for the `nano` tier (0 <= 4096, METRIC.md
    sec.5) -- unlike net13, which is stuck in `micro` by its reused
    artifact."""
    import time
    from app.neurogolf.config import tier_for
    ag = Net14Agent()
    man = ag.manifest()
    assert tier_for(man.size_bytes) == "nano"
    rows = _load(_SEALED)[:60]
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


# ---- anti-cheat: no solver / oracle access, no artifact load ---------------- #

def test_source_has_no_solver_or_oracle_access():
    """Transitive scan (see tests/_oracle_scan.py, gen-9 T1): the OLD
    single-file scan (docstrings/comments excluded, imports of THIS file
    only) could be routed around by importing the oracle through a helper
    module -- net14.py imports app.agents.encode, which legitimately
    imports ONLY the two allowlisted pure board-math helpers from
    app.solver.solver; the new scan walks the FULL app.** import graph
    reachable from net14.py, and additionally forbids private Solver
    methods, dynamic solver imports, and label-file reads anywhere in
    that closure. See tests/test_oracle_scan_selftest.py for the proof
    this scan can actually fail on a laundered import."""
    assert_no_oracle_access('app.agents.net14', extra_banned_tokens=('.npz', 'load_npz'))


def test_never_loads_any_artifact():
    """net14 must never touch the filesystem for a learned artifact -- trip
    a guard on `open()` for any .npz-looking path during construction AND
    select_move."""
    import builtins
    _orig_open = builtins.open
    violations = []

    def guard(file, *a, **k):
        fn = str(file).replace("\\", "/").lower()
        if fn.endswith(".npz"):
            violations.append(file)
            raise AssertionError(f"net14 opened an artifact file: {file}")
        return _orig_open(file, *a, **k)

    builtins.open = guard
    try:
        ag = Net14Agent()
        for p in _load(_SEALED)[:20]:
            b = Board.from_moves(p["board"])
            ag.select_move(b)
    finally:
        builtins.open = _orig_open
    assert violations == []


# ---- registration decision: gen-10 T-lever REGISTERED, now HEADLINE -------- #

def test_registered_per_gen10_bitboard_leaf_headline_result():
    """gen-9 T3 registered net14 (M=1428, grid-scan leaf) as a
    Pareto-non-dominated, sub-HEADLINE agent (opt=0.95 < net4's 0.956667).
    gen-10's T-lever (bitboard leaf, see `heuristic_eval_bb.py` +
    net14.py's module docstring "gen-10 T-lever" addendum) cut
    OPS_PER_NODE 3500->1000 (machine-measured max observed ops/node=648,
    see tests/test_net14_flop_honesty.py), raising the honest M_max
    1428->4999. At M=4999 net14's sealed(300) optimality rose 0.95->0.96
    (288/300), which now EXCEEDS net4's HEADLINE (287/300, 0.956667) --
    confirmed three independent ways (not selected on sealed alone, per
    METRIC.md sec.8's "never select on sealed" discipline): (1) committed
    sealed(300) 288 vs net4's 287; (2) pooled(6000) dev_big+dev_big2+
    dev_big3 0.9680 vs net4's 0.9428 (McNemar chi2_cc=66.372, decisively
    significant -- the real, well-powered signal, since the committed-
    sealed 1-position edge alone is at the set's own resolution floor,
    1/300=0.0033, and should not be oversold on its own); (3) a fresh
    seed=99 sealed(300) draw, 0.9567 vs net4's 0.93 (net14 wins by 8
    positions there too). All three independently agree net14 now beats
    net4; see the coder's gen-10 report for the full (M, pooled_opt) sweep
    curve and the old-vs-new-leaf paired discordance."""
    from app.agents import registry
    from app.agents.net14 import Net14Agent
    from app.neurogolf import strength, cost
    from app.neurogolf.positions import load_set

    assert "neurofour-net14" in registry.agent_names()
    ag = registry.make_agent("neurofour-net14")
    assert isinstance(ag, Net14Agent)

    positions = load_set(_SEALED)
    sc = strength.score(Net14Agent(), positions)
    cc = cost.measure(Net14Agent(), positions)
    assert sc.optimality == 0.96
    assert cc.size_bytes == 0
    assert cc.flops_per_move < FLOP_CAP
    assert not cc.over_budget

    # net14's sealed optimality (0.96) now EXCEEDS net4's (0.9566...) --
    # a zero-byte pure-search agent becomes the new HEADLINE candidate.
    net4_sc = strength.score(registry.make_agent("neurofour-net4"), positions)
    assert sc.optimality > net4_sc.optimality

    # Pareto non-domination: no OTHER registered agent both (opt>=,
    # size<=, flops<=) net14 with at least one strict inequality.
    # `size_bytes`/`flops_per_move` come straight from `agent.manifest()`
    # -- structural constants, NOT sample-dependent -- so they are checked
    # first, exactly, over ALL registered agents at zero cost (no
    # `select_move` calls at all, so this does not pay `perfect`'s slow
    # live-solve tax). Only agents that pass BOTH structural conditions
    # (i.e. could possibly dominate) need an optimality comparison at all,
    # and that is run over the FULL 300-position sealed set (not a
    # subsample -- a small subsample can spuriously tie/flip close
    # optimality values, as `minimax-4`'s 0.9367 vs net14's 0.9500 on the
    # full set demonstrated on a 60-position draw during development of
    # this test). `perfect` structurally fails on flops alone
    # (50,000,000 > net14's 4,998,028) and is never select_move'd here.
    def _dominates(a, b):
        ge = (a["opt"] >= b["opt"] and a["size"] <= b["size"]
              and a["flops"] <= b["flops"])
        if not ge:
            return False
        return (a["opt"] > b["opt"] or a["size"] < b["size"]
                or a["flops"] < b["flops"])

    net14_man = ag.manifest()
    net14_card = {"opt": sc.optimality, "size": net14_man.size_bytes,
                  "flops": net14_man.flops_per_move}
    possible_dominators = []
    for name in registry.agent_names():
        if name == "neurofour-net14":
            continue
        other_ag = registry.make_agent(name)
        man = other_ag.manifest()
        if man.size_bytes <= net14_card["size"] and man.flops_per_move <= net14_card["flops"]:
            possible_dominators.append((name, other_ag, man))

    for name, other_ag, man in possible_dominators:
        other_opt = strength.score(other_ag, positions).optimality
        other_card = {"opt": other_opt, "size": man.size_bytes,
                      "flops": man.flops_per_move}
        assert not _dominates(other_card, net14_card), (
            f"{name} dominates net14 on the FULL sealed(300) set: "
            f"{other_card} vs {net14_card}")
