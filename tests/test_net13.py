"""Tests for `neurofour-net13` (gen-7 PIVOT: two-currency leaf-eval N /
search-node M hard-budget alpha-beta, TT+PVS+killer/history, over net1's
frozen leaf). Registered: at N=64,M=14617,max_depth=14 net13 wins the
pooled-6000 corpus (0.9645 vs net4's 0.9428 / net11(N=520)'s 0.9570,
McNemar-significant vs both) and TIES net4 exactly on sealed(300) (287/300
each -- 0 lost positions, well within the 1-position registration
tolerance), and beats net4 on the fresh seed99 anti-memorization check
(288/300 vs 279/300). See net13.py's module docstring for the full
decision-rule trace and frontier."""
import ast
import io
import json
import os
import tokenize

import pytest

from tests._oracle_scan import assert_no_oracle_access

from app.engine.board import Board, WIDTH
from app.agents.net1 import DEFAULT_ARTIFACT
from app.agents.net13 import Net13Agent, OPS_PER_NODE
from app.neurogolf import strength
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")
_NET13_SRC = os.path.join(_ROOT, "app", "agents", "net13.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_ARTIFACT),
    reason="neurofour-net13's leaf artifact (net1's) not trained yet",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- tactical guard invariants --------------------------------------------- #

def test_agent_never_misses_win_or_block_on_sealed_sample():
    ag = Net13Agent()
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
    ag = Net13Agent()
    for p in _load(_SEALED)[:40]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


# ---- the two hard structural budgets ---------------------------------------- #

def test_leaf_eval_budget_is_structural():
    """`self._evals` must never exceed `n_budget`, for every board -- the
    same invariant net11.py asserts, now for net13's check-before-spend
    design (no exception needed: `_value` is only ever called from
    `_leaf_value_nonterminal` after `self._evals < self.n_budget`)."""
    for N, M in [(0, 500), (16, 500), (64, 2000)]:
        ag = Net13Agent(n_budget=N, m_budget=M, max_depth=8)
        for p in _load(_SEALED)[:40]:
            b = Board.from_moves(p["board"])
            ag.select_move(b)
            assert ag._evals <= N, f"N={N}: actual_evals={ag._evals} > N"


def test_search_node_budget_is_structural():
    """`self._nodes` must never exceed `m_budget`, for every board -- every
    call site into `_negamax` (root loop, interior children loop, PVS
    re-search) checks `self._nodes < self.m_budget` BEFORE calling; see the
    module docstring's "Two-currency hard-stop contract"."""
    for N, M in [(0, 100), (16, 500), (64, 3000)]:
        ag = Net13Agent(n_budget=N, m_budget=M, max_depth=10)
        for p in _load(_SEALED)[:40]:
            b = Board.from_moves(p["board"])
            ag.select_move(b)
            assert ag._nodes <= M, f"M={M}: actual_nodes={ag._nodes} > M"


def test_zero_node_budget_still_returns_legal_move():
    """M=0 is a legitimate degenerate config: `select_move`'s own root loop
    checks `self._nodes >= self.m_budget` before the FIRST candidate, so no
    root candidate is ever searched via `_negamax` -- the tactical guard (or
    the `legal[0]` fallback) must still produce a legal move."""
    ag = Net13Agent(n_budget=0, m_budget=0, max_depth=5)
    for p in _load(_SEALED)[:20]:
        b = Board.from_moves(p["board"])
        m = ag.select_move(b)
        assert m in b.legal_moves()


# ---- manifest honesty -------------------------------------------------------- #

def test_manifest_is_honest_and_within_caps():
    ag = Net13Agent()
    man = ag.manifest()
    assert man.kind == "search"
    assert man.size_bytes == os.path.getsize(ag.artifact_path)
    assert man.size_bytes <= 32_768           # micro tier
    assert man.flops_per_move <= FLOP_CAP
    assert man.params > 0


def test_manifest_flops_match_two_currency_formula():
    """flops_per_move must equal N*(2*params+FEATURE_DIM) + M*OPS_PER_NODE +
    guard_bitops EXACTLY -- the closed-form two-currency formula, never a
    separately hand-tuned number."""
    from app.agents.encode import FEATURE_DIM
    for N, M in [(0, 100), (64, 14617), (520, 19)]:
        ag = Net13Agent(n_budget=N, m_budget=M)
        man = ag.manifest()
        expected = N * (2 * man.params + FEATURE_DIM) + M * OPS_PER_NODE + 4 * WIDTH
        assert man.flops_per_move == expected
        assert ag._max_leaf_calls() == N
        assert ag._max_node_calls() == M


def test_qualifies_micro_budget():
    import time
    ag = Net13Agent()
    man = ag.manifest()
    assert man.size_bytes <= 32_768
    assert man.flops_per_move < FLOP_CAP
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


# ---- anti-cheat: no solver / oracle access at inference --------------------- #

def test_source_has_no_solver_or_oracle_access():
    """Transitive scan (see tests/_oracle_scan.py, gen-9 T1): the OLD
    single-file scan (docstrings/comments excluded, imports of THIS file
    only) could be routed around by importing the oracle through a helper
    module -- net13.py imports app.agents.encode, which legitimately
    imports ONLY the two allowlisted pure board-math helpers from
    app.solver.solver; the new scan walks the FULL app.** import graph
    reachable from net13.py, and additionally forbids private Solver
    methods, dynamic solver imports, and label-file reads anywhere in
    that closure. See tests/test_oracle_scan_selftest.py for the proof
    this scan can actually fail on a laundered import."""
    assert_no_oracle_access('app.agents.net13')


# ---- registration + the decision-rule claim --------------------------------- #

def test_registered():
    from app.agents import registry
    assert "neurofour-net13" in registry.agent_names()
    ag = registry.make_agent("neurofour-net13")
    assert isinstance(ag, Net13Agent)
    # gen-8 T3 re-gate: honest OPS_PER_NODE=350 (was dishonestly-priced 300)
    # caps M at 12529 for N=64 (was 14617) -- see net13.py's module docstring
    # "gen-8 T1/T2/T3" sections and tests/test_net13_flop_honesty.py.
    assert ag.n_budget == 64 and ag.m_budget == 12529


def test_net13_within_one_position_of_net4_on_sealed_at_default_config():
    """gen-8 T3 re-gate: the honestly-priced default (N=64, M=12529, down
    from gen-7's dishonest M=14617) no longer ties net4's sealed(300)
    optimality EXACTLY -- it loses by exactly 1 position (286/300 vs
    287/300), which is AT (not beyond) the 1-position registration
    tolerance clause (b) uses. See net13.py's module docstring "gen-8 T3
    RE-GATE" section for the full re-swept frontier and decision-rule
    trace."""
    from app.agents.net4 import Net4Agent
    rows = _load(_SEALED)
    ag4 = Net4Agent()
    ag13 = Net13Agent()
    c4 = strength.score(ag4, rows)
    c13 = strength.score(ag13, rows)
    gap = round(c4.optimality * len(rows)) - round(c13.optimality * len(rows))
    assert gap <= 1, f"sealed_gap_positions={gap} exceeds the 1-position tolerance"
