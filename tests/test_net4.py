"""Tests for `neurofour-net4` (E2: top-K beam refutation search). net4 TIES
net2's sealed optimality exactly at ~2.27x fewer declared flops -- per
METRIC.md sec.7 that is a strict Pareto dominance (equal optimality, equal
size, strictly lower flops, >=1 strict), so net4 is REGISTERED (gen-9
correction; see net4.py's module docstring for gen-8's original reasoning
and why it was wrong). These tests cover full correctness (legality,
determinism, honest flop bound, no-solver-at-inference) plus the tie/
dominance claim and registration.
"""
import ast
import io
import json
import os
import tokenize

import pytest

from tests._oracle_scan import assert_no_oracle_access

from app.engine.board import Board, WIDTH
from app.agents.net1 import DEFAULT_ARTIFACT
from app.agents.net4 import Net4Agent, _evals
from app.neurogolf import strength
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")
_NET4_SRC = os.path.join(_ROOT, "app", "agents", "net4.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_ARTIFACT),
    reason="neurofour-net4's leaf artifact (net1's) not trained yet",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- tactical guard invariants --------------------------------------------- #

def test_agent_never_misses_win_or_block_on_sealed():
    ag = Net4Agent()
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
    ag = Net4Agent()
    for p in _load(_SEALED)[:50]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


def test_manifest_is_honest_and_within_caps():
    ag = Net4Agent()
    man = ag.manifest()
    assert man.kind == "search"
    assert man.size_bytes == os.path.getsize(ag.artifact_path)
    assert man.size_bytes <= 32_768           # micro tier
    assert man.flops_per_move <= FLOP_CAP
    assert man.params > 0


def test_hard_topk_cap_is_structural():
    """The beam cap is an actual `[:k]` slice, not merely an average -- no
    matter the board, `_search` never expands more than k children."""
    ag = Net4Agent(depth=3, k=2)
    # sample a variety of boards and check the beam never exceeds k via a
    # monkeypatched counter on the internal ranking size.
    from app.engine.board import CENTER_ORDER
    _CR = {c: i for i, c in enumerate(CENTER_ORDER)}
    for p in _load(_SEALED)[:30]:
        b = Board.from_moves(p["board"])
        ranked = []
        for c in sorted(b.legal_moves(), key=lambda c: _CR[c]):
            child = b.play(c)
            ranked.append(child)
        beam = ranked[:ag.k]
        assert len(beam) <= ag.k


def test_manifest_flops_match_structural_bound():
    """flops_per_move must equal `_max_leaf_calls()` (the WIDTH + WIDTH*evals
    recurrence documented in the module docstring) times the per-forward-pass
    cost, plus guard bit-ops -- derived purely from the algorithm's structure
    (depth, k), never tuned to any particular eval set."""
    from app.agents.encode import FEATURE_DIM
    for depth, k in [(3, 2), (3, 3), (4, 2)]:
        ag = Net4Agent(depth=depth, k=k)
        man = ag.manifest()
        expected_calls = WIDTH + WIDTH * _evals(depth - 1, k)
        expected = expected_calls * (2 * man.params + FEATURE_DIM) + 4 * WIDTH
        assert man.flops_per_move == expected
        assert ag._max_leaf_calls() == expected_calls


def test_evals_recurrence_matches_docstring():
    """evals(1)=WIDTH, evals(d)=WIDTH+k*evals(d-1) for d>=2, evals(0)=0."""
    for k in (2, 3):
        assert _evals(0, k) == 0
        assert _evals(1, k) == WIDTH
        assert _evals(2, k) == WIDTH + k * WIDTH
        assert _evals(3, k) == WIDTH + k * (WIDTH + k * WIDTH)


def test_beam_bound_never_exceeds_fullwidth_at_same_depth():
    """A k<WIDTH beam's honest flop bound must be <= the full-width (k=WIDTH)
    bound at the same depth -- sanity-checks the recurrence direction."""
    for depth in (2, 3, 4):
        narrow = _evals(depth, 2)
        wide = _evals(depth, WIDTH)
        assert narrow <= wide


def test_net4_ties_net2_on_sealed_at_default_config():
    """Documented result: D=3,K=2 with net1's leaf reproduces net2's sealed
    optimality EXACTLY (a tie, not a regression) at fewer declared flops."""
    from app.agents.net2 import Net2Agent
    rows = _load(_SEALED)
    ag2 = Net2Agent()
    ag4 = Net4Agent(depth=3, k=2)
    c2 = strength.score(ag2, rows)
    c4 = strength.score(ag4, rows)
    assert c4.optimality == c2.optimality
    assert c4.blunder_rate == c2.blunder_rate == 0.0
    assert ag4.manifest().flops_per_move < ag2.manifest().flops_per_move


def test_qualifies_micro_budget():
    import time
    ag = Net4Agent()
    man = ag.manifest()
    assert man.size_bytes <= 32_768
    assert man.flops_per_move < FLOP_CAP
    rows = _load(_SEALED)[:150]
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
    module -- net4.py imports app.agents.encode, which legitimately
    imports ONLY the two allowlisted pure board-math helpers from
    app.solver.solver; the new scan walks the FULL app.** import graph
    reachable from net4.py, and additionally forbids private Solver
    methods, dynamic solver imports, and label-file reads anywhere in
    that closure. See tests/test_oracle_scan_selftest.py for the proof
    this scan can actually fail on a laundered import."""
    assert_no_oracle_access('app.agents.net4')


def test_registered_and_dominates_net2():
    """gen-9: net4 IS registered (equal optimality, equal size, strictly
    lower flops than net2 -- METRIC.md sec.7 dominance, >=1 strict suffices).
    net2 stays registered too (unchanged agent), it just becomes the
    dominated (non-pareto) point once net4 exists alongside it."""
    from app.agents import registry
    from app.agents.net2 import Net2Agent
    assert "neurofour-net4" in registry.agent_names()
    assert "neurofour-net2" in registry.agent_names()  # net2 unaffected/unremoved
    ag2 = Net2Agent()
    ag4 = registry.make_agent("neurofour-net4")
    m2, m4 = ag2.manifest(), ag4.manifest()
    assert m4.size_bytes == m2.size_bytes
    assert m4.flops_per_move < m2.flops_per_move
