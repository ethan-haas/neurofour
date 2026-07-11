"""Tests for `neurofour-net16`: gen-2 "research-model / cost-axis compression"
-- net1's exact leaf net (same 4705 params, same tactical guard + 1-ply
negamax search) re-quantised to per-row int5 (5-bit) with 40% global
magnitude pruning of W1, packed in a hand-rolled bit-container and whole-
file zlib-compressed (`app/agents/mlp.py` `save_compressed`/`load_compressed`,
`scripts/compress_net1.py`). Winner of a 5(bits) x 7(prune) sweep, picked by
measured dev_big(2000) accuracy, not assumed -- see `scripts/eval_net16_sweep.py`.

Committed artifact: 2867 bytes (net1 is 4837B, the NANO leader net0b is
3290B) -- see net16.py's module docstring for the full sweep table and the
`scripts/eval_resolution.py` dev_big + multi-seed results this file's
dominance assertions are drawn from (measured, not re-derived here, since a
fresh dev_big pass in a unit test would be slow; the numbers below are
pinned to the exact artifact committed at `DEFAULT_ARTIFACT`).
"""
import json
import os

import pytest

from tests._oracle_scan import assert_no_oracle_access

from app.engine.board import Board
from app.agents.net1 import Net1Agent, DEFAULT_ARTIFACT as NET1_ARTIFACT
from app.agents.net0b import Net0bAgent
from app.agents.net16 import Net16Agent, DEFAULT_ARTIFACT
from app.agents import registry
from app.neurogolf import strength
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")
_DEV_BIG = os.path.join(_ROOT, "bench_data", "dev_big.jsonl")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_ARTIFACT),
    reason="neurofour-net16 artifact not built yet (run scripts/compress_net1.py)",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- tactical guard invariants (inherited from Net1Agent, unchanged) ------ #

def test_agent_never_misses_win_or_block_on_sealed():
    ag = Net16Agent()
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
    ag = Net16Agent()
    for p in _load(_SEALED)[:50]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


def test_manifest_is_honest_and_within_nano_cap():
    ag = Net16Agent()
    man = ag.manifest()
    assert man.kind == "search"
    assert man.name == "neurofour-net16"
    assert man.size_bytes == os.path.getsize(ag.artifact_path)
    assert man.size_bytes <= 4_096, (
        f"neurofour-net16 must fit the NANO tier: size={man.size_bytes}B > 4096B")
    assert man.size_bytes < os.path.getsize(NET1_ARTIFACT), (
        "the whole point of gen-2: net16 must be strictly smaller than net1's "
        "own artifact it was compressed from")
    assert man.flops_per_move <= FLOP_CAP
    assert man.params == Net1Agent().manifest().params, (
        "dequantisation happens once at __init__; the dense forward pass -- "
        "and therefore param count/flops -- must match net1's exactly (no "
        "flops smuggled out by pruning, since zeros still multiply through)")
    assert man.flops_per_move == Net1Agent().manifest().flops_per_move


def test_registered_in_registry():
    names = registry.agent_names()
    assert "neurofour-net16" in names
    assert "neurofour-net1" in names   # unchanged
    ag = registry.make_agent("neurofour-net16")
    assert isinstance(ag, Net16Agent)


def test_inherits_net1_search_structure():
    """net16 subclasses Net1Agent and overrides only artifact loading /
    manifest -- select_move (tactical guard + 1-ply negamax) is verbatim."""
    assert Net16Agent.select_move is Net1Agent.select_move
    assert Net16Agent._value is Net1Agent._value


def test_not_significantly_worse_than_net1_on_dev_big():
    """The compression's whole point: accuracy must not be significantly
    worse than net1's own (McNemar on the SAME positions, dev_big(2000) --
    see scripts/eval_resolution.py for the full paired test; re-derived here
    at unit-test scope on the committed artifact so a future artifact swap
    that regresses accuracy is caught)."""
    rows = _load(_DEV_BIG)
    net1 = Net1Agent()
    net16 = Net16Agent()
    hits1 = 0
    hits16 = 0
    b_only = 0   # net1 hit, net16 miss
    c_only = 0   # net1 miss, net16 hit
    for p in rows:
        board = Board.from_moves(p["board"])
        m1 = net1.select_move(board)
        m16 = net16.select_move(board)
        h1 = 1 if m1 in p["optimal_cols"] else 0
        h16 = 1 if m16 in p["optimal_cols"] else 0
        hits1 += h1
        hits16 += h16
        if h1 == 1 and h16 == 0:
            b_only += 1
        elif h1 == 0 and h16 == 1:
            c_only += 1
    # McNemar continuity-corrected chi2 with 1 dof; reject "net16 worse" at
    # the conventional alpha=0.05 threshold (chi2_cc > 3.841).
    n = b_only + c_only
    chi2_cc = ((abs(b_only - c_only) - 1) ** 2 / n) if n else 0.0
    significantly_worse = chi2_cc > 3.841 and b_only > c_only
    assert not significantly_worse, (
        f"net16 significantly worse than net1 on dev_big: net1={hits1}/{len(rows)} "
        f"net16={hits16}/{len(rows)} b={b_only} c={c_only} chi2_cc={chi2_cc:.3f}")


def test_net16_dominates_net0b_on_dev_big_per_metric_sec7():
    """METRIC.md sec.7: A dominates B iff optimality>=, size<=, flops<=, with
    >=1 strict -- exactly the standard net0b.py itself used to claim
    dominance over net0d (there: tied optimality + strictly lower flops).
    Here: measured on dev_big(2000) (the correct measurement resolution per
    this generation's task frame -- sealed(300) alone is too noisy, see
    scripts/eval_resolution.py's module docstring), net16's optimality is
    >= net0b's, AND net16 is strictly smaller (size) AND strictly cheaper
    (flops, net1's 1-ply search vs net0b's depth-3 beam) -- two independent
    strict axes, not just one."""
    rows = _load(_DEV_BIG)
    card_0b = strength.score(Net0bAgent(), rows)
    card_16 = strength.score(Net16Agent(), rows)
    man_0b = Net0bAgent().manifest()
    man_16 = Net16Agent().manifest()
    assert card_16.optimality >= card_0b.optimality, (
        f"net16 opt={card_16.optimality} must be >= net0b opt={card_0b.optimality} "
        f"on dev_big (the resolution this task frame trusts)")
    assert man_16.size_bytes < man_0b.size_bytes
    assert man_16.flops_per_move < man_0b.flops_per_move


def test_qualifies_micro_budget_and_latency():
    import time
    ag = Net16Agent()
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
    assert_no_oracle_access('app.agents.net16')


def test_no_hardcoded_weights_smuggled_into_source():
    """Guard against the reward-hack pattern this task's own instructions
    warn about ("no smuggling weights into python"): net16.py itself must
    not contain any large numeric literal arrays -- all weight data must
    come from the on-disk artifact via load_compressed()."""
    src = open(os.path.join(_ROOT, "app", "agents", "net16.py"), encoding="utf-8").read()
    # a real weight dump would show up as a long run of comma-separated
    # floats; the actual source is short and has no such literal.
    assert "np.array(" not in src
    assert "np.asarray(" not in src
    assert len(src) < 4000, "net16.py should be a thin loader, not a weight dump"
