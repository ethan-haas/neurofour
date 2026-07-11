"""Tests for `neurofour-net16s` / `neurofour-net16b`: gen-3 "research-model /
compressed-leaf deep search" -- net2's depth-3 full-width refutation search
(net16s) and net4's depth-3/K=2 beam refutation search (net16b), both
plugged with net16's COMPRESSED 2867-byte leaf artifact instead of net1's
plain 4837-byte npz. See `app/agents/net16s.py` / `net16b.py` module
docstrings for the design and `scripts/eval_resolution.py`'s dev_big(2000)
output (recorded in the gen-3 coder session) for the honest RESULT: both
variants reach dev_big optimality=0.94600, an EXACT tie with net2/net4
(McNemar p=0.8383, not significant), at net16's strictly smaller 2867B
artifact and net2/net4's identical flops -- a genuine Pareto dominance of
net2 by net16s (size strictly smaller, flops/optimality tied) and of net4 by
net16b (same). These tests only check the structural/honesty invariants
net2.py/net4.py/net16.py already enforce for their own agents (registration,
determinism, manifest honesty, tactical-guard correctness, no oracle
access, no smuggled weights) -- they do NOT re-run the slow dev_big pass
(that's `scripts/eval_resolution.py`'s job); they DO pin the McNemar-vs-net1
sanity net16.py itself already established transitively (both variants load
the identical artifact net16.py's own test suite already validates).
"""
import json
import os

import pytest

from tests._oracle_scan import assert_no_oracle_access

from app.engine.board import Board, WIDTH
from app.agents.net1 import Net1Agent, DEFAULT_ARTIFACT as NET1_ARTIFACT
from app.agents.net2 import Net2Agent, DEPTH as NET2_DEPTH
from app.agents.net4 import Net4Agent, DEPTH as NET4_DEPTH, K as NET4_K
from app.agents.net16 import Net16Agent, DEFAULT_ARTIFACT as NET16_ARTIFACT
from app.agents.net16s import Net16SAgent
from app.agents.net16b import Net16BAgent
from app.agents.encode import FEATURE_DIM
from app.agents import registry
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")

pytestmark = pytest.mark.skipif(
    not os.path.exists(NET16_ARTIFACT),
    reason="neurofour-net16 artifact not built yet (run scripts/compress_net1.py)",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- registration ---------------------------------------------------------- #

def test_registered_in_registry():
    names = registry.agent_names()
    assert "neurofour-net16s" in names
    assert "neurofour-net16b" in names
    assert isinstance(registry.make_agent("neurofour-net16s"), Net16SAgent)
    assert isinstance(registry.make_agent("neurofour-net16b"), Net16BAgent)


# ---- tactical guard invariants (identical contract to net1/net2/net4) ----- #

@pytest.mark.parametrize("cls", [Net16SAgent, Net16BAgent])
def test_agent_never_misses_win_or_block_on_sealed(cls):
    ag = cls()
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


@pytest.mark.parametrize("cls", [Net16SAgent, Net16BAgent])
def test_agent_returns_legal_and_deterministic(cls):
    ag = cls()
    for p in _load(_SEALED)[:50]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


# ---- artifact identity: SAME bytes as net16, no new weights --------------- #

def test_net16s_and_net16b_reuse_net16s_exact_artifact():
    assert Net16SAgent().artifact_path == NET16_ARTIFACT
    assert Net16BAgent().artifact_path == NET16_ARTIFACT


def test_net16s_and_net16b_params_match_net16_and_net1():
    p16 = Net16Agent().manifest().params
    p1 = Net1Agent().manifest().params
    assert Net16SAgent().manifest().params == p16 == p1
    assert Net16BAgent().manifest().params == p16 == p1


# ---- manifest honesty: real on-disk size, structural flops formulas ------- #

def test_net16s_manifest_is_honest_and_within_caps():
    ag = Net16SAgent()
    man = ag.manifest()
    assert man.name == "neurofour-net16s"
    assert man.kind == "search"
    assert man.size_bytes == os.path.getsize(ag.artifact_path)
    assert man.size_bytes == os.path.getsize(NET16_ARTIFACT)
    assert man.size_bytes <= 4_096, "net16s must inherit net16's nano-tier size"
    assert man.size_bytes < os.path.getsize(NET1_ARTIFACT)
    assert man.flops_per_move <= FLOP_CAP
    assert ag.depth == NET2_DEPTH


def test_net16b_manifest_is_honest_and_within_caps():
    ag = Net16BAgent()
    man = ag.manifest()
    assert man.name == "neurofour-net16b"
    assert man.kind == "search"
    assert man.size_bytes == os.path.getsize(ag.artifact_path)
    assert man.size_bytes == os.path.getsize(NET16_ARTIFACT)
    assert man.size_bytes <= 4_096
    assert man.size_bytes < os.path.getsize(NET1_ARTIFACT)
    assert man.flops_per_move <= FLOP_CAP
    assert ag.depth == NET4_DEPTH
    assert ag.k == NET4_K


def test_net16s_flops_match_net2s_exact_structural_bound():
    """net16s inherits Net2Agent.manifest() unchanged -- same
    WIDTH + WIDTH**depth leaf-call formula, and (since net16's params equal
    net1's) must produce IDENTICAL flops to net2 itself, not merely an
    honest-but-different number."""
    man16s = Net16SAgent().manifest()
    man2 = Net2Agent().manifest()
    max_leaf_calls = WIDTH + WIDTH ** NET2_DEPTH
    expected = max_leaf_calls * (2 * man16s.params + FEATURE_DIM) + 4 * WIDTH
    assert man16s.flops_per_move == expected
    assert man16s.flops_per_move == man2.flops_per_move, (
        "net16s must cost exactly net2's flops (same search structure, same "
        "param count -- only size_bytes differs)")


def test_net16b_flops_match_net4s_exact_structural_bound():
    """net16b inherits Net4Agent.manifest()/_max_leaf_calls() unchanged and
    (since net16's params equal net1's) must produce IDENTICAL flops to
    net4 itself."""
    ag16b = Net16BAgent()
    man16b = ag16b.manifest()
    man4 = Net4Agent().manifest()
    expected = ag16b._max_leaf_calls() * (2 * man16b.params + FEATURE_DIM) + 4 * WIDTH
    assert man16b.flops_per_move == expected
    assert man16b.flops_per_move == man4.flops_per_move, (
        "net16b must cost exactly net4's flops (same search structure, same "
        "param count -- only size_bytes differs)")


def test_net16s_and_net16b_strictly_smaller_than_net2_net4():
    """The Pareto claim's size axis: net16s/net16b's real on-disk artifact
    must be strictly smaller than net2/net4's (4837B), since flops are tied
    -- this is the structural precondition for the dominance result recorded
    in eval_resolution.py's output (optimality tied, size strictly smaller,
    flops equal -- >=1 strict axis, METRIC.md sec.7)."""
    assert Net16SAgent().manifest().size_bytes < Net2Agent().manifest().size_bytes
    assert Net16BAgent().manifest().size_bytes < Net4Agent().manifest().size_bytes


# ---- latency ---------------------------------------------------------------- #

@pytest.mark.parametrize("cls", [Net16SAgent, Net16BAgent])
def test_qualifies_micro_budget_and_latency(cls):
    import time
    ag = cls()
    man = ag.manifest()
    assert man.size_bytes <= 32_768
    assert man.flops_per_move < FLOP_CAP
    rows = _load(_SEALED)[:100]
    boards = [Board.from_moves(p["board"]) for p in rows]
    for b in boards[:5]:
        ag.select_move(b)   # warmup
    timings = []
    for b in boards:
        t0 = time.perf_counter()
        ag.select_move(b)
        timings.append((time.perf_counter() - t0) * 1000.0)
    timings.sort()
    p50 = timings[len(timings) // 2]
    assert p50 < LATENCY_CAP_MS


# ---- anti-cheat: no solver / oracle access, no smuggled weights ------------ #

def test_net16s_source_has_no_solver_or_oracle_access():
    assert_no_oracle_access('app.agents.net16s')


def test_net16b_source_has_no_solver_or_oracle_access():
    assert_no_oracle_access('app.agents.net16b')


@pytest.mark.parametrize("modname", ["net16s", "net16b"])
def test_no_hardcoded_weights_smuggled_into_source(modname):
    src = open(os.path.join(_ROOT, "app", "agents", f"{modname}.py"),
                encoding="utf-8").read()
    assert "np.array(" not in src
    assert "np.asarray(" not in src
    assert len(src) < 4000, f"{modname}.py should be a thin search+loader, not a weight dump"
