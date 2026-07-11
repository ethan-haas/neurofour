"""Tests for `neurofour-net15` (joint policy-value 1-ply agent) and
`neurofour-net15s` (net2's search structure + net15's leaf). See
`app/agents/net15.py` / `net15s.py` docstrings for the honest RESULTS
(rigorous negative vs both net1 and net2 on dev_big(2000)) -- these tests
only check the same structural/honesty invariants net1.py/net2.py already
enforce for their own agents (registration, determinism, manifest honesty,
tactical-guard correctness, no oracle access); they do NOT assert net15
beats anything, matching the measured result.
"""
import json
import os

import pytest

from tests._oracle_scan import assert_no_oracle_access

from app.engine.board import Board, _won, _bottom_mask, _column_mask
from app.agents.net1 import DEFAULT_ARTIFACT as NET1_ARTIFACT
from app.agents.net15 import Net15Agent, DEFAULT_ARTIFACT as NET15_ARTIFACT
from app.agents.net15s import Net15SAgent
from app.agents.net2 import DEPTH
from app.agents import registry
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")

pytestmark = pytest.mark.skipif(
    not os.path.exists(NET15_ARTIFACT),
    reason="neurofour-net15 artifact not trained yet (run train_net15.py)",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- registration ----------------------------------------------------- #

def test_registered_in_registry():
    names = registry.agent_names()
    assert "neurofour-net15" in names
    assert "neurofour-net15s" in names


# ---- tactical guard invariants (identical contract to net1/net2) ------- #

@pytest.mark.parametrize("cls", [Net15Agent, Net15SAgent])
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
        opp = b.mask ^ b.cur
        blocks = [c for c in moves
                  if _won(opp | ((b.mask + _bottom_mask(c)) & _column_mask(c)))]
        if blocks:
            assert m in blocks, f"missed a forced block at {p['board']}"


@pytest.mark.parametrize("cls", [Net15Agent, Net15SAgent])
def test_agent_returns_legal_and_deterministic(cls):
    ag = cls()
    for p in _load(_SEALED)[:50]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


# ---- manifest honesty ---------------------------------------------------- #

def test_net15_manifest_is_honest_and_within_caps():
    ag = Net15Agent()
    man = ag.manifest()
    assert man.name == "neurofour-net15"
    assert man.kind == "search"
    assert man.size_bytes == os.path.getsize(NET15_ARTIFACT)
    assert man.size_bytes <= 32_768
    assert man.flops_per_move <= FLOP_CAP
    assert man.params > 0


def test_net15s_manifest_is_honest_and_within_caps():
    ag = Net15SAgent()
    man = ag.manifest()
    assert man.name == "neurofour-net15s"
    assert man.kind == "search"
    assert man.size_bytes == os.path.getsize(ag.artifact_path)
    assert man.size_bytes <= 32_768
    assert man.flops_per_move <= FLOP_CAP
    assert man.params > 0
    assert ag.depth == DEPTH


def test_net15s_manifest_flops_match_structural_bound():
    """Same structural bound net2 enforces (test_net2.py) -- net15s reuses
    Net2Agent's manifest() unchanged, only the artifact differs, so the
    formula must still be exactly WIDTH + WIDTH**depth leaf calls."""
    from app.engine.board import WIDTH
    from app.agents.encode import FEATURE_DIM
    ag = Net15SAgent()
    man = ag.manifest()
    max_leaf_calls = WIDTH + WIDTH ** ag.depth
    expected = max_leaf_calls * (2 * man.params + FEATURE_DIM) + 4 * WIDTH
    assert man.flops_per_move == expected


def test_net15_size_and_flops_identical_formula_to_net1():
    """The policy head is training-only and dropped at export -- net15's
    exported artifact is a plain net1-format 4-array npz, so params/size are
    computed purely from the artifact, matching net1's formula exactly (the
    policy head can never smuggle uncounted flops into inference)."""
    from app.engine.board import WIDTH
    from app.agents.encode import FEATURE_DIM
    ag = Net15Agent()
    man = ag.manifest()
    expected = WIDTH * (2 * man.params + FEATURE_DIM)
    assert man.flops_per_move == expected


# ---- anti-cheat: no solver / oracle access at inference -------------------- #

def test_net15_source_has_no_solver_or_oracle_access():
    assert_no_oracle_access('app.agents.net15')


def test_net15s_source_has_no_solver_or_oracle_access():
    assert_no_oracle_access('app.agents.net15s')


# ---- honest sanity: the exported artifact is a proper npz value net ------- #

def test_net15_loads_and_scores_legal_moves_on_sealed():
    ag = Net15Agent()
    for p in _load(_SEALED)[:100]:
        b = Board.from_moves(p["board"])
        assert ag.select_move(b) in b.legal_moves()
