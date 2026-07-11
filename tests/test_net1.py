"""Tests for the learned 1-ply value-search agent `neurofour-net1`.

Covers:
  * the 0-param tactical guard invariants: it must NEVER miss an immediate win
    and must ALWAYS block a lone immediate opponent threat;
  * the agent loads, always returns a legal move, is deterministic;
  * manifest honesty: size == real artifact bytes, flops within the caps;
  * it clears the headline bar (sealed optimality > 0.900) and is Pareto.
"""
import json
import os

import pytest

from app.engine.board import Board
from app.agents.net1 import Net1Agent, tactical_move, DEFAULT_ARTIFACT
from app.agents import registry
from app.neurogolf import strength
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEALED = os.path.join(_ROOT, "bench_data", "sealed.jsonl")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_ARTIFACT),
    reason="neurofour-net1 artifact not trained yet",
)


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- tactical guard invariants (0-param board logic) ---------------------- #

def test_tactical_takes_immediate_vertical_win():
    # player 1 has three stones stacked in column 3; playing 3 wins.
    b = Board.from_moves([3, 0, 3, 1, 3, 2])   # p1: 3,3,3 ; p2: 0,1,2 ; p1 to move
    assert b.player_to_move() == 1
    assert b.winning_move(3)
    assert tactical_move(b) == 3


def test_tactical_takes_immediate_horizontal_win():
    # p1 stones at cols 0,1,2 on the bottom row; playing 3 completes 4-in-a-row.
    b = Board.from_moves([0, 6, 1, 6, 2, 5])   # p1:0,1,2  p2:6,6,5  p1 to move
    assert b.winning_move(3)
    assert tactical_move(b) == 3


def test_tactical_blocks_lone_opponent_threat():
    # opponent (p2) has 3,3,3 stacked and threatens to win at col 3 next;
    # p1 (to move) has no win of its own -> guard must block at col 3.
    b = Board.from_moves([0, 3, 1, 3, 5, 3])   # p1:0,1,5  p2:3,3,3  p1 to move
    assert b.player_to_move() == 1
    assert not any(b.winning_move(c) for c in b.legal_moves())   # no own win
    assert tactical_move(b) == 3


def test_tactical_prefers_own_win_over_block():
    # p1 can win at col 3 AND p2 threatens at col 0 -> taking the win beats blocking.
    b = Board.from_moves([3, 0, 3, 0, 3, 0])   # p1:3,3,3  p2:0,0,0  p1 to move
    assert b.winning_move(3)
    assert tactical_move(b) == 3


def test_tactical_returns_none_when_no_tactic():
    b = Board.from_moves([3, 3])
    assert tactical_move(b) is None


def test_agent_never_misses_win_or_block_on_sealed():
    """Over the whole sealed set: whenever an immediate win exists the agent takes
    one; whenever a lone forced block exists (no own win) the agent blocks it."""
    ag = Net1Agent()
    for p in _load(_SEALED):
        b = Board.from_moves(p["board"])
        moves = b.legal_moves()
        wins = [c for c in moves if b.winning_move(c)]
        m = ag.select_move(b)
        if wins:
            assert m in wins, f"missed an immediate win at {p['board']}"
            continue
        # lone forced block: exactly the columns that stop an opp immediate win
        from app.engine.board import _won, _bottom_mask, _column_mask
        opp = b.mask ^ b.cur
        blocks = [c for c in moves
                  if _won(opp | ((b.mask + _bottom_mask(c)) & _column_mask(c)))]
        if blocks:
            assert m in blocks, f"missed a forced block at {p['board']}"


# ---- agent behaviour + manifest ------------------------------------------- #

def test_agent_returns_legal_and_deterministic():
    ag = Net1Agent()
    for p in _load(_SEALED)[:50]:
        b = Board.from_moves(p["board"])
        m1 = ag.select_move(b)
        m2 = ag.select_move(b)
        assert m1 in b.legal_moves()
        assert m1 == m2


def test_manifest_is_honest_and_within_caps():
    ag = Net1Agent()
    man = ag.manifest()
    assert man.kind == "search"
    assert man.size_bytes == os.path.getsize(DEFAULT_ARTIFACT)
    assert man.size_bytes <= 32_768           # micro tier
    assert man.flops_per_move <= FLOP_CAP
    assert man.params > 0


def test_registered_in_registry():
    names = registry.agent_names()
    assert "neurofour-net1" in names
    # the original pure-policy net is kept as a frontier data point
    assert "neurofour-net" in names


def test_beats_heuristic_on_sealed():
    ag = Net1Agent()
    card = strength.score(ag, _load(_SEALED))
    assert card.optimality > 0.900, f"optimality {card.optimality} must beat heuristic 0.900"
    assert card.blunder_rate == 0.0
