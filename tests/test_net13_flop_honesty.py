"""gen-8 T1/T2: machine-checked `OPS_PER_NODE` honesty for `neurofour-net13`.

An independent gen-8 auditor found the gen-7 `OPS_PER_NODE=300` constant was
reverse-engineered to the flop cap (break-even 300.015 for M=14617), not
derived from the code, and that its own worst-case-node claim was backwards
(claimed interior > leaf; the leaf's free-exact-resolution loop, up to 7
`winning_move()` calls each doing a full `_won()` board scan, is actually the
most expensive single node). See `app/agents/net13.py`'s module docstring,
"gen-8 T1 AUDIT + FIX" / "gen-8 T2 FIX" sections, for the full per-node-type
op derivation this test enforces.

Methodology (function-call counting via ordinary monkeypatch wrapping of the
board/solver primitives net13 calls -- NOT interpreter-level tracing hooks,
NOT anything installed inside the agent module itself): each primitive is
wrapped to add its own HAND-DERIVED, documented arithmetic/bit-op weight
(matching the module docstring's per-primitive breakdown) to a per-node
"currently executing frame" accumulator. `Net13Agent._negamax` is wrapped to
push/pop that accumulator around each of its own invocations, so a child
node's primitive calls (made while the child's OWN frame is on top of the
stack) are correctly excluded from the parent's own-cost total -- only work
done DIRECTLY inside a node's own `_negamax` body (before/after recursing,
never work attributable to a deeper call) is attributed to that node. A fixed
`NODE_BOOKKEEPING_ALLOWANCE` covers the remaining non-primitive-call inline
arithmetic (self._nodes += 1, PVS call-site negations, `_order_moves`'s
history-key negations, `_record_cutoff`'s multiply-add) that this instrumen-
tation cannot attribute via primitive-call wrapping alone -- see the module
docstring for its own derivation (worst case ~81 ops, allowance rounded to
100 for margin).

This test intentionally tracks BOTH the pre-T2 primitive (`Board.winning_
move`, called in a per-column loop) and the post-T2 primitives
(`_compute_winning_position`/`_possible`, called once per node) so it keeps
working unmodified across the T1 -> T2 code transition: whichever primitive
the CURRENT code actually calls contributes its measured weight; the other
contributes zero.
"""
from __future__ import annotations

import json
import os
import random

import pytest

import app.engine.board as B
from app.engine.board import Board
from app.agents.net1 import DEFAULT_ARTIFACT
import app.agents.net13 as N13
from app.agents.net13 import Net13Agent, OPS_PER_NODE

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "bench_data")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_ARTIFACT),
    reason="neurofour-net13's leaf artifact (net1's) not trained yet",
)

# ---- hand-derived, documented per-primitive weights (arithmetic/bit ops
# only: << >> & | ^ + - * plus unary -/~; matches net13.py's module docstring
# derivation exactly) -----------------------------------------------------
WON_OPS = 19               # _won(): 4 shift/and pairs, all-false (worst) path
WINNER_OPS = 39            # winner(): 1 xor + 2x _won() worst-case
WINNING_MOVE_OPS = 33      # winning_move(): can_play(5) + move_bit(8) + 1 or + _won(19)
PLAY_OPS = 11              # play(): top_mask guard(5) + new_mask(4) + xor(1) + n+1(1)
LEGAL_MOVES_OPS = 35       # legal_moves(): 7x [top_mask(4) + 1 and]
COMPUTE_WINNING_POSITION_OPS = 74   # vertical(5) + 3x[horizontal/diag block](22 each) + 3
POSSIBLE_OPS = 2           # (mask + BOTTOM) & BOARD_MASK

# non-primitive inline call-site arithmetic this instrumentation cannot
# attribute via function wrapping (self._nodes += 1, PVS negations,
# _order_moves history-key negation, _record_cutoff) -- see module docstring
# "gen-8 T1" derivation: worst case ~81 ops (7-wide interior, 6 PVS
# re-searches); rounded up to 100 for margin.
NODE_BOOKKEEPING_ALLOWANCE = 100


def _load(path, k=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if k and len(out) >= k:
                break
    return out


def _varied_boards(n_midgame=300):
    """>=300 boards spanning early-ply full-width, midgame, and near-terminal
    positions (the three phases the module docstring's worst-case analysis
    depends on: full 7-wide branching only occurs early; near-terminal boards
    exercise the free-exact-resolution leaf path most often)."""
    boards = []
    ps = _load(os.path.join(_DATA, "dev_big3.jsonl"), n_midgame)
    boards += [Board.from_moves(p["board"]) for p in ps]

    # early-ply full-width (every single first move + a spread of 2-ply lines)
    boards += [Board.from_moves(seq) for seq in
               ([[]] + [[c] for c in range(7)] +
                [[3, 3], [3, 2], [2, 4], [3, 3, 3], [0, 6], [3, 4, 2, 5],
                 [1, 5], [0, 2], [6, 4]])]

    # near-terminal: many random legal moves, stop before terminal
    rng = random.Random(20260710)
    for _ in range(60):
        b = Board.empty()
        depth = rng.randint(28, 41)
        for _ in range(depth):
            if b.is_terminal():
                break
            b = b.play(rng.choice(b.legal_moves()))
        if not b.is_terminal():
            boards.append(b)

    assert len(boards) >= 300, f"only {len(boards)} boards, need >= 300"
    return boards


class _Instrumentation:
    """Installs/uninstalls the weighted primitive-call + per-node stack
    instrumentation. Use as a context manager."""

    def __enter__(self):
        self.stack = []
        self.max_ops = 0.0
        self.observations = []

        self._orig_winner = Board.winner
        self._orig_wm = Board.winning_move
        self._orig_play = Board.play
        self._orig_lm = Board.legal_moves
        self._orig_negamax = Net13Agent._negamax
        self._orig_cwp = getattr(N13, "_compute_winning_position", None)
        self._orig_poss = getattr(N13, "_possible", None)

        stack = self.stack

        def add(x):
            if stack:
                stack[-1] += x

        def winner_w(bself):
            add(WINNER_OPS)
            return self._orig_winner(bself)

        def wm_w(bself, col):
            add(WINNING_MOVE_OPS)
            return self._orig_wm(bself, col)

        def play_w(bself, col):
            add(PLAY_OPS)
            return self._orig_play(bself, col)

        def lm_w(bself):
            add(LEGAL_MOVES_OPS)
            return self._orig_lm(bself)

        Board.winner = winner_w
        Board.winning_move = wm_w
        Board.play = play_w
        Board.legal_moves = lm_w

        if self._orig_cwp is not None:
            def cwp_w(pos, mask):
                add(COMPUTE_WINNING_POSITION_OPS)
                return self._orig_cwp(pos, mask)
            N13._compute_winning_position = cwp_w

        if self._orig_poss is not None:
            def poss_w(mask):
                add(POSSIBLE_OPS)
                return self._orig_poss(mask)
            N13._possible = poss_w

        def negamax_w(aself, parent, col, depth_used, depth_remaining, alpha, beta):
            stack.append(0.0)
            r = self._orig_negamax(aself, parent, col, depth_used, depth_remaining, alpha, beta)
            own = stack.pop() + NODE_BOOKKEEPING_ALLOWANCE
            self.observations.append(own)
            if own > self.max_ops:
                self.max_ops = own
            return r

        Net13Agent._negamax = negamax_w
        return self

    def __exit__(self, *exc):
        Board.winner = self._orig_winner
        Board.winning_move = self._orig_wm
        Board.play = self._orig_play
        Board.legal_moves = self._orig_lm
        Net13Agent._negamax = self._orig_negamax
        if self._orig_cwp is not None:
            N13._compute_winning_position = self._orig_cwp
        if self._orig_poss is not None:
            N13._possible = self._orig_poss
        return False


def test_max_observed_ops_per_node_within_declared_bound():
    """The core honesty gate: the weighted, per-node-isolated op cost this
    instrumentation observes, over >=300 boards spanning early-ply
    full-width / midgame / near-terminal positions and several (N, M)
    configs, must never exceed the declared `OPS_PER_NODE`. If this fails,
    raise `OPS_PER_NODE` -- never weaken this test."""
    boards = _varied_boards()
    configs = [(0, 2000), (64, 14617), (128, 8000), (256, 3000)]

    overall_max = 0.0
    with _Instrumentation() as inst:
        for n_budget, m_budget in configs:
            agent = Net13Agent(n_budget=n_budget, m_budget=m_budget, max_depth=14)
            for b in boards:
                agent.select_move(b)
            if inst.max_ops > overall_max:
                overall_max = inst.max_ops

    assert overall_max <= OPS_PER_NODE, (
        f"OBSERVED max ops/node = {overall_max} EXCEEDS declared "
        f"OPS_PER_NODE={OPS_PER_NODE} -- the constant is dishonest, raise it "
        f"(never weaken this test)."
    )
    # sanity: we should have actually exercised real search nodes, not a
    # degenerate all-tactical-guard sample.
    assert overall_max > 0


def test_actual_evals_and_nodes_never_exceed_declared_budget():
    """Kept alongside the ops-per-node check per the T1 spec: the existing
    hard dual-counter self-check, re-verified here too."""
    boards = _varied_boards()
    for n_budget, m_budget in [(0, 500), (64, 14617), (128, 8000), (256, 3000)]:
        agent = Net13Agent(n_budget=n_budget, m_budget=m_budget, max_depth=14)
        for b in boards[:200]:
            agent.select_move(b)
            assert agent._evals <= n_budget
            assert agent._nodes <= m_budget
