"""gen-8 T4 (updated gen-10 T-lever): machine-checked `OPS_PER_NODE` honesty
for `neurofour-net14` (the zero-byte, zero-param pure-search agent). Mirrors
`tests/test_net13_flop_honesty.py`'s methodology exactly -- see that file's
module docstring for the full instrumentation rationale.

gen-10 update: net14's leaf evaluator changed from `heuristic_eval.
evaluate()` (a python-grid window scan, priced as one flat per-call weight
since it wasn't bit-op-shaped) to `heuristic_eval_bb.evaluate_bb()` (a
bitboard reimplementation, see that module's docstring). Because the new
leaf IS bit-op-shaped, this test now wraps its INDIVIDUAL primitives
(`_and`, `_xor`, `_popcount`, `_table_lookup`, `_center_combine`) and sums
their REAL observed call counts per node, instead of a single hand-derived
flat weight for the whole function -- a strictly more honest, finer-grained
machine check than the gen-8/9 version had available.
"""
from __future__ import annotations

import json
import os
import random

import app.agents.heuristic_eval as HE
from app.engine.board import Board
import app.agents.net14 as N14
import app.agents.heuristic_eval_bb as HEBB
from app.agents.net14 import Net14Agent, OPS_PER_NODE, EVAL_OPS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "bench_data")

# ---- hand-derived, documented per-primitive weights (see net13.py's module
# docstring for the board-primitive derivation; heuristic_eval_bb.py's
# module docstring for the bitboard-leaf-primitive derivation) -----------
WINNER_OPS = 39
PLAY_OPS = 11
LEGAL_MOVES_OPS = 35
COMPUTE_WINNING_POSITION_OPS = 74
POSSIBLE_OPS = 2
NODE_BOOKKEEPING_ALLOWANCE = 100

# gen-10 bitboard-leaf primitive weights (heuristic_eval_bb.py convention:
# `&`/`^` = 1 unit each, matching app/solver/solver.py's
# _compute_winning_position honesty price; `.bit_count()` and each
# list-index read = 1 unit each, matching heuristic_eval.py's own
# "bit ops, arithmetic, comparisons, and list-index reads... 1 unit each"
# convention).
AND_OP = 1
XOR_OP = 1
POPCOUNT_OP = 1
TABLE_LOOKUP_OP = 2      # table[mc][tc] = two chained index reads
CENTER_COMBINE_OP = 3    # 2 multiplies + 1 subtract


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


def _varied_boards():
    boards = []
    ps = _load(os.path.join(_DATA, "dev_big3.jsonl"), 300)
    boards += [Board.from_moves(p["board"]) for p in ps]
    boards += [Board.from_moves(seq) for seq in
               ([[]] + [[c] for c in range(7)] +
                [[3, 3], [3, 2], [2, 4], [3, 3, 3], [0, 6], [3, 4, 2, 5],
                 [1, 5], [0, 2], [6, 4]])]
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
    assert len(boards) >= 300
    return boards


class _Instrumentation:
    def __enter__(self):
        self.stack = []
        self.max_ops = 0.0

        self._orig_winner = Board.winner
        self._orig_play = Board.play
        self._orig_lm = Board.legal_moves
        self._orig_negamax = Net14Agent._negamax
        self._orig_cwp = N14._compute_winning_position
        self._orig_poss = N14._possible
        # gen-10: net14.heuristic_evaluate IS heuristic_eval_bb.evaluate_bb
        # (imported under that alias in net14.py) -- wrap its INDIVIDUAL
        # bitboard primitives (module-level functions on heuristic_eval_bb)
        # rather than the whole function as one flat weight, since it is
        # now genuinely bit-op-shaped and each primitive call is real,
        # separately-measurable work.
        self._orig_and = HEBB._and
        self._orig_xor = HEBB._xor
        self._orig_popcount = HEBB._popcount
        self._orig_table_lookup = HEBB._table_lookup
        self._orig_center_combine = HEBB._center_combine

        stack = self.stack

        def add(x):
            if stack:
                stack[-1] += x

        def winner_w(bself):
            add(WINNER_OPS)
            return self._orig_winner(bself)

        def play_w(bself, col):
            add(PLAY_OPS)
            return self._orig_play(bself, col)

        def lm_w(bself):
            add(LEGAL_MOVES_OPS)
            return self._orig_lm(bself)

        def cwp_w(pos, mask):
            add(COMPUTE_WINNING_POSITION_OPS)
            return self._orig_cwp(pos, mask)

        def poss_w(mask):
            add(POSSIBLE_OPS)
            return self._orig_poss(mask)

        def and_w(a, b):
            add(AND_OP)
            return self._orig_and(a, b)

        def xor_w(a, b):
            add(XOR_OP)
            return self._orig_xor(a, b)

        def popcount_w(x):
            add(POPCOUNT_OP)
            return self._orig_popcount(x)

        def table_lookup_w(table, mc, tc):
            add(TABLE_LOOKUP_OP)
            return self._orig_table_lookup(table, mc, tc)

        def center_combine_w(my_c, opp_c):
            add(CENTER_COMBINE_OP)
            return self._orig_center_combine(my_c, opp_c)

        Board.winner = winner_w
        Board.play = play_w
        Board.legal_moves = lm_w
        N14._compute_winning_position = cwp_w
        N14._possible = poss_w
        HEBB._and = and_w
        HEBB._xor = xor_w
        HEBB._popcount = popcount_w
        HEBB._table_lookup = table_lookup_w
        HEBB._center_combine = center_combine_w

        def negamax_w(aself, parent, col, depth_used, depth_remaining, alpha, beta):
            stack.append(0.0)
            r = self._orig_negamax(aself, parent, col, depth_used, depth_remaining, alpha, beta)
            own = stack.pop() + NODE_BOOKKEEPING_ALLOWANCE
            if own > self.max_ops:
                self.max_ops = own
            return r

        Net14Agent._negamax = negamax_w
        return self

    def __exit__(self, *exc):
        Board.winner = self._orig_winner
        Board.play = self._orig_play
        Board.legal_moves = self._orig_lm
        Net14Agent._negamax = self._orig_negamax
        N14._compute_winning_position = self._orig_cwp
        N14._possible = self._orig_poss
        HEBB._and = self._orig_and
        HEBB._xor = self._orig_xor
        HEBB._popcount = self._orig_popcount
        HEBB._table_lookup = self._orig_table_lookup
        HEBB._center_combine = self._orig_center_combine
        return False


def test_max_observed_ops_per_node_within_declared_bound():
    boards = _varied_boards()
    configs = [200, 1428, 3000]

    overall_max = 0.0
    with _Instrumentation() as inst:
        for m_budget in configs:
            agent = Net14Agent(m_budget=m_budget, max_depth=14)
            for b in boards:
                agent.select_move(b)
            if inst.max_ops > overall_max:
                overall_max = inst.max_ops

    assert overall_max <= OPS_PER_NODE, (
        f"OBSERVED max ops/node = {overall_max} EXCEEDS declared "
        f"OPS_PER_NODE={OPS_PER_NODE} -- raise it (never weaken this test)."
    )
    assert overall_max > 0


def test_actual_nodes_never_exceed_declared_budget():
    boards = _varied_boards()
    for m_budget in [50, 1428, 3000]:
        agent = Net14Agent(m_budget=m_budget, max_depth=14)
        for b in boards[:200]:
            agent.select_move(b)
            assert agent._nodes <= m_budget


def test_manifest_flops_formula_no_n_term():
    """flops_per_move must equal M*OPS_PER_NODE + guard_bitops EXACTLY --
    net14 has NO leaf-eval (N) term at all, unlike net13."""
    from app.engine.board import WIDTH
    for m_budget in [50, 1428, 5000]:
        ag = Net14Agent(m_budget=m_budget)
        man = ag.manifest()
        expected = m_budget * OPS_PER_NODE + 4 * WIDTH
        assert man.flops_per_move == expected
        assert man.params == 0
        assert man.size_bytes == 0
