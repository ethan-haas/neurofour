"""Solver correctness tests.

Exact full solves are only fast for positions with enough stones, so tactical
checks (immediate win / forced block) use `solve_best`, which short-circuits on an
immediate win, and the exact per-column / determinism / symmetry checks use deep
positions (few empty cells). The empty-board (first-player-wins-with-center) fact
is checked separately and only when the committed opening book is present.
"""
import random

import pytest

from app.engine.board import Board, SIZE
from app.solver.solver import Solver, solve, _load_book


def _sign(x):
    return (x > 0) - (x < 0)


def _deep_nonterminal(seed, lo=22, hi=32):
    rng = random.Random(seed)
    while True:
        b = Board.empty()
        for _ in range(rng.randint(lo, hi)):
            if b.is_terminal():
                break
            b = b.play(rng.choice(b.legal_moves()))
        if not b.is_terminal() and b.legal_moves():
            return b


# ---- tactical: immediate wins (solve_best short-circuits -> fast) ---------- #
def test_immediate_horizontal_win_chosen():
    b = Board.from_moves([0, 6, 1, 6, 2, 5])       # P1 has 0,1,2 on row0
    s = Solver().solve_best(b, mode="scored")
    assert s.best_col == 3
    assert s.value == SIZE - b.n                   # win this move


def test_immediate_vertical_win_chosen():
    b = Board.from_moves([3, 4, 3, 4, 3, 4])
    s = Solver().solve_best(b, mode="scored")
    assert s.best_col == 3
    assert s.value > 0


def test_immediate_diagonal_win_chosen():
    # P1 completes a "/" diagonal (0,0),(1,1),(2,2),(3,3)
    b = Board.from_moves([0, 1, 1, 2, 2, 6, 2, 3, 3, 5, 3])
    assert any(b.winning_move(c) for c in b.legal_moves())   # a win is available
    s = Solver().solve_best(b, mode="scored")
    assert s.value > 0
    assert b.winning_move(s.best_col)


# ---- forced block on a deep position (fast to solve) ---------------------- #
def _find_forced_block(seed):
    """A deep position where the mover has no immediate win but the opponent has
    exactly one immediate winning reply -> the mover must occupy that column."""
    rng = random.Random(seed)
    for _ in range(4000):
        b = Board.empty()
        depth = rng.randint(16, 30)
        for _ in range(depth):
            if b.is_terminal():
                break
            b = b.play(rng.choice(b.legal_moves()))
        if b.is_terminal() or not b.legal_moves():
            continue
        my_wins = [c for c in b.legal_moves() if b.winning_move(c)]
        if my_wins:
            continue
        opp_threats = []
        for c in b.legal_moves():
            after = b.play(c)
            if any(after.winning_move(oc) for oc in after.legal_moves()):
                opp_threats.append(c)
        # exactly one move is "safe" -> that is the forced block
        safe = [c for c in b.legal_moves() if c not in opp_threats]
        if len(safe) == 1:
            return b, safe[0]
    return None, None


def test_forced_block_chosen():
    b, block = _find_forced_block(12345)
    if b is None:
        pytest.skip("no forced-block position sampled")
    s = Solver().solve_best(b, mode="scored")
    assert s.best_col == block


# ---- exact solves on deep positions --------------------------------------- #
def test_determinism_repeat():
    b = _deep_nonterminal(3)
    s = Solver()
    s1 = s.solve(b)
    s2 = s.solve(b)
    assert s1.per_col == s2.per_col
    assert s1.optimal_cols == s2.optimal_cols
    assert s1.best_col == s2.best_col


def test_mirror_consistency():
    # play a deep non-terminal game and its column-mirror; the solver must agree
    seq = None
    for seed in range(200):
        rng = random.Random(seed)
        s2 = []
        bb = Board.empty()
        for _ in range(rng.randint(22, 30)):
            if bb.is_terminal():
                break
            c = rng.choice(bb.legal_moves())
            s2.append(c)
            bb = bb.play(c)
        if not bb.is_terminal() and bb.legal_moves():
            seq = s2
            break
    assert seq is not None
    src = Board.from_moves(seq)
    mir = Board.from_moves([6 - c for c in seq])
    s = Solver()
    sb = s.solve(src)
    sm = s.solve(mir)
    assert sorted(6 - c for c in sb.optimal_cols) == sm.optimal_cols
    assert sb.value == sm.value


def test_value_equals_sign_of_scored():
    s = Solver()
    checked = 0
    for seed in range(40):
        b = _deep_nonterminal(100 + seed)
        sc = s.solve(b, mode="scored")
        vv = s.solve(b, mode="value")
        assert vv.value == _sign(sc.value)
        assert {c: _sign(v) for c, v in sc.per_col.items()} == vv.per_col
        checked += 1
    assert checked > 5


def test_solve_best_matches_solve():
    s = Solver()
    for seed in range(30):
        b = _deep_nonterminal(200 + seed)
        full = s.solve(b, mode="value")
        best = s.solve_best(b, mode="value")
        assert best.value == full.value
        assert best.best_col in full.optimal_cols


def test_solve_best_scored_matches_solve_scored():
    """Regression: `solve_best(mode="scored")` used to trust a narrow-window
    sibling search as if it were exact. A narrow-window search that ties or
    fails high at the window boundary is only a BOUND, not the true value --
    treating it as exact could (and did) inflate a suboptimal move's recorded
    score up to the current best, corrupting both `optimal_cols` and (via the
    center-preference tie-break) `best_col` itself. Concretely, on
    board="4,3,1,1,5,0,0,6,0,1,5,6,1,2" the old code returned best_col=2
    (true scored value 6) instead of the actual best, value 12 (cols 0/4/5).
    """
    s = Solver()
    checked = 0
    for seed in range(12):
        b = _deep_nonterminal(100 + seed, lo=18, hi=26)
        full = s.solve(b, mode="scored")
        best = s.solve_best(b, mode="scored")
        assert best.value == full.value, (seed, b.to_key())
        assert best.best_col in full.optimal_cols, (seed, b.to_key())
        assert best.optimal_cols == full.optimal_cols, (seed, b.to_key())
        checked += 1
    assert checked > 5


def test_solve_best_scored_known_regression_board():
    b = Board.from_moves("4,3,1,1,5,0,0,6,0,1,5,6,1,2")
    full = Solver().solve(b, mode="scored")
    best = Solver().solve_best(b, mode="scored")
    assert full.value == 12
    assert full.optimal_cols == [0, 4, 5]
    assert best.value == 12
    assert best.optimal_cols == [0, 4, 5]
    assert best.best_col == 4


def test_scored_prefers_faster_mate():
    # an immediate win outscores any slower continuation
    b = Board.from_moves([0, 6, 1, 6, 2, 5])
    s = Solver().solve_best(b, mode="scored")
    assert s.value == SIZE - b.n                    # earliest possible win value


@pytest.mark.skipif(not _load_book(), reason="opening book not built")
def test_empty_board_is_first_player_win_center():
    s = solve(Board.empty(), mode="value")
    assert s.value == 1
    assert s.best_col == 3
