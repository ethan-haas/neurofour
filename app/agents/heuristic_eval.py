"""Shared static evaluation used by the heuristic and minimax agents.

Pure function of the board (no solver). Window-based Connect-4 heuristic plus a
center-column bonus, scored from the side-to-move's perspective.
"""
from __future__ import annotations

from app.engine.board import WIDTH, HEIGHT

# precompute all 4-in-a-row windows as lists of (row, col)
_WINDOWS = []
for r in range(HEIGHT):
    for c in range(WIDTH):
        if c + 3 < WIDTH:
            _WINDOWS.append([(r, c + i) for i in range(4)])          # horizontal
        if r + 3 < HEIGHT:
            _WINDOWS.append([(r + i, c) for i in range(4)])          # vertical
        if c + 3 < WIDTH and r + 3 < HEIGHT:
            _WINDOWS.append([(r + i, c + i) for i in range(4)])      # diag "/"
        if c - 3 >= 0 and r + 3 < HEIGHT:
            _WINDOWS.append([(r + i, c - i) for i in range(4)])      # diag "\"

_CENTER = WIDTH // 2

WIN_SCORE = 100_000


def _score_window(win_vals, me: int, opp: int) -> int:
    mine = win_vals.count(me)
    theirs = win_vals.count(opp)
    empty = win_vals.count(0)
    if mine and theirs:
        return 0
    if mine == 4:
        return WIN_SCORE
    if theirs == 4:
        return -WIN_SCORE
    if mine == 3 and empty == 1:
        return 50
    if mine == 2 and empty == 2:
        return 10
    if mine == 1 and empty == 3:
        return 1
    if theirs == 3 and empty == 1:
        return -60          # slightly heavier: value blocking
    if theirs == 2 and empty == 2:
        return -10
    if theirs == 1 and empty == 3:
        return -1
    return 0


def evaluate(board) -> int:
    """Static score from the perspective of the side to move."""
    me = board.player_to_move()
    opp = 2 if me == 1 else 1
    grid = board.cells()
    score = 0
    for win in _WINDOWS:
        vals = [grid[r][c] for (r, c) in win]
        score += _score_window(vals, me, opp)
    # center-column preference
    for r in range(HEIGHT):
        if grid[r][_CENTER] == me:
            score += 6
        elif grid[r][_CENTER] == opp:
            score -= 6
    return score
