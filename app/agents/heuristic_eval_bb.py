"""gen-10 T-lever: bitboard reimplementation of `heuristic_eval.evaluate()`
for `neurofour-net14`'s leaf, used ONLY by `net14.py` (never by the
`heuristic`/`minimax` baseline agents, which keep using
`app/agents/heuristic_eval.py::evaluate()` unchanged, per the task's "prefer
a NEW module over editing heuristic_eval.py" instruction).

`heuristic_eval.evaluate()` builds a 6x7 python GRID via `board.cells()`
(42 cells, each a bit-test into two of the object's ints) then scans 69
4-cell windows via THREE `list.count()` calls + a ~9-branch score-bucket
chain per window. None of that is bit-op-shaped even though the underlying
board (`app/engine/board.py`) already stores state as two native bitboards
(`board.cur` -- the SIDE-TO-MOVE's stones, `board.mask` -- all occupied
cells; see that module's docstring). This module skips the grid entirely and
classifies each window directly from the two bitboards:

  * 69 window bitmasks (SAME 4-cell windows, SAME iteration order as
    `heuristic_eval.py`'s `_WINDOWS` list) precomputed once at import via
    `bit(col, row) = col*H1 + row` (`app.engine.board.H1`, matching
    `Board.cells()`'s own bit-test convention exactly).
  * Per window: `mine = cur & w`, `theirs = opp & w`, `mc = mine.bit_count()`,
    `tc = theirs.bit_count()`, then ONE `_SCORE_TABLE[mc][tc]` lookup. The
    5x5 `_SCORE_TABLE` is built ONCE at import by calling
    `heuristic_eval._score_window` itself (imported, not re-derived by hand)
    on a synthetic `mc`-ones/`tc`-twos/rest-zeros list for every
    `(mc, tc)` with `mc+tc<=4` -- this makes the two modules' scoring rule
    provably the SAME function (see
    `tests/test_heuristic_eval_bb_equivalence.py`), not a hand-copied
    lookalike that could silently drift.
  * Center-column bonus: same `and`+`bit_count` pattern against a
    precomputed column bitmask, `+CENTER_BONUS`/`-CENTER_BONUS` per stone,
    matching `heuristic_eval.evaluate()`'s hardcoded `+6`/`-6` (imported
    from nowhere since it's a bare literal there too; `CENTER_BONUS = 6`
    here is the same value, unit-tested for equality below, not just
    ranking-equality).

Every individual primitive (`&`, `^`, `.bit_count()`, and the table lookup)
is routed through a thin module-level wrapper function (`_and`, `_xor`,
`_popcount`, `_table_lookup`, `_center_combine`) purely so
`tests/test_net14_flop_honesty.py` can machine-count the REAL number of
primitive calls one `evaluate_bb()` invocation performs (mirrors how that
test already wraps `Board.winner`/`Board.play`/etc. -- see that file's
module docstring) instead of hand-waving a flat per-call constant the way
the OLD `heuristic_evaluate` wrap necessarily did (it wasn't bit-op-shaped,
so there was nothing finer to instrument). Costed under the SAME
atomic-operator convention `app/solver/solver.py`'s
`_compute_winning_position` is honestly priced under in that test file
(`COMPUTE_WINNING_POSITION_OPS` -- each `<<`/`&`/`|`/`^` = 1 unit): `_and`
and `_xor` = 1 unit each. `_popcount` (`int.bit_count()`, a single builtin
call with no Python-level branching) and each list-index read inside
`_table_lookup` are priced under this same codebase's OTHER established
convention (`heuristic_eval.py`'s own docstring: "bit ops, arithmetic,
comparisons, and list-index reads all counted at 1 unit each") -- so
`_popcount` = 1 unit, `_table_lookup` (two chained index reads,

`table[mc][tc]`) = 2 units, `_center_combine` (two multiplies + one
subtract) = 3 units. No operation here is priced any lower than the
`heuristic_eval.py`/`solver.py` conventions this codebase already uses
elsewhere for a comparable primitive; see
`tests/test_net14_flop_honesty.py` for the exact weights and the
machine-checked sum.
"""
from __future__ import annotations

from app.agents.heuristic_eval import _score_window
from app.engine.board import WIDTH, HEIGHT, H1

CENTER_BONUS = 6   # matches heuristic_eval.evaluate()'s hardcoded +6/-6


def _and(a: int, b: int) -> int:
    return a & b


def _xor(a: int, b: int) -> int:
    return a ^ b


def _popcount(x: int) -> int:
    return x.bit_count()


def _table_lookup(table, mc: int, tc: int) -> int:
    return table[mc][tc]


def _center_combine(my_center: int, opp_center: int) -> int:
    return CENTER_BONUS * my_center - CENTER_BONUS * opp_center


def _bit(col: int, row: int) -> int:
    return 1 << (col * H1 + row)


def _build_windows():
    """SAME window geometry as heuristic_eval.py's `_WINDOWS` (same nested
    loop, same append order) so `_WINDOW_MASKS[i]` corresponds 1:1 to
    `heuristic_eval._WINDOWS[i]`."""
    windows = []
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if c + 3 < WIDTH:
                windows.append([(r, c + i) for i in range(4)])          # horizontal
            if r + 3 < HEIGHT:
                windows.append([(r + i, c) for i in range(4)])          # vertical
            if c + 3 < WIDTH and r + 3 < HEIGHT:
                windows.append([(r + i, c + i) for i in range(4)])      # diag "/"
            if c - 3 >= 0 and r + 3 < HEIGHT:
                windows.append([(r + i, c - i) for i in range(4)])      # diag "\"
    return windows


_WINDOW_CELLS = _build_windows()
_WINDOW_MASKS = tuple(
    sum(_bit(c, r) for (r, c) in win) for win in _WINDOW_CELLS
)
assert len(_WINDOW_MASKS) == 69, len(_WINDOW_MASKS)

_CENTER = WIDTH // 2
_CENTER_MASK = sum(_bit(_CENTER, r) for r in range(HEIGHT))


def _build_score_table():
    """5x5 table[mc][tc] built by calling heuristic_eval.py's OWN
    `_score_window` (imported, never re-derived) on a synthetic
    mc-ones/tc-twos/rest-zeros 4-cell list with me=1, opp=2 -- guarantees
    byte-for-byte equivalence with the original bucket rule, not a
    hand-copied lookalike."""
    table = [[0] * 5 for _ in range(5)]
    for mc in range(5):
        for tc in range(5):
            if mc + tc <= 4:
                vals = [1] * mc + [2] * tc + [0] * (4 - mc - tc)
                table[mc][tc] = _score_window(vals, 1, 2)
    return table


_SCORE_TABLE = _build_score_table()


def evaluate_bb(board) -> int:
    """Static score from the side-to-move's perspective. `board.cur` is
    ALWAYS the side-to-move's stones (see `app/engine/board.py`'s module
    docstring), so no `player_to_move()`/grid lookup is needed at all --
    `cur` IS "mine", `mask ^ cur` IS "theirs"."""
    cur = board.cur
    opp = _xor(board.mask, board.cur)
    score = 0
    for w in _WINDOW_MASKS:
        mine = _popcount(_and(cur, w))
        theirs = _popcount(_and(opp, w))
        score += _table_lookup(_SCORE_TABLE, mine, theirs)
    my_center = _popcount(_and(cur, _CENTER_MASK))
    opp_center = _popcount(_and(opp, _CENTER_MASK))
    score += _center_combine(my_center, opp_center)
    return score
