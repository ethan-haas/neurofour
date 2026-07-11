import random

import pytest
from hypothesis import given, settings, strategies as st

from app.engine.board import Board, WIDTH, HEIGHT, IllegalMove


def test_gravity():
    b = Board.empty().play(3)
    assert b.cells()[0][3] == 1
    b = b.play(3)
    assert b.cells()[1][3] == 2
    assert b.player_to_move() == 1


def test_horizontal_win():
    b = Board.from_moves([0, 6, 1, 6, 2, 5, 3])
    assert b.winner() == 1
    assert b.is_terminal()


def test_horizontal_win_right_edge():
    b = Board.from_moves([3, 3, 4, 4, 5, 5, 6])
    assert b.winner() == 1


def test_vertical_win():
    b = Board.from_moves([3, 4, 3, 4, 3, 4, 3])
    assert b.winner() == 1


def test_vertical_win_edge_column():
    b = Board.from_moves([0, 1, 0, 1, 0, 1, 0])
    assert b.winner() == 1


def test_diagonal_up_right_win():
    b = Board.from_moves([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3])
    assert b.winner() == 1


def test_diagonal_up_left_win():
    # mirror of the up-right win
    b = Board.from_moves([6, 5, 5, 4, 4, 3, 4, 3, 3, 0, 3])
    assert b.winner() == 1


def test_no_false_win_across_columns():
    # four discs in row 0 of adjacent columns of *different* players must not win
    b = Board.from_moves([0, 1, 2, 3])
    assert b.winner() == 0


def test_illegal_full_column():
    b = Board.from_moves([3, 3, 3, 3, 3, 3])
    assert not b.can_play(3)
    with pytest.raises(IllegalMove):
        b.play(3)


def test_illegal_out_of_range():
    b = Board.empty()
    with pytest.raises(IllegalMove):
        b.play(7)
    with pytest.raises(IllegalMove):
        b.play(-1)


def test_legal_moves_shrink_when_column_fills():
    b = Board.from_moves([0, 0, 0, 0, 0, 0])
    assert 0 not in b.legal_moves()
    assert set(b.legal_moves()) == {1, 2, 3, 4, 5, 6}


def test_draw_full_board_no_winner():
    # search a random full game that ends in a draw
    for attempt in range(20000):
        rng = random.Random(attempt)
        b = Board.empty()
        while not b.is_terminal():
            b = b.play(rng.choice(b.legal_moves()))
        if b.n == WIDTH * HEIGHT and b.winner() == 0:
            assert b.is_draw()
            assert b.is_terminal()
            return
    pytest.fail("no draw found (unexpected)")


def test_mirror_canonical_key():
    a = Board.from_moves([0, 1, 0])
    m = Board.from_moves([6, 5, 6])
    assert a.to_key() == m.to_key()
    assert a == m
    assert hash(a) == hash(m)


def test_from_key_roundtrip():
    b = Board.from_moves([3, 2, 4, 3, 1])
    k = b.to_key()
    assert Board.from_key(k).to_key() == k


def test_player_to_move_alternation():
    b = Board.empty()
    for i in range(6):
        assert b.player_to_move() == (1 if i % 2 == 0 else 2)
        b = b.play(i % WIDTH)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=6), max_size=42))
def test_play_never_exceeds_capacity(moves):
    b = Board.empty()
    for c in moves:
        if b.is_terminal() or not b.can_play(c):
            break
        b = b.play(c)
    grid = b.cells()
    # column height consistency
    for col in range(WIDTH):
        heights = [grid[r][col] != 0 for r in range(HEIGHT)]
        # no gaps: a filled cell cannot sit above an empty one
        seen_empty = False
        for r in range(HEIGHT):
            if grid[r][col] == 0:
                seen_empty = True
            elif seen_empty:
                pytest.fail("floating disc (gravity violated)")
    assert 0 <= b.n <= 42


# ---- ESCAPE 3 regression: a move sequence must not be applicable past a --
# ---- terminal position (SPEC sec.1 -- see Board.from_moves docstring). ---
def test_from_moves_stops_applying_illegal_post_terminal_moves():
    # The auditor's exact repro: P1 completes a bottom-row horizontal four
    # (cols 0-3) at move 19 (0-indexed move 18), but the sequence keeps
    # going for 5 more (illegal, post-terminal) moves, reaching a board
    # where -- pre-fix -- BOTH players appeared to have a four-in-a-row and
    # `winner()` reported the wrong player (2, not the actual winner 1).
    full = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    truncated = full[:19]  # stop right at the winning move
    b_full = Board.from_moves(full)
    b_trunc = Board.from_moves(truncated)
    # constructing from the full (corrupted) sequence must land on the exact
    # same board as constructing from the truncated-at-terminal sequence --
    # every column after move 19 is a no-op.
    assert b_full.to_key() == b_trunc.to_key()
    assert b_full.n == b_trunc.n == 19
    assert b_full.is_terminal()
    assert b_full.winner() == 1


def test_from_moves_terminal_draw_also_clamps():
    # a full-board draw sequence with extra trailing columns appended must
    # likewise stop the instant the board fills, not error or misreport.
    draw_seq = [0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0,
                2, 3, 2, 3, 2, 3,
                3, 2, 3, 2, 3, 2,
                4, 5, 4, 5, 4, 5,
                5, 4, 5, 4, 5, 4,
                6, 6, 6, 6, 6, 6]
    b = Board.from_moves(draw_seq)
    assert b.is_draw()
    padded = Board.from_moves(draw_seq + [0, 1, 2, 3])  # extra post-terminal noise
    assert padded.to_key() == b.to_key()


@settings(max_examples=300, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=6), max_size=42),
       st.lists(st.integers(min_value=0, max_value=6), max_size=20))
def test_from_moves_never_yields_a_board_with_two_winners(prefix, suffix):
    """Property: no board built from ANY move sequence -- however many
    illegal/post-terminal columns are appended -- can ever have both
    players simultaneously holding a 4-in-a-row. `Board.from_moves`
    stopping at the first terminal position is what guarantees this.

    `prefix` is trimmed down to a genuinely legal, non-terminal opening (no
    caller -- API, bench loader, or agent -- ever legitimately plays into an
    already-full, still-in-progress column; that is a separate, pre-
    existing `IllegalMove` case unrelated to ESCAPE 3). `suffix` is then
    spliced on unfiltered to fuzz arbitrary post-terminal continuations,
    exactly the shape of the auditor's repro. If a suffix column happens to
    hit a full-but-not-yet-terminal column (the unrelated pre-existing
    IllegalMove case), the example is discarded via `assume`."""
    from hypothesis import assume
    from app.engine.board import _won, IllegalMove

    legal_prefix = []
    b = Board.empty()
    for c in prefix:
        if b.is_terminal() or not b.can_play(c):
            break
        legal_prefix.append(c)
        b = b.play(c)

    full_seq = legal_prefix + suffix
    try:
        b = Board.from_moves(full_seq)
    except IllegalMove:
        assume(False)  # unrelated pre-terminal full-column case; not ESCAPE 3
        return

    p1_won = _won(b._p1())
    p2_won = _won(b._p2())
    assert not (p1_won and p2_won), "board has two simultaneous winners"
    # winner() must agree with whichever single player (if any) actually won
    w = b.winner()
    assert w in (0, 1, 2)
    if p1_won:
        assert w == 1
    if p2_won:
        assert w == 2
    if not p1_won and not p2_won:
        assert w == 0
