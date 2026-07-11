"""Shared board -> feature encoder used by BOTH the trainer and inference agents.

The encoding is *perspective-normalised*: plane A is always the side-to-move's
stones, plane B the opponent's. This makes learned agents colour-agnostic.

Feature layout (float32):
    [0:42]    my stones     (row-major, row 0 = bottom)   {0,1}
    [42:84]   opponent stones                             {0,1}
    [84:126]  my completable-4 threat cells (plane)       {0,1}
    [126:168] opp completable-4 threat cells (plane)      {0,1}
    [168:175] "I win immediately by playing column c"      {0,1}
    [175:182] "opponent wins next in column c" (must-block) {0,1}
    [182:189] "column c is legal" (not full)               {0,1}
    [189]     side-to-move is the FIRST player (odd/even parity seat)  {0,1}
    [190:194] my_odd, my_even, opp_odd, opp_even threat-cell counts /6 (float)

The parity block is the classic Connect-4 zugzwang signal (odd vs even row
threats favour a specific seat once the board fills up); it is a cheap,
board-only aggregate, computed from the same threat bitboards already used
for the per-column immediate-win/must-block flags above.

All features are computed from the Board alone -- no solver, no labels.
"""
from __future__ import annotations

import numpy as np

from app.engine.board import WIDTH, HEIGHT, H1
from app.solver.solver import _compute_winning_position, _possible

# 84 disc planes + 84 threat planes (my/opp completable-4 cells) + 21 column
# flags + 1 first-player flag + 4 odd/even threat-count aggregates
FEATURE_DIM = 4 * WIDTH * HEIGHT + 3 * WIDTH + 1 + 4    # 189 + 5 = 194


def _odd_even_counts(bits: int) -> tuple[int, int]:
    """(odd_row_count, even_row_count) of set cells in a threat bitboard.
    Row 0 (bottom) is conventionally "odd" (1st row), row index r -> odd iff
    r is even (0-indexed)."""
    odd = 0
    even = 0
    for c in range(WIDTH):
        for r in range(HEIGHT):
            if bits & (1 << (c * H1 + r)):
                if r % 2 == 0:
                    odd += 1
                else:
                    even += 1
    return odd, even


def _plane_from_bits(vec, offset, bits):
    idx = offset
    for c in range(WIDTH):
        for r in range(HEIGHT):
            if bits & (1 << (c * H1 + r)):
                vec[idx] = 1.0
            idx += 1


def _col_flags_from_bits(bits: int) -> list[int]:
    """Reduce a bitboard of 'winning empty cells' to per-column presence flags."""
    flags = [0] * WIDTH
    for c in range(WIDTH):
        col_mask = (((1 << HEIGHT) - 1) << (c * H1))
        if bits & col_mask:
            flags[c] = 1
    return flags


def encode(board) -> np.ndarray:
    cur, mask, n = board.cur, board.mask, board.n
    opp = cur ^ mask

    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    # planes (perspective: cur = me)
    idx = 0
    for c in range(WIDTH):
        for r in range(HEIGHT):
            bit = 1 << (c * H1 + r)
            if cur & bit:
                vec[idx] = 1.0
            idx += 1
    for c in range(WIDTH):
        for r in range(HEIGHT):
            bit = 1 << (c * H1 + r)
            if opp & bit:
                vec[idx] = 1.0
            idx += 1

    # full threat structure (all empty cells that would complete a 4), as planes
    my_threats = _compute_winning_position(cur, mask)
    opp_threats = _compute_winning_position(opp, mask)
    _plane_from_bits(vec, 84, my_threats)
    _plane_from_bits(vec, 126, opp_threats)

    possible = _possible(mask)
    my_wins = my_threats & possible          # immediately playable win
    opp_wins = opp_threats & possible        # must-block
    base = 168
    for c, f in enumerate(_col_flags_from_bits(my_wins)):
        vec[base + c] = f
    for c, f in enumerate(_col_flags_from_bits(opp_wins)):
        vec[base + WIDTH + c] = f
    # legal-column flags
    for c in range(WIDTH):
        top = 1 << (c * H1 + HEIGHT - 1)
        if (mask & top) == 0:
            vec[base + 2 * WIDTH + c] = 1.0

    # zugzwang parity block (cheap aggregate, no extra board scans: reuses
    # the my_threats/opp_threats bitboards already computed above)
    p = base + 3 * WIDTH        # = 189
    vec[p] = 1.0 if board.player_to_move() == 1 else 0.0
    my_odd, my_even = _odd_even_counts(my_threats)
    op_odd, op_even = _odd_even_counts(opp_threats)
    vec[p + 1] = my_odd / float(HEIGHT)
    vec[p + 2] = my_even / float(HEIGHT)
    vec[p + 3] = op_odd / float(HEIGHT)
    vec[p + 4] = op_even / float(HEIGHT)

    return vec


def legal_mask(board) -> np.ndarray:
    """Boolean length-7 mask of legal columns."""
    m = np.zeros(WIDTH, dtype=bool)
    for c in board.legal_moves():
        m[c] = True
    return m
