"""`encode_v2`: an EXTENDED board -> feature encoder for `neurofour-net2` only.

`app.agents.encode.encode`/`FEATURE_DIM` (the "v1" encoder) stays completely
unchanged and is still what `neurofour-net1` uses -- this module never imports
`app.agents.net1` and never mutates anything in `encode.py`. This lets net1's
artifact + encoder combination stay byte-for-byte reproducible while net2 gets
its own (possibly larger) feature vector and its own dedicated artifact.

`encode_v2(board, blocks=...)` = the full v1 194-dim vector, UNCHANGED, with
extra engineered TACTICAL feature blocks appended in a fixed canonical order
(only the blocks named in `blocks` are appended, so callers can ablate). Every
new feature is computed from the Board ALONE via bitboard pattern ops (the
same style as `encode.py`'s `_compute_winning_position`/`_possible` calls and
`board.py`'s own `_won`/`_mirror` -- board-structure bit-twiddling, never the
game-tree solver). No feature reads a label, a `.jsonl` file, or a solved
value at inference time.

New blocks (canonical order -- always appended in this order, subset chosen
by `blocks`):

  "fork"    (7 dims)  per-column flag: does playing THIS column create a
             position where I have >=2 distinct columns with an immediately
             playable winning cell (an unstoppable double threat)? Not
             expressible from the v1 encoding, which only has single-ply
             immediate-win/must-block flags for the CURRENT position, not
             "what happens after I play here".

  "stacked" (14 dims = 7+7) classic Connect-4 stacked-threat motif: a
             completable-4 cell for me directly ABOVE a completable-4 cell
             for the opponent in the same column (opponent is deterred from
             filling their own threat, since doing so hands me the winning
             cell right above it) -- and the symmetric opponent-over-me case.
             Computed with a single bit-shift-by-1 (same-column row+1, exactly
             like `board.py`'s own row-adjacency idiom) + AND, then reduced to
             per-column flags with the same `_col_flags_from_bits` helper v1
             already uses for its immediate-win/must-block planes.

  "parity"  (4 dims) refines the existing raw odd/even THREAT-CELL COUNTS
             (v1 feature [190:194]) with an OWNERSHIP-aware claimeven/
             baseinverse-style signal: for each column, find the LOWEST
             (nearest-to-playable) completable-4 cell of EITHER player and
             classify it by (owner, row-parity). Reports 4 column-fractions:
             my-lowest-is-odd, opp-lowest-is-even (both classically
             *favourable* zugzwang patterns), my-lowest-is-even,
             opp-lowest-is-odd (both classically *unfavourable*). This is a
             genuinely different signal from the raw counts: two positions
             can have identical odd/even threat COUNTS while differing in
             which player's threat sits lowest (i.e. resolves first as the
             column fills), which is what actually decides the zugzwang.

  "dblock"  (1 dim) explicit flag: opponent already has >=2 distinct
             immediate-winning columns (an already-lost position -- at most
             one can be blocked). Derived directly from v1's own must-block
             column flags (v1 feature slice [175:182]), just re-expressed as
             a single aggregate signal instead of 7 raw per-column bits.

All blocks are pure functions of `(cur, mask)` (plus the two threat
bitboards already needed to build them); none call `Board.play` more than the
`WIDTH` times structurally required for the "fork" block's per-column
what-if.
"""
from __future__ import annotations

import numpy as np

from app.engine.board import WIDTH, HEIGHT, H1, _bottom_mask, _column_mask
from app.solver.solver import _compute_winning_position, _possible
from app.agents.encode import (
    encode as encode_v1,
    FEATURE_DIM as FEATURE_DIM_V1,
    _col_flags_from_bits,
)

_BLOCK_SIZES = {"fork": WIDTH, "stacked": 2 * WIDTH, "parity": 4, "dblock": 1}
_CANONICAL_ORDER = ("fork", "stacked", "parity", "dblock")

ALL_BLOCKS = _CANONICAL_ORDER


def feature_dim_v2(blocks=ALL_BLOCKS) -> int:
    return FEATURE_DIM_V1 + sum(_BLOCK_SIZES[b] for b in blocks if b in _BLOCK_SIZES)


def _fork_flags(cur: int, mask: int) -> list[int]:
    """Per column c: 1 iff playing c gives ME >=2 distinct columns with an
    immediately-playable winning cell (an unstoppable fork). Board-only:
    simulates the single drop with the same fill-trick bit op `net1.py`'s
    `tactical_move`/`board.py`'s `winning_move` already use, then re-uses
    `_compute_winning_position`/`_possible` (already used by v1) on the
    resulting (cur|move_bit, mask|move_bit)."""
    flags = [0] * WIDTH
    top_bits = [1 << (c * H1 + HEIGHT - 1) for c in range(WIDTH)]
    for c in range(WIDTH):
        if mask & top_bits[c]:
            continue    # column c is full -> can't play there
        move_bit = (mask + _bottom_mask(c)) & _column_mask(c)
        new_cur = cur | move_bit
        new_mask = mask | move_bit
        new_threats = _compute_winning_position(new_cur, new_mask)
        win_now = new_threats & _possible(new_mask)
        if sum(_col_flags_from_bits(win_now)) >= 2:
            flags[c] = 1
    return flags


def _stacked_flags(my_threats: int, opp_threats: int):
    """(my_over_opp_col_flags, opp_over_my_col_flags): per-column flags for the
    stacked-threat motif. Shifting the whole bitboard by 1 moves every set bit
    from row r to row r+1 WITHIN THE SAME COLUMN (H1=HEIGHT+1 bits/column with
    an always-empty sentinel row -- the identical idiom `board.py` relies on
    for `_won`'s vertical check); any bleed into a neighbouring column's
    always-empty sentinel bit is masked out for free by the AND below, since
    threat bitboards are never set on sentinel bits."""
    stacked_my_over_opp = opp_threats & (my_threats >> 1)
    stacked_opp_over_my = my_threats & (opp_threats >> 1)
    return (_col_flags_from_bits(stacked_my_over_opp),
            _col_flags_from_bits(stacked_opp_over_my))


def _low_threat_parity_fracs(my_threats: int, opp_threats: int):
    """4 column-fractions: (my_lowest_is_odd, opp_lowest_is_even,
    my_lowest_is_even, opp_lowest_is_odd), where "lowest" = the completable-4
    cell nearest the current stack top in that column (whichever player owns
    it), classified by row parity (row 0 = "odd", matching `encode.py`'s own
    convention). This is an ownership+parity refinement of the raw odd/even
    threat COUNTS already in v1 ([190:194])."""
    my_odd = my_even = opp_odd = opp_even = 0
    for c in range(WIDTH):
        col_mask = ((1 << HEIGHT) - 1) << (c * H1)
        combined = (my_threats | opp_threats) & col_mask
        if combined == 0:
            continue
        lowest = combined & (-combined)          # isolate lowest set bit
        row = (lowest.bit_length() - 1) - c * H1
        if lowest & my_threats:
            if row % 2 == 0:
                my_odd += 1
            else:
                my_even += 1
        else:
            if row % 2 == 0:
                opp_odd += 1
            else:
                opp_even += 1
    w = float(WIDTH)
    return my_odd / w, opp_even / w, my_even / w, opp_odd / w


def encode_v2(board, blocks=ALL_BLOCKS) -> np.ndarray:
    v1 = encode_v1(board)
    fdim = feature_dim_v2(blocks)
    vec = np.zeros(fdim, dtype=np.float32)
    vec[:FEATURE_DIM_V1] = v1

    cur, mask = board.cur, board.mask
    opp = cur ^ mask
    need_threats = ("stacked" in blocks) or ("parity" in blocks)
    my_threats = _compute_winning_position(cur, mask) if need_threats else 0
    opp_threats = _compute_winning_position(opp, mask) if need_threats else 0

    idx = FEATURE_DIM_V1
    for block in _CANONICAL_ORDER:
        if block not in blocks:
            continue
        if block == "fork":
            for f in _fork_flags(cur, mask):
                vec[idx] = f
                idx += 1
        elif block == "stacked":
            my_over, opp_over = _stacked_flags(my_threats, opp_threats)
            for f in my_over:
                vec[idx] = f
                idx += 1
            for f in opp_over:
                vec[idx] = f
                idx += 1
        elif block == "parity":
            for f in _low_threat_parity_fracs(my_threats, opp_threats):
                vec[idx] = f
                idx += 1
        elif block == "dblock":
            # v1[175:182] = opp must-block per-column flags (see encode.py's
            # own FEATURE_DIM layout docstring); already-lost iff >=2 set.
            vec[idx] = 1.0 if float(v1[175:182].sum()) >= 2.0 else 0.0
            idx += 1
    assert idx == fdim
    return vec
