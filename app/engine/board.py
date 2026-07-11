"""Bitboard Connect 4 board.

Layout: 7 columns x 6 rows, with a 1-bit sentinel row on top of every column,
so each column occupies H1 = HEIGHT + 1 = 7 bits.

    bit(col, row) = col * H1 + row      (row 0 = bottom, row 5 = top playable)

The sentinel row (row 6) is always empty; it stops win-detection bit shifts from
wrapping between adjacent columns.

Internal state is two integers:
    cur  -- bitboard of the stones belonging to the player *to move*
    mask -- bitboard of all occupied cells
plus n, the number of stones played (0..42). The player to move is deducible from
n (1 when n is even, 2 when n is odd), but we keep it explicit-derivable.

`to_key()` returns a canonical, mirror-normalised string used by the transposition
table and by the benchmark de-duplication. Internal (cur, mask) are always the
*exact* position; only to_key() applies horizontal-mirror normalisation.
"""
from __future__ import annotations

WIDTH = 7
HEIGHT = 6
H1 = HEIGHT + 1          # 7 bits per column (incl. sentinel)
H2 = HEIGHT + 2          # 8
SIZE = WIDTH * HEIGHT    # 42 playable cells

# center-first column ordering for move ordering in the solver
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)

# --- precomputed masks -------------------------------------------------------
def _bottom_mask(col: int) -> int:
    return 1 << (col * H1)

def _top_mask(col: int) -> int:
    return 1 << (col * H1 + HEIGHT - 1)   # topmost *playable* cell (row 5)

def _column_mask(col: int) -> int:
    return ((1 << HEIGHT) - 1) << (col * H1)

# full bottom row (one bit per column at row 0) -- used for the fill trick
BOTTOM = 0
for _c in range(WIDTH):
    BOTTOM |= _bottom_mask(_c)
BOARD_MASK = BOTTOM * ((1 << HEIGHT) - 1)   # all 42 playable bits set


def _won(pos: int) -> bool:
    """True if the bitboard `pos` contains any 4-in-a-row."""
    # horizontal (shift by H1)
    m = pos & (pos >> H1)
    if m & (m >> (2 * H1)):
        return True
    # diagonal "\"  (shift by HEIGHT)
    m = pos & (pos >> HEIGHT)
    if m & (m >> (2 * HEIGHT)):
        return True
    # diagonal "/"  (shift by H2)
    m = pos & (pos >> H2)
    if m & (m >> (2 * H2)):
        return True
    # vertical (shift by 1)
    m = pos & (pos >> 1)
    if m & (m >> 2):
        return True
    return False


def _mirror(bb: int) -> int:
    """Reflect a bitboard horizontally (column c -> column WIDTH-1-c)."""
    out = 0
    for c in range(WIDTH):
        col_bits = (bb >> (c * H1)) & ((1 << H1) - 1)
        out |= col_bits << ((WIDTH - 1 - c) * H1)
    return out


class IllegalMove(ValueError):
    """Raised when attempting to play into a full or out-of-range column."""


class Board:
    __slots__ = ("cur", "mask", "n")

    def __init__(self, cur: int = 0, mask: int = 0, n: int = 0):
        self.cur = cur
        self.mask = mask
        self.n = n

    # ---- constructors -------------------------------------------------------
    @classmethod
    def empty(cls) -> "Board":
        return cls(0, 0, 0)

    @classmethod
    def from_moves(cls, seq) -> "Board":
        """Build from an iterable of column indices.

        ESCAPE 3 root-cause fix: a move sequence must not be applicable past
        a terminal position. SPEC.md sec.1 defines the game as ending the
        instant a player "wins by making 4 in a row" (or the board fills for
        a draw) -- there is no such thing as a legal move played against an
        already-decided game, so any columns in `seq` *after* the position
        first becomes terminal are not part of "Standard Connect 4" and are
        ignored. We choose "stop at the first terminal position and report
        that winner" (not "reject the whole sequence as illegal input")
        because this is the single canonical constructor used everywhere a
        Board is built from a move sequence (the API's `/analyze`, the bench
        position loaders, every agent's internal search) -- it must be
        impossible, BY CONSTRUCTION, for any of them to ever observe a
        corrupted board where more than one player appears to have won.
        Rejecting outright would push a try/except onto every one of those
        call sites for a case none of them can legitimately produce (self-
        play/ladder/position-generation code already stops at the first
        `is_terminal()` board, so this is a strict no-op for every
        legitimate sequence and only changes behaviour for malformed,
        already-decided-but-continued input like the auditor's repro).
        """
        b = cls.empty()
        if isinstance(seq, str):
            seq = [int(ch) for ch in seq.replace(",", " ").split()] if seq.strip() else []
        for col in seq:
            if b.is_terminal():
                break
            b = b.play(int(col))
        return b

    @classmethod
    def from_key(cls, key: str) -> "Board":
        """Rebuild a board from a to_key() string 'mask:cur'.

        The result is the canonical (mirror-normalised) position; it is equal
        (same to_key) to the board that produced the key.
        """
        mask_s, cur_s = key.split(":")
        mask = int(mask_s)
        cur = int(cur_s)
        n = bin(mask).count("1")
        return cls(cur, mask, n)

    # ---- basic queries ------------------------------------------------------
    def player_to_move(self) -> int:
        return 1 if (self.n & 1) == 0 else 2

    def _p1(self) -> int:
        """Bitboard of player 1's stones."""
        return self.cur if (self.n & 1) == 0 else (self.mask ^ self.cur)

    def _p2(self) -> int:
        return (self.mask ^ self.cur) if (self.n & 1) == 0 else self.cur

    def legal_moves(self) -> list[int]:
        return [c for c in range(WIDTH) if (self.mask & _top_mask(c)) == 0]

    def can_play(self, col: int) -> bool:
        return 0 <= col < WIDTH and (self.mask & _top_mask(col)) == 0

    def is_full(self) -> bool:
        return self.n >= SIZE

    def play(self, col: int) -> "Board":
        """Return the position after dropping a disc in `col`. Does not mutate self."""
        if not (0 <= col < WIDTH):
            raise IllegalMove(f"column {col} out of range 0..{WIDTH-1}")
        if (self.mask & _top_mask(col)) != 0:
            raise IllegalMove(f"column {col} is full")
        new_mask = self.mask | (self.mask + _bottom_mask(col))
        # after flipping, `cur` becomes the opponent's stones (the new side to move);
        # the just-placed stone stays with the previous mover (not in new cur).
        new_cur = self.cur ^ self.mask
        return Board(new_cur, new_mask, self.n + 1)

    def winning_move(self, col: int) -> bool:
        """True if playing `col` gives the side-to-move an immediate 4-in-a-row."""
        if not self.can_play(col):
            return False
        move_bit = (self.mask + _bottom_mask(col)) & _column_mask(col)
        return _won(self.cur | move_bit)

    # ---- terminal detection -------------------------------------------------
    def winner(self) -> int:
        """0 = no winner (in progress or draw), 1 or 2 = winning player."""
        # only the last mover can have completed a line; check both defensively
        last_mover_board = self.mask ^ self.cur
        if _won(last_mover_board):
            # the last mover is the opponent of the side to move
            return 2 if self.player_to_move() == 1 else 1
        if _won(self.cur):
            return self.player_to_move()
        return 0

    def is_draw(self) -> bool:
        return self.n >= SIZE and self.winner() == 0

    def is_terminal(self) -> bool:
        return self.winner() != 0 or self.n >= SIZE

    # ---- canonical key / hashing -------------------------------------------
    def _pair(self) -> tuple[int, int]:
        return (self.mask, self.cur)

    def _mirror_pair(self) -> tuple[int, int]:
        return (_mirror(self.mask), _mirror(self.cur))

    def to_key(self) -> str:
        """Canonical mirror-normalised key: 'mask:cur' of the smaller reflection."""
        a = self._pair()
        b = self._mirror_pair()
        m, c = a if a <= b else b
        return f"{m}:{c}"

    def __hash__(self) -> int:
        return hash(self.to_key())

    def __eq__(self, other) -> bool:
        return isinstance(other, Board) and self.to_key() == other.to_key()

    # ---- rendering ----------------------------------------------------------
    def cells(self) -> list[list[int]]:
        """6x7 grid, cells[row][col]; row 0 = bottom. Values 0/1/2."""
        p1, p2 = self._p1(), self._p2()
        grid = [[0] * WIDTH for _ in range(HEIGHT)]
        for c in range(WIDTH):
            for r in range(HEIGHT):
                bit = 1 << (c * H1 + r)
                if p1 & bit:
                    grid[r][c] = 1
                elif p2 & bit:
                    grid[r][c] = 2
        return grid

    def moves_remaining(self) -> int:
        return SIZE - self.n

    def __str__(self) -> str:
        grid = self.cells()
        sym = {0: ".", 1: "X", 2: "O"}
        rows = ["|" + " ".join(sym[grid[r][c]] for c in range(WIDTH)) + "|"
                for r in reversed(range(HEIGHT))]
        rows.append(" " + " ".join(str(c) for c in range(WIDTH)) + " ")
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"Board(n={self.n}, to_move={self.player_to_move()}, key={self.to_key()})"
