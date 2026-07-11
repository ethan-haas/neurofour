"""Exact negamax + alpha-beta + transposition table Connect 4 solver.

Value convention (scored mode), from the perspective of the side to move:
    * a win by placing stone number k (1-indexed over all 42 cells) scores
      ``SIZE - (k - 1) = SIZE - k + 1`` -- i.e. (remaining cells after the win) + 1.
      The earliest possible win (k = 7) scores the most (36); a later win scores
      less, so a mate-in-fewer is strictly preferred.
    * a loss scores the negation of the opponent's winning score.
    * a draw scores 0.

The value-only convention (win/draw/loss = +1/0/-1) is exactly ``sign(scored)``,
because every win is strictly positive, every loss strictly negative, draw 0.

Public API:
    solve(board, mode="scored"|"value") -> Solved(value, optimal_cols, best_col, per_col)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from app.engine.board import (
    WIDTH, HEIGHT, H1, H2, SIZE, BOTTOM, BOARD_MASK, CENTER_ORDER, _bottom_mask,
)

MIN_SCORE = -SIZE
MAX_SCORE = SIZE

# center-preference rank for deterministic best_col tie-breaking
_CENTER_RANK = {c: i for i, c in enumerate(CENTER_ORDER)}

# committed opening book (value/best_col for near-empty positions the live solver
# cannot reach quickly in pure Python). Loaded lazily, once per process.
_BOOK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "opening_book.json")
_BOOK = None


def _load_book() -> dict:
    global _BOOK
    if _BOOK is None:
        try:
            with open(_BOOK_PATH, "r", encoding="utf-8") as f:
                _BOOK = json.load(f)
        except (FileNotFoundError, ValueError):
            _BOOK = {}
    return _BOOK


def _book_lookup(board, mode: str):
    """Return a Solved from the opening book (value mode only), or None."""
    if mode != "value":
        return None
    entry = _load_book().get(board.to_key())
    if entry is None:
        return None
    value = entry["value"]
    optimal = entry["optimal_cols"]
    best_col = entry["best_col"]
    legal = board.legal_moves()
    if "per_col" in entry:
        per = {int(k): v for k, v in entry["per_col"].items()}
    else:
        # reconstruct: optimal cols achieve `value`; others are (approximately) worse
        per = {c: (value if c in optimal else min(value, 0)) for c in legal}
    return Solved(value, sorted(optimal), best_col, per, mode="value")


# --------------------------------------------------------------------------- #
# int-level bit helpers (operate directly on (cur, mask) for speed)           #
# --------------------------------------------------------------------------- #
def _possible(mask: int) -> int:
    """Bitboard of the next playable cell in every non-full column."""
    return (mask + BOTTOM) & BOARD_MASK


def _compute_winning_position(position: int, mask: int) -> int:
    """All empty cells that would complete a 4-in-a-row for `position`'s owner."""
    # vertical
    r = (position << 1) & (position << 2) & (position << 3)

    # horizontal (shift H1)
    p = (position << H1) & (position << (2 * H1))
    r |= p & (position << (3 * H1))
    r |= p & (position >> H1)
    p = (position >> H1) & (position >> (2 * H1))
    r |= p & (position << H1)
    r |= p & (position >> (3 * H1))

    # diagonal "\" (shift HEIGHT)
    p = (position << HEIGHT) & (position << (2 * HEIGHT))
    r |= p & (position << (3 * HEIGHT))
    r |= p & (position >> HEIGHT)
    p = (position >> HEIGHT) & (position >> (2 * HEIGHT))
    r |= p & (position << HEIGHT)
    r |= p & (position >> (3 * HEIGHT))

    # diagonal "/" (shift H2)
    p = (position << H2) & (position << (2 * H2))
    r |= p & (position << (3 * H2))
    r |= p & (position >> H2)
    p = (position >> H2) & (position >> (2 * H2))
    r |= p & (position << H2)
    r |= p & (position >> (3 * H2))

    return r & (BOARD_MASK & ~mask)


def _mirror(bb: int) -> int:
    out = 0
    for c in range(WIDTH):
        out |= ((bb >> (c * H1)) & 0x7F) << ((WIDTH - 1 - c) * H1)
    return out


def _canon(cur: int, mask: int) -> tuple[int, int]:
    """Mirror-normalised (mask, cur) tuple used as the TT key."""
    mm, mc = _mirror(mask), _mirror(cur)
    a = (mask, cur)
    b = (mm, mc)
    return a if a <= b else b


@dataclass
class Solved:
    value: int                       # scored value (or sign in value mode)
    optimal_cols: list[int]
    best_col: int
    per_col: dict[int, int]          # scored value achieved by each legal move
    mode: str = "scored"


class Solver:
    """Reusable solver with a persistent transposition table (warmup-friendly)."""

    def __init__(self, use_nonlosing: bool = True, use_maxbound: bool = False) -> None:
        # scored TT: key -> (value, flag)  flag: 0 exact, -1 upper bound, +1 lower bound
        self.tt: dict[tuple[int, int], tuple[int, int]] = {}
        # weak (win/draw/loss) TT: separate scale, must not collide with scored TT
        self.tt_weak: dict[tuple[int, int], tuple[int, int]] = {}
        self.nodes = 0
        self.use_nonlosing = use_nonlosing
        self.use_maxbound = use_maxbound

    # ---- move ordering: threats created (desc), then center ------------- #
    @staticmethod
    def _ordered(cur: int, mask: int, candidate_moves: int):
        """Yield (move_bit,) ordered by #winning-threats the move creates, then center."""
        scored = []
        for col in CENTER_ORDER:
            move_bit = candidate_moves & (((1 << HEIGHT) - 1) << (col * H1))
            if not move_bit:
                continue
            new_mask = mask | move_bit
            threats = _compute_winning_position(cur | move_bit, new_mask)
            scored.append((threats.bit_count(), _CENTER_RANK[col], move_bit))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [(mb,) for _, _, mb in scored]

    # ---- non-losing-move generation (huge pruning win) ------------------- #
    @staticmethod
    def _non_losing_moves(cur: int, mask: int) -> int:
        opp = cur ^ mask
        possible = _possible(mask)
        opp_win = _compute_winning_position(opp, mask)
        forced = possible & opp_win
        if forced:
            if forced & (forced - 1):
                return 0                      # >1 immediate opponent threat -> lost
            possible = forced
        # do not play directly beneath a cell where the opponent could win
        return possible & ~(opp_win >> 1)

    # ---- core negamax ---------------------------------------------------- #
    def _negamax(self, cur: int, mask: int, n: int, alpha: int, beta: int) -> int:
        self.nodes += 1

        # draw: board full with no immediate win available
        if n >= SIZE:
            return 0

        # immediate win for side to move?
        if _compute_winning_position(cur, mask) & _possible(mask):
            return SIZE - n

        # moves that don't hand the opponent an immediate win
        if self.use_nonlosing:
            candidate_moves = self._non_losing_moves(cur, mask)
            if candidate_moves == 0:
                # every legal move lets the opponent win on their next move
                return -(SIZE - (n + 1))
        else:
            candidate_moves = _possible(mask)

        # upper bound: soonest we can win is our next stone (number n+3) -> SIZE-(n+3)+1
        if self.use_maxbound:
            max_possible = SIZE - (n + 3) + 1     # = SIZE - n - 2
            if beta > max_possible:
                beta = max_possible
                if alpha >= beta:
                    return beta

        key = _canon(cur, mask)
        entry = self.tt.get(key)
        if entry is not None:
            val, flag = entry
            if flag == 0:
                return val
            elif flag == 1:      # lower bound
                if val > alpha:
                    alpha = val
            else:                # upper bound
                if val < beta:
                    beta = val
            if alpha >= beta:
                return val

        alpha0 = alpha

        # order candidate moves by threats created, then center
        best = MIN_SCORE
        new_cur = cur ^ mask              # switch perspective (see Board.play)
        for (move_bit,) in self._ordered(cur, mask, candidate_moves):
            new_mask = mask | move_bit
            score = -self._negamax(new_cur, new_mask, n + 1, -beta, -alpha)
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        # store TT entry
        if best <= alpha0:
            self.tt[key] = (best, -1)     # upper bound
        elif best >= beta:
            self.tt[key] = (best, 1)      # lower bound
        else:
            self.tt[key] = (best, 0)      # exact
        return best

    # ---- fast weak (win/draw/loss) negamax ------------------------------- #
    def _weak(self, cur: int, mask: int, n: int, alpha: int, beta: int) -> int:
        """Return value in {-1,0,1} for the side to move (null-window friendly)."""
        self.nodes += 1

        if n >= SIZE:
            return 0
        if _compute_winning_position(cur, mask) & _possible(mask):
            return 1                              # win available now

        candidate_moves = self._non_losing_moves(cur, mask)
        if candidate_moves == 0:
            return -1                             # forced loss next move

        # weak bounds are [-1, 1]; use the raw position as the TT key (skip the
        # per-node mirror canonicalisation -- correctness-preserving, just less
        # sharing; mirror symmetry is applied at the root instead).
        key = (mask, cur)
        entry = self.tt_weak.get(key)
        if entry is not None:
            val, flag = entry
            if flag == 0:
                return val
            elif flag == 1:
                if val > alpha:
                    alpha = val
            else:
                if val < beta:
                    beta = val
            if alpha >= beta:
                return val

        alpha0 = alpha
        best = -1
        new_cur = cur ^ mask
        for (move_bit,) in self._ordered(cur, mask, candidate_moves):
            new_mask = mask | move_bit
            score = -self._weak(new_cur, new_mask, n + 1, -beta, -alpha)
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if best <= alpha0:
            self.tt_weak[key] = (best, -1)
        elif best >= beta:
            self.tt_weak[key] = (best, 1)
        else:
            self.tt_weak[key] = (best, 0)
        return best

    def _weak_move(self, cur: int, mask: int, n: int, col: int) -> int:
        """Value {-1,0,1} of playing `col`, from the mover's perspective."""
        move_bit = (mask + _bottom_mask(col)) & (((1 << HEIGHT) - 1) << (col * H1))
        if _compute_winning_position(cur, mask) & move_bit:
            return 1
        new_mask = mask | move_bit
        new_cur = cur ^ mask
        return -self._weak(new_cur, new_mask, n + 1, -1, 1)

    # ---- root evaluation ------------------------------------------------- #
    def _score_move(self, cur: int, mask: int, n: int, col: int) -> int:
        """Exact scored value of playing `col` (full-window child search)."""
        move_bit = (mask + _bottom_mask(col)) & (((1 << HEIGHT) - 1) << (col * H1))
        new_mask = mask | move_bit
        new_cur = cur ^ mask
        # if this move wins immediately, value is SIZE - n
        if _compute_winning_position(cur, mask) & move_bit:
            return SIZE - n
        return -self._negamax(new_cur, new_mask, n + 1, MIN_SCORE, MAX_SCORE)

    def solve_best(self, board, mode: str = "value") -> Solved:
        """Single rooted alpha-beta returning (value, best_col) WITHOUT full per_col.

        Much cheaper than `solve` for near-empty positions (one effective search
        instead of one-per-legal-move). Used to build the opening book.
        """
        if board.is_terminal():
            raise ValueError("cannot solve a terminal position")
        cur, mask, n = board.cur, board.mask, board.n
        legal = board.legal_moves()

        booked = _book_lookup(board, mode)
        if booked is not None:
            return booked

        # immediate win?
        winners = [c for c in legal if board.winning_move(c)]
        if winners:
            best_col = min(winners, key=lambda c: _CENTER_RANK[c])
            if mode == "value":
                return Solved(1, sorted(winners), best_col, {best_col: 1}, mode="value")
            v = SIZE - n
            return Solved(v, [best_col], best_col, {best_col: v}, mode="scored")

        best_val = None
        best_col = None
        per = {}
        if mode == "value":
            alpha, beta = -1, 1
            for (mb,) in self._ordered(cur, mask, self._non_losing_moves(cur, mask) or _possible(mask)):
                col = (mb.bit_length() - 1) // H1
                v = -self._weak(cur ^ mask, mask | mb, n + 1, -beta, -alpha)
                per[col] = v
                if best_val is None or v > best_val:
                    best_val, best_col = v, col
                if v > alpha:
                    alpha = v
                if best_val >= 1:      # a win is the maximum; stop (avoid null-window churn)
                    break
            for c in legal:
                per.setdefault(c, -1)
            optimal = sorted(c for c, val in per.items() if val == best_val)
            best_col = min(optimal, key=lambda c: _CENTER_RANK[c])
            return Solved(best_val, optimal, best_col, per, mode="value")
        else:
            alpha, beta = MIN_SCORE, MAX_SCORE
            max_possible = SIZE - n - 2      # cannot win faster than our next stone
            first = True
            for (mb,) in self._ordered(cur, mask, _possible(mask)):
                col = (mb.bit_length() - 1) // H1
                if first:
                    # first move: full window, guaranteed exact.
                    v = -self._negamax(cur ^ mask, mask | mb, n + 1, -beta, -alpha)
                    first = False
                else:
                    # Narrow-window probe against the current best (`alpha`).
                    # By the negamax fail-soft theorem this is EXACT only when
                    # it lands strictly below `alpha`; a probe >= alpha is only
                    # a BOUND (this move ties-or-beats the champion, exact
                    # value unknown) and must NOT be trusted as-is -- doing so
                    # let a fail-high/tie artifact masquerade as an exact
                    # score, which could (and did) make this function report
                    # a worse move as the scored-optimal `best_col` (e.g. it
                    # once reported col 2 = 12 here when the true value is 6).
                    # Re-search any such candidate with the real full window
                    # (same technique `solve()`'s `_score_move` uses) before
                    # trusting it.
                    probe = -self._negamax(cur ^ mask, mask | mb, n + 1, -MAX_SCORE, -alpha)
                    if probe >= alpha:
                        v = -self._negamax(cur ^ mask, mask | mb, n + 1, MIN_SCORE, MAX_SCORE)
                    else:
                        v = probe   # genuine fail-low: true value <= v, provably not optimal
                per[col] = v
                if best_val is None or v > best_val:
                    best_val, best_col = v, col
                if v > alpha:
                    alpha = v
                if best_val >= max_possible:
                    break
            optimal = sorted(c for c, val in per.items() if val == best_val)
            best_col = min(optimal, key=lambda c: _CENTER_RANK[c])
            return Solved(best_val, optimal, best_col, per, mode="scored")

    def solve(self, board, mode: str = "scored") -> Solved:
        if board.is_terminal():
            raise ValueError("cannot solve a terminal position")
        cur, mask, n = board.cur, board.mask, board.n
        legal = board.legal_moves()

        booked = _book_lookup(board, mode)
        if booked is not None:
            return booked

        if mode == "value":
            sper = {c: self._weak_move(cur, mask, n, c) for c in legal}
            best_sign = max(sper.values())
            optimal = sorted(c for c, v in sper.items() if v == best_sign)
            best_col = min(optimal, key=lambda c: _CENTER_RANK[c])
            return Solved(best_sign, optimal, best_col, sper, mode="value")

        per_col: dict[int, int] = {}
        for col in legal:
            per_col[col] = self._score_move(cur, mask, n, col)

        best_val = max(per_col.values())
        optimal = sorted(c for c, v in per_col.items() if v == best_val)
        best_col = min(optimal, key=lambda c: _CENTER_RANK[c])
        return Solved(best_val, optimal, best_col, per_col, mode="scored")


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


# --------------------------------------------------------------------------- #
# module-level singleton (shared TT -> warmup persists within a process)      #
# --------------------------------------------------------------------------- #
_SINGLETON: Solver | None = None


def get_solver() -> Solver:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Solver()
    return _SINGLETON


def solve(board, mode: str = "scored") -> Solved:
    return get_solver().solve(board, mode)
