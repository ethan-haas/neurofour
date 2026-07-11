"""Baseline agents: random, heuristic, minimax-k, perfect."""
from __future__ import annotations

import os
import random as _random

from app.engine.board import CENTER_ORDER
from app.agents.base import Agent, AgentManifest
from app.agents.heuristic_eval import evaluate, WIN_SCORE
from app.neurogolf.config import EXACT_SOLVE_MIN_PLY

_CENTER_RANK = {c: i for i, c in enumerate(CENTER_ORDER)}


def _seed() -> int:
    try:
        return int(os.environ.get("NEUROFOUR_SEED", "4"))
    except ValueError:
        return 4


class RandomAgent(Agent):
    name = "random"
    kind = "random"

    def __init__(self):
        self._rng = _random.Random(_seed())

    def select_move(self, board) -> int:
        moves = board.legal_moves()
        # deterministic given the seed and move history length
        r = _random.Random((_seed() * 1_000_003) ^ (board.n * 2_654_435_761) ^ board.mask)
        return r.choice(moves)

    def manifest(self) -> AgentManifest:
        return AgentManifest(self.name, self.kind, params=0, size_bytes=0,
                             flops_per_move=7)


class HeuristicAgent(Agent):
    """1-ply: take immediate win, block immediate loss, else best static eval."""
    name = "heuristic"
    kind = "heuristic"

    def select_move(self, board) -> int:
        moves = board.legal_moves()
        # 1. immediate win
        wins = [c for c in moves if board.winning_move(c)]
        if wins:
            return min(wins, key=lambda c: _CENTER_RANK[c])
        # 2. block: does opponent have an immediate win after we do nothing?
        #    i.e. any column where the opponent would win next -> we must occupy it.
        opp_threats = []
        for c in moves:
            after = board.play(c)
            # after our move, opponent to move; can they win immediately?
            if any(after.winning_move(oc) for oc in after.legal_moves()):
                continue  # this move allows an opponent win -> avoid if possible
            opp_threats.append(c)
        # prefer moves that do NOT hand opponent an immediate win
        candidates = opp_threats if opp_threats else moves
        # 3. static eval one ply
        best_score = None
        best_col = candidates[0]
        for c in candidates:
            after = board.play(c)
            if after.winner() == board.player_to_move():
                return c
            # score from our perspective = -eval(after) (after is opponent's turn)
            s = -evaluate(after)
            key = (s, -_CENTER_RANK[c])
            if best_score is None or key > best_score:
                best_score = key
                best_col = c
        return best_col

    def manifest(self) -> AgentManifest:
        return AgentManifest(self.name, self.kind, params=0, size_bytes=0,
                             flops_per_move=7 * 70)  # ~7 moves * ~69 windows


class MinimaxAgent(Agent):
    """Depth-limited negamax with alpha-beta over the static eval (no solver)."""
    kind = "search"

    def __init__(self, depth: int):
        self.depth = depth
        self.name = f"minimax-{depth}"

    def _negamax(self, board, depth: int, alpha: int, beta: int) -> int:
        w = board.winner()
        if w != 0:
            # someone just won; the side to move is the loser
            return -WIN_SCORE - depth        # prefer slower losses / faster wins
        if board.n >= 42:
            return 0
        if depth == 0:
            return evaluate(board)
        best = -(WIN_SCORE * 10)
        for c in CENTER_ORDER:
            if not board.can_play(c):
                continue
            child = board.play(c)
            score = -self._negamax(child, depth - 1, -beta, -alpha)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def select_move(self, board) -> int:
        moves = board.legal_moves()
        # immediate win shortcut
        wins = [c for c in moves if board.winning_move(c)]
        if wins:
            return min(wins, key=lambda c: _CENTER_RANK[c])
        best_score = None
        best_col = min(moves, key=lambda c: _CENTER_RANK[c])
        alpha, beta = -(WIN_SCORE * 10), (WIN_SCORE * 10)
        for c in sorted(moves, key=lambda c: _CENTER_RANK[c]):
            child = board.play(c)
            score = -self._negamax(child, self.depth - 1, -beta, -alpha)
            key = (score, -_CENTER_RANK[c])
            if best_score is None or key > best_score:
                best_score = key
                best_col = c
            if score > alpha:
                alpha = score
        return best_col

    def manifest(self) -> AgentManifest:
        # rough branching estimate: 7^depth leaves * ~70 flops/eval
        leaves = 7 ** self.depth
        return AgentManifest(self.name, self.kind, params=0, size_bytes=0,
                             flops_per_move=leaves * 70)


class PerfectAgent(Agent):
    """Exact-solver reference -- the strength & cost ceiling (intentionally over budget).

    Move selection must always be game-theoretically optimal (a move in the
    position's mate-distance-aware `optimal_cols`, i.e. the scored argmax the
    solver / label generator define -- see app/solver/solver.py and
    app/neurogolf/positions.py):
      * opening positions (near-empty) come from the committed opening book, if any;
      * `app/data/solver_cache.json` (a memoised transposition book over every
        labelled train/dev/sealed position) is consulted next as a *speed*
        optimisation only -- it must never be the sole source of correctness;
      * on a cache/book MISS, positions with >= EXACT_SOLVE_MIN_PLY stones are
        solved exactly & quickly (scored, mate-aware) via the live solver. This
        threshold is the SAME one `/analyze` (app/main.py) uses, and is chosen
        so it is <= the minimum stone-count of any labelled dev/sealed position
        (see tests/test_perfect_agent.py::test_exact_solve_min_ply_covers_all_labelled_positions).
        That means a cache/book miss can NEVER silently downgrade a
        strength-scored position to the non-exact fallback below.
      * the only remaining case -- a genuinely shallow (< EXACT_SOLVE_MIN_PLY)
        position that is off-book -- can appear during ladder self-play from a
        near-empty opening; there we fall back to a deep alpha-beta so we never
        trigger a multi-minute pure-Python full solve. This path is provably
        unreachable for anything strength.score() ever queries, because dev/sealed
        positions are never that shallow -- it is NOT relied upon for correctness.

    Regression history: this agent previously used FAST_PLY=16, one ply above
    dev/sealed's own minimum of 14. On a cache miss for a 14- or 15-stone
    labelled position it would silently fall through to the non-exact,
    depth-6, static-eval fallback below -- which can (and did) prefer a
    losing move over a drawing one on a genuine horizon effect. Cache misses
    should not normally happen (the shipped cache covers every labelled
    position), but nothing enforced that invariant, so a stale/corrupted/
    incomplete cache file could silently make "perfect" not perfect while its
    published leaderboard card kept claiming optimality=1.0. Fixed by aligning
    FAST_PLY with the actual data floor (EXACT_SOLVE_MIN_PLY) so a cache miss
    on any position that can be strength-scored always takes the exact-solve
    path instead.
    """
    name = "perfect"
    kind = "search"
    FAST_PLY = EXACT_SOLVE_MIN_PLY   # >= this many stones: exact scored solve is fast

    def __init__(self):
        # imported lazily so ordinary agents never pull the solver into scope
        import json as _json
        from app.solver.solver import Solver, _load_book
        self._solver = Solver()
        self._book = _load_book()
        # precomputed transposition cache over the benchmark set (see positions.py)
        self._cache = {}
        cpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "solver_cache.json")
        try:
            with open(cpath, "r", encoding="utf-8") as f:
                self._cache = _json.load(f)
        except (FileNotFoundError, ValueError):
            self._cache = {}
        # fast bounded search for shallow off-book positions (ladder openings);
        # deep positions are still solved exactly, so strength scoring is unaffected.
        self._fallback = MinimaxAgent(6)

    def select_move(self, board) -> int:
        key = board.to_key()
        entry = self._cache.get(key) or self._book.get(key)
        if entry is not None:
            return entry["best_col"]
        if board.n >= self.FAST_PLY:
            return self._solver.solve(board, mode="scored").best_col
        # immediate win / block first, then a deep bounded search
        moves = board.legal_moves()
        wins = [c for c in moves if board.winning_move(c)]
        if wins:
            return min(wins, key=lambda c: _CENTER_RANK[c])
        return self._fallback.select_move(board)

    def manifest(self) -> AgentManifest:
        # honest lower bound: a full game-tree search is far above the FLOP cap
        return AgentManifest(self.name, self.kind, params=0, size_bytes=0,
                             flops_per_move=50_000_000)
