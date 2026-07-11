"""Paired round-robin ladder + Elo (METRIC.md §2).

Each ordered pair (A, B) plays every opening book with A moving first, then B
moving first, so colour is balanced. Win=1, draw=0.5, loss=0. Elo is fit by a
deterministic iterative logistic update, anchoring `random` at 0. Elo is a
secondary signal (not the pass/fail gate) but must be reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.engine.board import Board
from app.neurogolf.config import OPENING_BOOKS


def play_game(agent_first, agent_second, opening: list[int]) -> float:
    """Return result for `agent_first`: 1.0 win, 0.5 draw, 0.0 loss."""
    b = Board.empty()
    for col in opening:
        if b.is_terminal() or not b.can_play(col):
            break
        b = b.play(col)
    # players keyed by side-to-move value: opening length parity decides who is "first agent"
    # agent_first plays whenever it's the side that moved on ply 0 of *its* turn schedule.
    # Simpler: agent_first controls player 1, agent_second controls player 2.
    ply = b.n
    while not b.is_terminal():
        mover = agent_first if (b.player_to_move() == 1) else agent_second
        col = mover.select_move(b)
        if not b.can_play(col):
            # illegal move => forfeit
            return 0.0 if mover is agent_first else 1.0
        b = b.play(col)
        ply += 1
    w = b.winner()
    if w == 0:
        return 0.5
    return 1.0 if w == 1 else 0.0


@dataclass
class LadderResult:
    scores: dict[str, float]        # total points
    games: dict[str, int]
    elo: dict[str, int]
    matrix: dict[str, dict[str, float]]


def run(agents, openings=None) -> LadderResult:
    if openings is None:
        openings = OPENING_BOOKS
    names = [a.name for a in agents]
    by_name = {a.name: a for a in agents}

    scores = {n: 0.0 for n in names}
    games = {n: 0 for n in names}
    matrix = {n: {m: 0.0 for m in names} for n in names}
    # per-pair aggregated results for Elo fitting
    pair_games = []   # (i_name, j_name, result_for_i)

    for a in names:
        for bname in names:
            if a == bname:
                continue
            for opening in openings:
                r = play_game(by_name[a], by_name[bname], opening)
                scores[a] += r
                scores[bname] += (1.0 - r)
                games[a] += 1
                games[bname] += 1
                matrix[a][bname] += r
                pair_games.append((a, bname, r))

    elo = _fit_elo(names, pair_games)
    return LadderResult(scores=scores, games=games, elo=elo, matrix=matrix)


def _fit_elo(names, pair_games, iters: int = 4000, k: float = 8.0) -> dict[str, int]:
    ratings = {n: 0.0 for n in names}
    for _ in range(iters):
        grad = {n: 0.0 for n in names}
        for (i, j, r) in pair_games:
            expected = 1.0 / (1.0 + 10 ** ((ratings[j] - ratings[i]) / 400.0))
            grad[i] += (r - expected)
            grad[j] += ((1.0 - r) - (1.0 - expected))
        for n in names:
            ratings[n] += k * grad[n] / max(1, len(pair_games))
        # anchor random at 0
        if "random" in ratings:
            shift = ratings["random"]
            for n in names:
                ratings[n] -= shift
    return {n: int(round(ratings[n])) for n in names}
