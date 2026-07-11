"""gen-10 T-lever: `heuristic_eval_bb.evaluate_bb()` must return the EXACT
same integer score as `heuristic_eval.evaluate()` for every board tested --
a strictly stronger guarantee than "same argmax/ranking" (identical scores
trivially implies identical ranking, and identical move selection by any
search that consumes this leaf value)."""
from __future__ import annotations

import json
import os
import random

from app.agents.heuristic_eval import evaluate
from app.agents.heuristic_eval_bb import evaluate_bb
from app.engine.board import Board

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "bench_data")


def _load(path, k=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if k and len(out) >= k:
                break
    return out


def _varied_boards(n=600):
    boards = []
    for name in ("dev_big.jsonl", "dev_big2.jsonl", "dev_big3.jsonl"):
        p = os.path.join(_DATA, name)
        if os.path.exists(p):
            ps = _load(p, 200)
            boards += [Board.from_moves(row["board"]) for row in ps]
    # edge cases: empty, single moves, small openings
    boards += [Board.from_moves(seq) for seq in
               ([[]] + [[c] for c in range(7)] +
                [[3, 3], [3, 2], [2, 4], [3, 3, 3], [0, 6], [3, 4, 2, 5],
                 [1, 5], [0, 2], [6, 4], [3, 3, 4, 4, 2, 2, 5, 5]])]
    rng = random.Random(20260710)
    while len(boards) < n:
        b = Board.empty()
        depth = rng.randint(1, 41)
        for _ in range(depth):
            if b.is_terminal():
                break
            b = b.play(rng.choice(b.legal_moves()))
        boards.append(b)
    return boards[:max(n, len(boards))]


def test_evaluate_bb_matches_evaluate_exactly_on_many_boards():
    boards = _varied_boards(600)
    assert len(boards) >= 500
    mismatches = []
    for b in boards:
        if b.is_terminal():
            # evaluate()/evaluate_bb() are only ever called on non-terminal
            # leaves by net14.py, but equivalence should still hold if
            # called (both are pure functions of the same board state).
            pass
        old = evaluate(b)
        new = evaluate_bb(b)
        if old != new:
            mismatches.append((b.to_key(), old, new))
    assert not mismatches, f"{len(mismatches)}/{len(boards)} score mismatches: {mismatches[:5]}"


def test_evaluate_bb_matches_evaluate_on_near_terminal_boards():
    """Boards one ply from a win/loss/draw -- the densest score buckets."""
    rng = random.Random(4242)
    mismatches = []
    n = 0
    for _ in range(2000):
        if n >= 150:
            break
        b = Board.empty()
        for _ in range(rng.randint(28, 40)):
            if b.is_terminal():
                break
            b = b.play(rng.choice(b.legal_moves()))
        if b.is_terminal():
            continue
        n += 1
        old = evaluate(b)
        new = evaluate_bb(b)
        if old != new:
            mismatches.append((b.to_key(), old, new))
    assert n >= 100
    assert not mismatches, f"{len(mismatches)}/{n} score mismatches: {mismatches[:5]}"
