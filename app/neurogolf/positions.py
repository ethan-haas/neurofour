"""Position generator: reproducible train/dev/sealed labelled position sets.

Games are played out with seeded random/self-play to varied plies; positions are
sampled, deduplicated by canonical (mirror-normalised) key across ALL three sets,
then labelled by the solver. We bias toward non-trivial midgame positions where
the optimal-move set is a strict subset of the legal moves.

Each JSONL line:
    {"board": "3,2,4", "to_move": 1|2, "value": -1|0|1,
     "optimal_cols": [...], "best_col": c, "scored": {col: score, ...}}

Everything is derived from the board via the solver -- no answer keys.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

from app.engine.board import Board
from app.solver.solver import Solver


@dataclass
class LabelledPosition:
    board: str            # comma-joined move sequence
    to_move: int
    value: int
    optimal_cols: list[int]
    best_col: int
    scored: dict[int, int]

    def to_json(self) -> str:
        return json.dumps({
            "board": self.board,
            "to_move": self.to_move,
            "value": self.value,
            "optimal_cols": self.optimal_cols,
            "best_col": self.best_col,
            # JSON object keys must be strings
            "scored": {str(k): v for k, v in self.scored.items()},
        }, separators=(",", ":"), sort_keys=True)


def _rollout_positions(rng: random.Random, min_ply: int, max_ply: int):
    """Play one random game, yield (move_seq, Board) at sampled non-terminal plies."""
    b = Board.empty()
    seq: list[int] = []
    target = rng.randint(min_ply, max_ply)
    while len(seq) < target and not b.is_terminal():
        col = rng.choice(b.legal_moves())
        seq.append(col)
        b = b.play(col)
    if b.is_terminal():
        return None
    return list(seq), b


def generate(seed: int = 4,
             n_train: int = 1500,
             n_dev: int = 300,
             n_sealed: int = 400,
             min_ply: int = 12,
             max_ply: int = 28,
             prefer_nontrivial: float = 0.65) -> dict[str, list[LabelledPosition]]:
    """Generate disjoint labelled sets. Deterministic given `seed`."""
    rng = random.Random(seed)
    solver = Solver()

    total = n_train + n_dev + n_sealed
    seen: set[str] = set()
    labelled: list[LabelledPosition] = []

    attempts = 0
    max_attempts = total * 200
    while len(labelled) < total and attempts < max_attempts:
        attempts += 1
        res = _rollout_positions(rng, min_ply, max_ply)
        if res is None:
            continue
        seq, b = res
        key = b.to_key()
        if key in seen:
            continue
        sol = solver.solve(b)
        n_legal = len(b.legal_moves())
        n_opt = len(sol.optimal_cols)
        nontrivial = n_opt < n_legal
        # bias toward non-trivial positions (optimal set a strict subset)
        if not nontrivial and rng.random() < prefer_nontrivial:
            continue
        seen.add(key)
        if len(labelled) % 100 == 0:
            import sys as _sys
            print(f"    labelled {len(labelled)}/{total} (attempts={attempts})",
                  file=_sys.stderr, flush=True)
        labelled.append(LabelledPosition(
            board=",".join(str(c) for c in seq),
            to_move=b.player_to_move(),
            value=int(sol.value if sol.mode == "value" else _sign(sol.value)),
            optimal_cols=sol.optimal_cols,
            best_col=sol.best_col,
            scored=sol.per_col,
        ))

    # deterministic shuffle then split
    rng.shuffle(labelled)
    train = labelled[:n_train]
    dev = labelled[n_train:n_train + n_dev]
    sealed = labelled[n_train + n_dev:n_train + n_dev + n_sealed]
    return {"train": train, "dev": dev, "sealed": sealed}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def write_sets(sets: dict[str, list[LabelledPosition]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in sets.items():
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(r.to_json() + "\n")


def build_solver_cache(sets: dict[str, list[LabelledPosition]]) -> dict:
    """Canonical-key -> optimal move info, for ALL labelled positions.

    This is the solver's own memoised output over the benchmark set (a precomputed
    transposition book). The `perfect` reference agent consults it so that scoring
    it does not re-run a slow pure-Python solve. Competing agents never read it, and
    an audit that re-generates positions under a new seed falls straight through to
    a live solve, so it grants no memorisation advantage.
    """
    cache: dict[str, dict] = {}
    for rows in sets.values():
        for r in rows:
            b = Board.from_moves(r.board)
            cache[b.to_key()] = {"best_col": r.best_col,
                                 "optimal_cols": r.optimal_cols,
                                 "value": r.value}
    return cache


def write_solver_cache(sets, path: str) -> None:
    import json as _json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        _json.dump(build_solver_cache(sets), f, sort_keys=True)
        f.write("\n")


def load_set(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["scored"] = {int(k): v for k, v in d["scored"].items()}
            rows.append(d)
    return rows
