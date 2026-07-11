"""Strength scoring over a labelled position set (METRIC.md §1).

All quantities are computed from the pre-labelled `scored` dict, so no solver call
is needed at scoring time:
    optimality   = mean[ agent_move in optimal_cols ]
    blunder      = mean[ sign(scored[agent_move]) < best_value ]
    soundness    = 1 - blunder_rate
plus per-outcome (win/draw/loss) optimality breakdowns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.engine.board import Board


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


@dataclass
class StrengthCard:
    optimality: float
    blunder_rate: float
    soundness: float
    n: int
    per_outcome: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "optimality": round(self.optimality, 6),
            "blunder_rate": round(self.blunder_rate, 6),
            "soundness": round(self.soundness, 6),
            "n": self.n,
            "per_outcome": self.per_outcome,
        }


def score(agent, positions: list[dict]) -> StrengthCard:
    total = len(positions)
    opt_hits = 0
    blunders = 0
    # per-outcome buckets keyed by position value (win/draw/loss for side to move)
    buckets = {1: [0, 0], 0: [0, 0], -1: [0, 0]}   # value -> [opt_hits, count]

    for p in positions:
        board = Board.from_moves(p["board"])
        m = agent.select_move(board)
        scored = p["scored"]
        best_val = p["value"]           # value-only best (sign of max scored)
        optimal = p["optimal_cols"]

        hit = 1 if m in optimal else 0
        opt_hits += hit

        # value-only outcome of the agent's move
        move_val = _sign(scored.get(m, best_val))
        if move_val < best_val:
            blunders += 1

        b = buckets[best_val]
        b[0] += hit
        b[1] += 1

    optimality = opt_hits / total if total else 0.0
    blunder_rate = blunders / total if total else 0.0
    per_outcome = {}
    names = {1: "winning", 0: "drawn", -1: "losing"}
    for val, (hits, cnt) in buckets.items():
        per_outcome[names[val]] = {
            "optimality": round(hits / cnt, 6) if cnt else None,
            "n": cnt,
        }

    return StrengthCard(
        optimality=optimality,
        blunder_rate=blunder_rate,
        soundness=1.0 - blunder_rate,
        n=total,
        per_outcome=per_outcome,
    )
