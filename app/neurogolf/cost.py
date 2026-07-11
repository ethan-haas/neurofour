"""Cost measurement (METRIC.md §3-4).

- size_bytes: real on-disk artifact size (from the manifest).
- params:     learned scalar count.
- flops_per_move: declared, with a plausibility check (nn: ~2*params*forward;
  reject values below a floor or implausibly low vs the model size).
- latency_ms: measured p50 over the sealed set (warmup excluded).
- over_budget: latency_p50 > LATENCY_CAP_MS OR flops_per_move > FLOP_CAP.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.engine.board import Board
from app.neurogolf.config import LATENCY_CAP_MS, FLOP_CAP


@dataclass
class CostCard:
    size_bytes: int
    params: int
    flops_per_move: int
    latency_ms: float
    over_budget: bool
    flops_plausible: bool

    def to_dict(self) -> dict:
        return {
            "size_bytes": self.size_bytes,
            "params": self.params,
            "flops_per_move": self.flops_per_move,
            "latency_ms": round(self.latency_ms, 4),
            "over_budget": self.over_budget,
            "flops_plausible": self.flops_plausible,
        }


def _flops_plausible(manifest) -> bool:
    """Sanity-check the declared flops for nn agents against ~2*params."""
    if manifest.kind == "nn":
        if manifest.params <= 0:
            return False
        lo = 2 * manifest.params            # single forward pass lower bound
        hi = 50 * manifest.params + 100_000  # generous upper bound
        return lo * 0.5 <= manifest.flops_per_move <= hi
    # non-nn agents: any non-negative declared value accepted
    return manifest.flops_per_move >= 0


def measure(agent, positions: list[dict], warmup: int = 8,
            max_samples: int = 200) -> CostCard:
    man = agent.manifest()
    boards = [Board.from_moves(p["board"]) for p in positions[:max_samples]]

    # warmup (JIT / cache priming) -- excluded from timing
    for b in boards[:warmup]:
        agent.select_move(b)

    timings = []
    for b in boards:
        t0 = time.perf_counter()
        agent.select_move(b)
        timings.append((time.perf_counter() - t0) * 1000.0)
    timings.sort()
    p50 = timings[len(timings) // 2] if timings else 0.0

    over = (p50 > LATENCY_CAP_MS) or (man.flops_per_move > FLOP_CAP)
    return CostCard(
        size_bytes=man.size_bytes,
        params=man.params,
        flops_per_move=man.flops_per_move,
        latency_ms=p50,
        over_budget=over,
        flops_plausible=_flops_plausible(man),
    )
