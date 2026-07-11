"""Agent interface + manifest."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AgentManifest:
    name: str
    kind: str                     # "table" | "nn" | "search" | "heuristic" | "random"
    params: int
    size_bytes: int
    flops_per_move: int
    artifact_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class Agent:
    """Base agent. `select_move` receives ONLY the board -- nothing else."""

    name: str = "agent"
    kind: str = "heuristic"

    def select_move(self, board) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def manifest(self) -> AgentManifest:  # pragma: no cover - interface
        raise NotImplementedError
