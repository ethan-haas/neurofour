"""Agent interface + manifest."""
from __future__ import annotations

import os
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
        """Serialize for API responses.

        BUG FIX (info leak): `artifact_path` is an absolute SERVER
        filesystem path (e.g. `/opt/render/project/src/app/agents/artifacts/
        neurofour-net.npz` in prod) -- a public API must never leak that.
        This is the one choke point every serialized manifest passes through
        (`/agents` and `/evaluate` in app/main.py both call
        `AgentManifest.to_dict()`), so collapsing the path to just its
        basename here closes the leak everywhere at once. The `artifact_path`
        *attribute* on the dataclass instance is left completely untouched --
        internal code (agent constructors, tests) that reads
        `self.artifact_path` still gets the real, usable path; only the
        JSON-serialized copy is redacted.
        """
        d = asdict(self)
        path = d.get("artifact_path")
        d["artifact_path"] = os.path.basename(path) if path else None
        return d


class Agent:
    """Base agent. `select_move` receives ONLY the board -- nothing else."""

    name: str = "agent"
    kind: str = "heuristic"

    def select_move(self, board) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def manifest(self) -> AgentManifest:  # pragma: no cover - interface
        raise NotImplementedError
