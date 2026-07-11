"""The flagship `neurofour-net` agent: a tiny numpy MLP over the shared encoder.

Loads its quantised weights from `app/agents/artifacts/neurofour-net.npz`.
`select_move` receives ONLY the board -- no solver, no labels.
"""
from __future__ import annotations

import os

import numpy as np

from app.agents.base import Agent, AgentManifest
from app.agents.encode import encode, FEATURE_DIM
from app.agents.mlp import forward_logits, masked_argmax, load_npz

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACT = os.path.join(_HERE, "artifacts", "neurofour-net.npz")


class NetAgent(Agent):
    name = "neurofour-net"
    kind = "nn"

    def __init__(self, artifact_path: str = DEFAULT_ARTIFACT):
        self.artifact_path = artifact_path
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"neurofour-net artifact missing: {artifact_path}. Run train_net.py first."
            )
        self._w = load_npz(artifact_path)

    def select_move(self, board) -> int:
        x = encode(board)
        logits = forward_logits(self._w, x)
        return masked_argmax(logits, board.legal_moves())

    def manifest(self) -> AgentManifest:
        params = int(self._w["params"])
        size = os.path.getsize(self.artifact_path)
        # nn cost ~ 2 * params for a single forward pass (+ small encoder cost)
        flops = 2 * params + FEATURE_DIM
        return AgentManifest(self.name, self.kind, params=params, size_bytes=size,
                             flops_per_move=flops, artifact_path=self.artifact_path)
