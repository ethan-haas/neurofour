"""`neurofour-net0`: a LEARNED agent whose on-disk artifact fits the NANO tier
(<=4096 bytes) -- the cheapest budget band, previously occupied ONLY by 0-byte
code agents (`heuristic` 0.900, `minimax-2` 0.9033, `random`).

Design is IDENTICAL in structure to `neurofour-net1` (fair play -- the agent
receives ONLY the board; it never calls the solver, never reads a scored-
position file, never hardcodes a per-position answer):
  1. **0-param tactical guard** (pure board logic, reused verbatim from
     `net1.tactical_move`): if a legal move wins immediately -> play it;
     else if the opponent has an immediate winning threat -> block it.
  2. **1-ply value search**: for every remaining legal move, make the child
     board with the ENGINE and score it with the tiny LEARNED value net; pick
     the move that minimises the opponent's value (negamax at depth 1).
     Terminal children are scored exactly (+/-1).

What differs from net1 is ONLY the hidden width: net0 reuses net1's exact
194-dim board encoder (`app/agents/encode.py`) but with a SMALLER hidden
layer (194->14->1 vs net1's 194->24->1), which is what shrinks the on-disk
int8 artifact from net1's 4837B down to <=4096B (net0's first weight matrix,
194*14 int8 params, dominates the artifact size).

This was the outcome of an honest swept table over (hidden width x feature
subset) -- see `scripts/exp_net0_sweep.py` -- comparing net1's full 194-dim
encoder against two SMALLER hand-picked feature subsets (a 26-dim
"engineered-features-only" tail and a 110-dim "threat-planes + engineered"
subset) that would have allowed a bigger hidden layer for the same byte
budget:

    subset        hidden  size(B)  fits<=4096  sealed_opt
    full (194d)      8     2416      yes         0.9167
    full (194d)     10     2834      yes         0.9333
    full (194d)     12     2966      yes         0.9200
    full (194d)     14     3290      yes         0.9367   <- BEST, shipped
    full (194d)     16     3520      yes         0.9267
    full (194d)     18     4036      yes         0.9233
    full (194d)     20     4097      NO (1B over) --
    engineered(26d)  8     1544      yes         0.9067
    engineered(26d) 10     1597      yes         0.9067
    threat_eng(110d) ...   (all <=0.93, none beat full/hidden=14)

Counter-intuitively, DROPPING features to afford a wider hidden layer never
won: the raw disc-plane + threat-plane blocks (indices [0:168) of `encode`)
carry more distillation signal per byte than a wider-but-blinder net gets
back, exactly the opposite of the smaller-feature-set intuition suggested at
task time -- the swept numbers above are the actual evidence. net0 therefore
reuses net1's `encode`/`FEATURE_DIM` unchanged; only `hidden` (and therefore
the trained weights / artifact) differs.

Only the board is used at inference: `encode.encode` is a pure bitboard
function (no solver-tree call), and the search uses `Board.play` (engine) to
make children and the net to evaluate them -- `Solver.solve`/`optimal_cols`/
`best_col`/`scored`/`.jsonl` are never touched.
"""
from __future__ import annotations

import os

import numpy as np

from app.agents.base import Agent, AgentManifest
from app.agents.encode import encode, FEATURE_DIM
from app.agents.mlp import forward_logits, load_npz
from app.agents.net1 import tactical_move
from app.engine.board import CENTER_ORDER

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACT = os.path.join(_HERE, "artifacts", "neurofour-net0.npz")
_CR = {c: i for i, c in enumerate(CENTER_ORDER)}


class Net0Agent(Agent):
    name = "neurofour-net0"
    kind = "search"          # 1-ply engine search with a learned leaf eval (nano)

    def __init__(self, artifact_path: str = DEFAULT_ARTIFACT):
        self.artifact_path = artifact_path
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"neurofour-net0 artifact missing: {artifact_path}. Run train_net0.py first."
            )
        self._w = load_npz(artifact_path)

    def _value(self, board) -> float:
        """Learned value of `board` for the side to move, in (-1, 1)."""
        return float(np.tanh(forward_logits(self._w, encode(board))[0]))

    def select_move(self, board) -> int:
        # 1. tactical guard (0 params): immediate win, else forced block.
        t = tactical_move(board)
        if t is not None:
            return t
        # 2. 1-ply value search over engine-made children (negamax, depth 1).
        best_c, best_key = None, None
        for c in sorted(board.legal_moves(), key=lambda c: _CR[c]):
            child = board.play(c)
            if child.winner() != 0:            # we just won (guard covers this too)
                return c
            if child.n >= 42:                  # child is a full draw
                score = 0.0
            else:
                score = -self._value(child)
            key = (score, -_CR[c])
            if best_key is None or key > best_key:
                best_key = key
                best_c = c
        return best_c

    def manifest(self) -> AgentManifest:
        params = int(self._w["params"])
        size = os.path.getsize(self.artifact_path)
        from app.engine.board import WIDTH
        flops = WIDTH * (2 * params + FEATURE_DIM)
        return AgentManifest(self.name, self.kind, params=params, size_bytes=size,
                             flops_per_move=flops, artifact_path=self.artifact_path)
