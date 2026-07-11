"""`neurofour-net1`: a LEARNED agent that legitimately beats the hand heuristic.

Design (fair play -- the agent receives ONLY the board; it never calls the solver):
  1. **0-param tactical guard** (pure board logic, negligible FLOPs):
       * if a legal move wins immediately -> play it;
       * else if the opponent has an immediate winning threat -> block it.
     This guarantees the agent never misses an immediate win or a forced block.
  2. **1-ply value search**: for every remaining legal move, make the child board
     with the ENGINE and score it with a tiny LEARNED value net; pick the move
     that minimises the opponent's value (negamax at depth 1). Terminal children
     are scored exactly (+/-1) so a winning/blocking line is always preferred.

The value net is a FEATURE_DIM->H->1 MLP with a tanh head, distilled from the
solver's *scored* position value on `train`(+`dev`) only (see `train_net1.py`).
Only the board is used at inference: the search uses `Board.play` (engine) to make
children and the net to evaluate them -- calling `solver.solve` is NOT done.
"""
from __future__ import annotations

import os

import numpy as np

from app.agents.base import Agent, AgentManifest
from app.agents.encode import encode, FEATURE_DIM
from app.agents.mlp import forward_logits, load_npz
from app.engine.board import CENTER_ORDER, _won, _bottom_mask, _column_mask

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACT = os.path.join(_HERE, "artifacts", "neurofour-net1.npz")
_CR = {c: i for i, c in enumerate(CENTER_ORDER)}


def tactical_move(board):
    """Return the forced tactical column, or None. 0 params, pure board logic.

    Priority: (1) an immediate winning move for the side to move; (2) if none,
    a column that blocks an immediate opponent win. Deterministic (center-first).
    """
    moves = board.legal_moves()
    wins = [c for c in moves if board.winning_move(c)]
    if wins:
        return min(wins, key=lambda c: _CR[c])
    opp = board.mask ^ board.cur
    blocks = []
    for c in moves:
        landing = (board.mask + _bottom_mask(c)) & _column_mask(c)
        if _won(opp | landing):
            blocks.append(c)
    if blocks:
        return min(blocks, key=lambda c: _CR[c])
    return None


class Net1Agent(Agent):
    name = "neurofour-net1"
    kind = "search"          # 1-ply engine search with a learned leaf eval

    def __init__(self, artifact_path: str = DEFAULT_ARTIFACT):
        self.artifact_path = artifact_path
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"neurofour-net1 artifact missing: {artifact_path}. Run train_net1.py first."
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
                # child is the opponent's turn; -value(child) = value for us
                score = -self._value(child)
            key = (score, -_CR[c])
            if best_key is None or key > best_key:
                best_key = key
                best_c = c
        return best_c

    def manifest(self) -> AgentManifest:
        params = int(self._w["params"])
        size = os.path.getsize(self.artifact_path)
        # honest cost: up to WIDTH child evals, each one forward pass (2*params)
        # over the FEATURE_DIM encoder; the tactical guard is O(WIDTH) bit-ops.
        from app.engine.board import WIDTH
        flops = WIDTH * (2 * params + FEATURE_DIM)
        return AgentManifest(self.name, self.kind, params=params, size_bytes=size,
                             flops_per_move=flops, artifact_path=self.artifact_path)
