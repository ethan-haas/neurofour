"""`neurofour-net16`: gen-2 "research-model / cost-axis compression" -- net1's
EXACT leaf value net (same 4705 params, FEATURE_DIM->24->1, tanh head, same
1-ply tactical-guard + negamax value search) re-serialised in a much smaller
artifact: per-ROW int4 quantisation of W1 (the dominant tensor) instead of
net1's per-tensor int8, plus optional global magnitude pruning of W1 before
quantisation, all packed in a hand-rolled binary container (no zip/npz
overhead) and whole-file zlib-compressed (see `app/agents/mlp.py`
`save_compressed`/`load_compressed`).

No retraining: the compressed artifact is built by re-quantising net1's own
*effective* (already-dequantised) weights -- see `scripts/compress_net1.py`
-- so any accuracy delta vs net1 is caused ONLY by the extra quantisation/
pruning error, not by different training data or a different objective.

Dequantisation happens once at __init__ (`mlp.load_compressed`); after that
this is byte-for-byte net1's own `select_move`/tactical-guard/1-ply-negamax
code (inherited, not reimplemented) over the SAME dense float32 forward
pass, so `flops_per_move` is computed with net1's exact formula (same
param count, same FEATURE_DIM encoder) -- no flops are "saved" by pruning
since the search still does a dense forward pass (zeros multiply through,
they are not skipped). Only `size_bytes` (the real on-disk artifact length)
differs. See `scripts/compress_net1.py`'s docstring for the honest sweep
results and the registered winner's level.
"""
from __future__ import annotations

import os

from app.agents.base import AgentManifest
from app.agents.net1 import Net1Agent
from app.agents.encode import FEATURE_DIM
from app.agents.mlp import load_compressed

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACT = os.path.join(_HERE, "artifacts", "neurofour-net16.b4")


class Net16Agent(Net1Agent):
    name = "neurofour-net16"
    kind = "search"

    def __init__(self, artifact_path: str = DEFAULT_ARTIFACT):
        self.artifact_path = artifact_path
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"neurofour-net16 artifact missing: {artifact_path}. "
                f"Run scripts/compress_net1.py first."
            )
        self._w = load_compressed(artifact_path)

    def manifest(self) -> AgentManifest:
        params = int(self._w["params"])
        size = os.path.getsize(self.artifact_path)
        from app.engine.board import WIDTH
        flops = WIDTH * (2 * params + FEATURE_DIM)   # net1's exact formula
        return AgentManifest(self.name, self.kind, params=params, size_bytes=size,
                             flops_per_move=flops, artifact_path=self.artifact_path)
