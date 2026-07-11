"""`neurofour-net16b`: net4's exact depth-3/top-K=2 beam refutation search
structure, plugged with `neurofour-net16`'s COMPRESSED leaf artifact (the
SAME 2867-byte per-row-int5 + 40%-prune file `net16.py`/`net16s.py` load via
`app.agents.mlp.load_compressed`) instead of net1's plain int8 npz net4
defaults to.

Companion to `net16s.py` (net2's full-width D=3 search over the same
compressed leaf): net4's D=3,K=2 beam ties net2's optimality at ~4x fewer
declared flops when paired with net1's uncompressed leaf (see net4.py's
docstring sweep table), so it's the natural second search structure to test
over the compressed leaf -- if net16s (full-width) already recovers net2-
class accuracy from the compressed leaf, net16b checks whether the SAME
accuracy is reachable at net4's cheaper flops budget too (i.e. whether a
nano-size, low-flops agent can match net2/net4's micro-tier accuracy).

Inference is bit-for-bit net4's own code path (tactical guard, 1-ply
ranking, depth-(D-1) top-K=2 beam refutation, identical `_max_leaf_calls()`/
manifest flops formula -- net16's params/FEATURE_DIM are identical to
net1's, so flops_per_move is computed with net4's exact formula) -- only
artifact LOADING (`load_compressed` instead of `load_npz`) and hence
`size_bytes` (net16's real 2867B artifact) differ from net4.
"""
from __future__ import annotations

from app.agents.net4 import Net4Agent, DEPTH, K, LOSS_THRESH
from app.agents.net16 import DEFAULT_ARTIFACT as NET16_ARTIFACT
from app.agents.mlp import load_compressed


class Net16BAgent(Net4Agent):
    name = "neurofour-net16b"
    kind = "search"

    def __init__(self, artifact_path: str = NET16_ARTIFACT, depth: int = DEPTH,
                 k: int = K, loss_thresh: float = LOSS_THRESH):
        # deliberately NOT calling Net4Agent.__init__ (it loads via load_npz)
        # -- duplicate its trivial setup and load with load_compressed
        # instead. select_move/_search/_value/manifest inherited verbatim.
        import os

        from app.agents.encode import encode, FEATURE_DIM

        self.artifact_path = artifact_path
        self.depth = depth
        self.k = k
        self.loss_thresh = loss_thresh
        self._encode = encode
        self.feature_dim = FEATURE_DIM
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"neurofour-net16b artifact missing: {artifact_path}. "
                f"Run scripts/compress_net1.py first (net16b reuses net16's "
                f"compressed artifact)."
            )
        self._w = load_compressed(artifact_path)
