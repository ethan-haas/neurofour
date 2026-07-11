"""`neurofour-net16s`: net2's exact depth-3 alpha-beta refutation search
structure, but plugged with `neurofour-net16`'s COMPRESSED leaf artifact
(the SAME 2867-byte per-row-int5 + 40%-prune file net16.py loads via
`app.agents.mlp.load_compressed`) instead of net1's plain int8 npz.

This is gen-16's own "search-plugged variant" pattern (see `net15s.py`,
which plugged net15's leaf into net2's search): the question isn't whether
the compressed leaf is within noise of net1 at 1-ply (net16.py already
answered that, dev_big 0.9405, McNemar p=0.72 vs net1) -- it's whether that
SAME already-lossy leaf still holds up once net2's depth-3 refutation search
amplifies whatever ranking errors the compression introduced. Search can
either help (catching tactics the compressed leaf's 1-ply ranking alone
misses, same mechanism that lifts net1 0.9467->net2 0.9567) or hurt (a
noisier evaluator gives deeper search more blind spots to walk into --
net2.py's own "minimax pathology" docstring section documents this exact
failure mode for a full-width maximiser; net16s.py measures whether it
also shows up for a REFUTATION-role search over a QUANTISED evaluator,
which is untested territory -- net2/net15s only ever varied the training
recipe, never the leaf's numeric precision).

Inference is bit-for-bit net2's own code path (tactical guard, 1-ply
ranking, depth-(D-1) alpha-beta refutation, identical manifest flops
formula: net16's params/FEATURE_DIM are IDENTICAL to net1's, dequantisation
happens once at __init__, so flops_per_move is computed with net2's exact
formula, no flops saved by quantisation) -- only artifact LOADING (
`load_compressed` instead of `load_npz`) and hence `size_bytes` (the real
2867B on-disk artifact, not net1's 4837B npz) differ from net2.

No new weight bytes: net16s.py loads the EXACT artifact file net16.py
already ships (`neurofour-net16.b4`), just wrapped in net2's search instead
of net16's 1-ply search. See this module's registration in `registry.py`
and `scripts/eval_resolution.py` output (recorded in the gen-3 coder run
that added this file) for the measured dev_big(2000) optimality + McNemar
vs net2.
"""
from __future__ import annotations

from app.agents.net2 import Net2Agent, DEPTH, LOSS_THRESH
from app.agents.net16 import DEFAULT_ARTIFACT as NET16_ARTIFACT
from app.agents.mlp import load_compressed


class Net16SAgent(Net2Agent):
    name = "neurofour-net16s"
    kind = "search"

    def __init__(self, artifact_path: str = NET16_ARTIFACT, depth: int = DEPTH,
                 loss_thresh: float = LOSS_THRESH):
        # deliberately NOT calling Net2Agent.__init__ (it loads via load_npz,
        # which can't parse net16's hand-rolled compressed container) --
        # duplicate its trivial setup here and load with load_compressed
        # instead. Everything else (select_move, _value, _negamax, manifest)
        # is inherited from Net2Agent verbatim.
        import os

        from app.agents.encode import encode, FEATURE_DIM

        self.artifact_path = artifact_path
        self.depth = depth
        self.loss_thresh = loss_thresh
        self._encode = encode
        self.feature_dim = FEATURE_DIM
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"neurofour-net16s artifact missing: {artifact_path}. "
                f"Run scripts/compress_net1.py first (net16s reuses net16's "
                f"compressed artifact)."
            )
        self._w = load_compressed(artifact_path)
