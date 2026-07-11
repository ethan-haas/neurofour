"""`neurofour-net15`: net1's exact 1-ply value-search structure, but the
leaf value net was trained with a JOINT POLICY-VALUE multitask objective
(MuZero-style auxiliary-loss regularization) instead of net1's plain MSE-only
distillation. See `train_net15.py` for the training-time loss and the
lambda=0 sanity control (which reproduces net1's own artifact bit-for-bit).

At INFERENCE this is bit-for-bit net1's own code path: same tactical guard,
same 1-ply negamax value search, same manifest flops formula (the training-
time policy head is never loaded here -- the exported artifact is a plain
net1-format 4-array npz, so `params`/`size_bytes`/`flops_per_move` are
computed identically to net1's, from the artifact alone).

RESULTS (sess "research-model / joint-policy-value", `scripts/eval_resolution.py`,
dev_big(2000), grain 0.0005 -- see `train_net15.py`'s docstring for the
lambda=0 bit-for-bit sanity control that verified this harness before trusting
any lambda>0 number):

    lambda   net15 opt   vs net2(0.94600)         vs net1(0.94200)
    0.0      0.94200     d=-0.00400 p=0.0433 SIG   d=+0.00000 (== net1, exact)
    0.1      0.93300     d=-0.01300 p=0.0024 SIG   d=-0.00900 p=0.0336 SIG (worse)
    0.3      0.94550     d=-0.00050 p=1.0 not-sig  d=+0.00350 p=0.4347 not-sig
    0.6      0.93650     d=-0.00950 p=0.0233 SIG   d=-0.00550 p=0.2148 not-sig

VERDICT: no lambda in the swept set produces a significant gain over EITHER
net1 (same architecture, isolates the objective) or net2 (the actual frontier
bar). lambda=0.1 and lambda=0.6 are REPRODUCIBLE LOSSES; lambda=0.3 is the
only non-harmful point and is indistinguishable from net1 (WITHIN NOISE, not
a win). A 3-seed sealed(300) noise-floor re-draw (seeds 4/7/11) additionally
flagged the lambda=0.3 net15 (1-ply, no search) as a sign-consistent REAL
NEGATIVE vs net2 there (mean -0.0122, exceeds the 0.0094 empirical floor) --
weaker evidence than the dev_big McNemar (different, smaller, sample) but
pointing the same direction, not the opposite one. Shipped default is
lambda=0.3 (the least-bad point) purely so this is a genuine, checkable
Pareto/negative-result data point, NOT a claimed improvement -- see
`neurofour-net15s` (net2's search structure + this leaf) for the paired
search-plugged variant, which is the closer (still non-significant) result:
dev_big d=+0.00100 (p=0.8937, not sig) vs net2. RIGOROUS NEGATIVE: the joint
policy-value auxiliary loss did not lift the learned Pareto frontier at this
capacity/data scale.
"""
from __future__ import annotations

import os

from app.agents.net1 import Net1Agent

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACT = os.path.join(_HERE, "artifacts", "neurofour-net15.npz")


class Net15Agent(Net1Agent):
    name = "neurofour-net15"
    kind = "search"

    def __init__(self, artifact_path: str = DEFAULT_ARTIFACT):
        super().__init__(artifact_path=artifact_path)
