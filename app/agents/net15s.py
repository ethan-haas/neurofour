"""`neurofour-net15s`: net2's exact depth-3 alpha-beta refutation search
structure, but plugged with `neurofour-net15`'s leaf value net (trained with
the joint policy-value multitask objective, see `train_net15.py`) instead of
net1's artifact. This is the "search-plugged variant" -- it tests whether the
new leaf net is a better move-ranker/refuter inside net2's proven search
structure, which is the actual bar to beat (net2 is the current best learned
micro agent, not net1's bare 1-ply search).

Inference is bit-for-bit net2's own code path (tactical guard, 1-ply
ranking, depth-(D-1) alpha-beta refutation, identical manifest flops
formula) -- only the artifact_path (hence the loaded value-net weights)
differs from net2's default.

RESULTS (`scripts/eval_resolution.py`, dev_big(2000) vs net2, lambda swept
{0.0,0.1,0.3,0.6} in `train_net15.py`; see `net15.py`'s docstring for the
1-ply-only comparison and the lambda=0 sanity control):

    lambda   net15s opt  vs net2(0.94600)
    0.0      0.94600     d=+0.00000 p=1.0 (== net2 exactly, sanity: net15's
                          artifact at lambda=0 is bit-identical to net1's,
                          so net2's search structure + that leaf == net2)
    0.1      0.93800     d=-0.00800 p=0.0489 SIG (worse)
    0.3      0.94700     d=+0.00100 p=0.8937 not-sig
    0.6      0.94250     d=-0.00350 p=0.4185 not-sig

VERDICT: lambda=0.3 is the best point (+0.001 on dev_big) but McNemar
p=0.8937 is nowhere near significant -- WITHIN NOISE, not a frontier win.
A 3-seed sealed(300) noise-floor re-draw (seeds 4/7/11) agrees: sign not
consistent across seeds (-0.0133, +0.0033, -0.0133), mean -0.0078, verdict
WITHIN NOISE. Shipped default is lambda=0.3 (the only non-harmful point)
purely as a checkable negative-result Pareto data point -- NOT a claimed
improvement over net2.
"""
from __future__ import annotations

import os

from app.agents.net2 import Net2Agent, DEPTH, LOSS_THRESH
from app.agents.net15 import DEFAULT_ARTIFACT as NET15_ARTIFACT


class Net15SAgent(Net2Agent):
    name = "neurofour-net15s"
    kind = "search"

    def __init__(self, artifact_path: str = NET15_ARTIFACT, depth: int = DEPTH,
                 loss_thresh: float = LOSS_THRESH):
        super().__init__(artifact_path=artifact_path, depth=depth, loss_thresh=loss_thresh)
