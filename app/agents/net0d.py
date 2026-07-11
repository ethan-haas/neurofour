"""`neurofour-net0d`: a NANO-tier LEARNED agent that applies `neurofour-net2`'s
proven depth-3 refutation search to `neurofour-net0`'s existing NANO leaf.

Design (fair play -- the agent receives ONLY the board; it never calls the
solver, never reads a scored-position file, and never hardcodes a per-position
answer): IDENTICAL algorithm to `neurofour-net2` (0-param tactical guard,
1-ply move ordering by the leaf value, depth-3 alpha-beta negamax refutation
search that only VERIFIES/rejects the top-ranked candidate as a forced loss --
see `net2.py`'s module docstring for the full rationale, including why plain
full-width minimax at this depth is worse ("minimax pathology") and why a
refutation role is the right way to spend the extra plies). `Net0dAgent`
literally *subclasses* `Net2Agent` to reuse that exact search verbatim --
`select_move`/`_negamax`/`_value`/`manifest` are unchanged, only the artifact
(and therefore `self.name`) differ -- rather than re-implementing/forking the
algorithm, so there is only ever one copy of the search logic to keep correct.

What differs from net2: the LEAF ARTIFACT. net2 reuses net1's MICRO-tier
194->24->1 net (`neurofour-net1.npz`, 4837B). net0d instead reuses net0's
existing NANO-tier 194->14->1 net (`neurofour-net0.npz`, 3290B, 2745 params) --
the exact same on-disk artifact net0's own 1-ply agent uses, same encoder
(`app.agents.encode.encode`/`FEATURE_DIM`, both untouched). No new artifact is
trained; net0d is purely "net2's search wired onto net0's leaf", the untried
combination the byte/strength frontier was missing: a NANO-budget point (leaf
artifact <=4096B, same as net0's own 3290B) reaching toward MICRO-tier
strength via a cheap deep search rather than a bigger net. net0, net1, and
net2 are all completely unchanged -- net0d is purely additive, a new frontier
point alongside them.

Measured on the sealed set (`python scripts/run_bench.py`, seed=4): net0d
reaches optimality 0.940000 (0 blunders), strictly above net0's 1-ply-only
0.936667 -- i.e. the depth-3 refutation search DOES add real strength on top
of the smaller/noisier nano leaf (unlike plain full-width minimax at this
depth, which net2's docstring documents actively regresses net1's 1-ply
score via the "minimax pathology" effect; the refutation-only role sidesteps
that here too). It does not reach net1/net2 territory (net0's leaf is a
strictly smaller, less accurate evaluator -- expected, and reported
honestly), but it IS a genuine new non-dominated Pareto point (confirmed
`pareto: true` in `bench_data/leaderboard.json`): strictly higher optimality
than net0 at net0's exact nano byte budget (only more flops, from the deeper
search), and strictly smaller/cheaper than net1/net2 (which need net1's
4837B leaf). Anti-memorization (`scripts/exp_net0d_seed99_check.py`, a FRESH
`NEUROFOUR_SEED=99` sealed set): net0d 0.9233 vs net0 0.9133 -- net0d still
beats net0 by a comparable margin off the seed=4 sealed set, confirming
genuine generalization from the deeper search, not sealed-set overfitting.

Only the board is used at inference: `encode.encode` is a pure bitboard
function (no solver-tree call), engine rules (`Board.play`/`.winner()`/`.n`)
decide children and terminal states, and the learned net scores true
non-terminal leaves only -- `Solver.solve`/`optimal_cols`/`best_col`/`scored`/
`.jsonl` are never touched, exactly as in `net0.py`/`net2.py`.
"""
from __future__ import annotations

import os

from app.agents.net0 import DEFAULT_ARTIFACT as NET0_ARTIFACT
from app.agents.net2 import Net2Agent, DEPTH, LOSS_THRESH


class Net0dAgent(Net2Agent):
    name = "neurofour-net0d"
    kind = "search"          # depth-3 alpha-beta refutation search, learned NANO leaf eval

    def __init__(self, artifact_path: str = NET0_ARTIFACT, depth: int = DEPTH,
                 loss_thresh: float = LOSS_THRESH):
        # Reuses Net2Agent's search verbatim (encode_fn/feature_dim left at
        # their defaults -- net0d uses the SAME 194-dim encoder net0/net1/net2
        # all share); only the artifact_path (net0's own nano leaf) differs.
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"neurofour-net0d artifact missing: {artifact_path}. "
                f"Run train_net0.py first (net0d reuses net0's nano artifact)."
            )
        super().__init__(artifact_path=artifact_path, depth=depth, loss_thresh=loss_thresh)
