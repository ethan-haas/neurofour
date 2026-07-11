"""`neurofour-net0b`: `neurofour-net4`'s top-K beam refutation search
(D=3, K=2) applied to `neurofour-net0`'s existing NANO leaf artifact --
the untried "cheap deep search" combination on the nano budget's OWN leaf,
mirroring how `net0d.py` applied net2's FULL-WIDTH depth-3 search to the same
leaf. `Net0bAgent` literally subclasses `Net4Agent` to reuse that exact
search verbatim (`select_move`/`_search`/`_value`/`manifest`/`_max_leaf_calls`
are unchanged, only the artifact -- and therefore `self.name` -- differs),
same pattern `net0d.py` used when it subclassed `Net2Agent`.

Why this exists (gen-9, PART 1 of the task spec): gen-8 measured this exact
config (`Net4Agent(artifact_path=NET0_ARTIFACT, depth=3, k=2)`) in its
E1xE2 sweep but never gave it a name or registered it. Re-measured
independently here on the currently committed artifacts (`python -c` probe,
sealed.jsonl, seed=4):

    agent   size(B)  params  flops      sealed_opt  seed99_opt
    net0       3290    2745       --      0.9367      0.9133
    net0d      3290    2745  1,989,428    0.9400      0.9233
    net0b      3290    2745    875,364    0.9433      0.9233

`net0b` has the SAME on-disk artifact as `net0d` (net0's own leaf, byte-
identical, `size_bytes` equal) and STRICTLY fewer flops (875,364 vs
1,989,428 -- net4's top-K=2 beam vs net0d's full-width depth-3 search, same
saving net4 demonstrated over net2 on the micro leaf). On the committed
seed=4 sealed set net0b's optimality (0.9433) is one position above net0d's
(0.9400) -- with only 300 sealed positions, a 1-position gap (0.00333) is
NOISE per the task's own <=2-position rule, and the fresh `NEUROFOUR_SEED=99`
anti-memorization check (`scripts/exp_net4_seed99_check.py`, already run by
gen-8 and re-confirmed here) shows net0b and net0d are an EXACT TIE at
seed99_opt=0.9233 -- so **the apparent sealed-set strength edge is noise,
not a real generalizing improvement; do not claim net0b is "stronger" than
net0d.** What IS real and reproducible: the flops saving. Per METRIC.md
sec.7 the dominance test is optimality >=, size <=, flops <=, with >=1
strict -- net0b's optimality is >= net0d's (0.9433 >= 0.9400, and they tie
exactly on the fresh seed99 set, so this is never a regression), size is
exactly equal, and flops is strictly and substantially lower. That already
satisfies the mechanical dominance relation regardless of whether the sealed
delta is noise, so net0b is registered as a genuine new Pareto point: same
byte budget as net0/net0d, same-or-better optimality, materially cheaper
inference -- a legitimate flops-axis win to ship, reported honestly rather
than oversold.

Only the board is used at inference: identical anti-cheat contract to
`net4.py`/`net0d.py` (0-param tactical guard, engine-made children, the
learned net scores true non-terminal leaves only; `Solver.solve`/
`optimal_cols`/`best_col`/`scored`/`.jsonl` are never touched).
"""
from __future__ import annotations

import os

from app.agents.net0 import DEFAULT_ARTIFACT as NET0_ARTIFACT
from app.agents.net4 import Net4Agent, DEPTH, K, LOSS_THRESH


class Net0bAgent(Net4Agent):
    name = "neurofour-net0b"
    kind = "search"          # depth-3 top-K=2 beam refutation search, learned NANO leaf eval

    def __init__(self, artifact_path: str = NET0_ARTIFACT, depth: int = DEPTH,
                 k: int = K, loss_thresh: float = LOSS_THRESH):
        # Reuses Net4Agent's search verbatim (encode_fn/feature_dim left at
        # their defaults -- net0b uses the SAME 194-dim encoder net0/net0d/
        # net1/net2/net4 all share); only the artifact_path (net0's own nano
        # leaf) differs.
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"neurofour-net0b artifact missing: {artifact_path}. "
                f"Run train_net0.py first (net0b reuses net0's nano artifact)."
            )
        super().__init__(artifact_path=artifact_path, depth=depth, k=k,
                          loss_thresh=loss_thresh)
