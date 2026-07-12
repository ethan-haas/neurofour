"""gen-11 T3: bitboard leaf + zugzwang odd/even threat-parity term, for
`neurofour-net14` ONLY (new module -- `heuristic_eval_bb.py` and
`heuristic_eval.py` are left completely untouched, per the task's "put new
leaf code in a NEW module" rule; the `heuristic`/`minimax` baseline agents
and net14's PRE-lever behavior are unaffected).

**Motivation (gen-11 T1 finding).** Classifying net14's 192/6000 pooled
residual errors (see `scripts/diag_net14_gen11_t1.py`) gives BLUNDER=95,
MATE-SPEED-REACHABLE=1, TOO-DEEP=96 -- TOO-DEEP dominates (96 vs 1) and,
more importantly, even its EASIEST case needs an estimated ~56,978 exact
search nodes to resolve (`m_needed`, extrapolated from the position's own
empirically observed effective branching factor), already above
`M_CEILING_ABSOLUTE=39,062` -- the absolute node ceiling ANY node-cost
reduction could ever buy under `FLOP_CAP=5,000,000` even with a
*hypothetical zero-cost leaf* (`MIN_NODE_COST=128`, the common-prefix +
free-exact-resolution floor). So neither more M (T2) nor a cheaper node can
ever close the TOO-DEEP gap -- confirmed structurally, not just empirically
saturated. **The only lever left is leaf QUALITY**, which can only act on the
BLUNDER bucket (95 positions where the search's own leaf value at the
horizon has the WRONG SIGN -- i.e. it misjudges who is actually better off,
not merely how fast the win/loss will land).

**The feature.** Classic Connect-4 zugzwang parity: with a 42-cell (even)
board, if the game fills up move-by-move without any earlier tactical win,
the player who moves FIRST overall ends up placing stones on ODD-numbered
cells (1st, 3rd, 5th, ... stone placed in any column, i.e. row-index 0, 2, 4
zero-indexed -- `board.py`'s row-major bit convention) and the second player
ends up on EVEN-numbered cells (row-index 1, 3, 5). A winning-threat cell
sitting on YOUR favourable parity is a genuine long-term (zugzwang) asset
even when the position is far too deep for exact resolution -- this is
exactly the signal `app/agents/encode.py`'s `_odd_even_counts` block already
aggregates (used by the LEARNED nets' feature vector) but net14's
zero-param static heuristic (`heuristic_eval_bb.py`) does not use at all.

**Bit-native, not the Python-loop `_odd_even_counts`.** `encode.py`'s
`_odd_even_counts` is a 42-cell double-`for` loop (NOT bit-op-shaped, same
convention violation `heuristic_eval.py`'s grid-based `evaluate()` had before
the gen-10 T-lever). This module instead precomputes fixed `ODD_ROW_MASK`/
`EVEN_ROW_MASK` bitboards once at import (all cells at row-index 0,2,4 /
1,3,5 across all 7 columns) and classifies threat bitmasks via `&` +
`.bit_count()` against those two masks -- no grid, no Python loop over
cells.

**Cost accounting -- `my_threats` reused, `opp_threats` is genuinely new.**
`net14.py`'s leaf already computes `my_win = _compute_winning_position(
board.cur, board.mask)` for the free-exact-resolution check BEFORE falling
through to the heuristic; `evaluate_bb_parity` takes that value as a
parameter (`my_threats`) instead of recomputing it, so the ONLY new
`_compute_winning_position` call this module adds is for the OPPONENT side
(`opp_threats`, never computed elsewhere in net14's leaf).

**Op pricing (machine-checked, see `tests/test_net14_flop_honesty.py`'s
updated instrumentation -- every primitive below is individually wrapped,
same convention as `heuristic_eval_bb.py`'s existing `_and`/`_popcount`
wrappers and `app/solver/solver.py`'s `_compute_winning_position` price,
`COMPUTE_WINNING_POSITION_OPS=74`; `&`=1 unit, `.bit_count()`=1 unit,
comparisons/arithmetic=1 unit each):
    1x `_compute_winning_position(opp, mask)` (NEW, opp_threats)   = 74
    2x `&` (my_threats & fav_mask, opp_threats & fav_mask)          = 2
    2x `.bit_count()`                                               = 2
    1x parity-seat branch (`player_to_move()==1`, 1 comparison)     = 1
    1x subtract + 1x multiply (my_fav - opp_fav) * PARITY_WEIGHT    = 2
    TOTAL new ops                                                   = 81
`PARITY_OPS = 120` declared (48% margin over the observed 81, in line with
this codebase's other margins -- see `tests/test_net14_flop_honesty.py` for
the exact machine-summed value, which is what actually gates `OPS_PER_NODE`,
not this hand count).

**Weight selection.** `PARITY_WEIGHT` was swept over {0,2,4,6,8,10,14,20} on
a 1200-position pooled sample at M=2000 (`scripts/exp_net14_parity.py
--weight-sweep`) then the winning weight was re-validated on the FULL
pooled 6000 at flop-cap-equalized M before being adopted here -- not tuned
against `sealed.jsonl` or any committed holdout. Re-run the sweep script
above to reproduce the raw numbers and the full-pooled/sealed validation.
"""
from __future__ import annotations

import app.agents.heuristic_eval_bb as _HEBB
import app.solver.solver as _SOLV
from app.agents.heuristic_eval_bb import WIDTH, HEIGHT, H1

PARITY_WEIGHT = 8   # see module docstring: swept on pooled sample, see report

# ODD_ROW_MASK: all cells at row-index 0,2,4 (zero-indexed; "1st/3rd/5th
# stone placed in a column" -- the parity favouring the player who moves
# first overall). EVEN_ROW_MASK: row-index 1,3,5.
ODD_ROW_MASK = sum((1 << (c * H1 + r)) for c in range(WIDTH) for r in range(HEIGHT) if r % 2 == 0)
EVEN_ROW_MASK = sum((1 << (c * H1 + r)) for c in range(WIDTH) for r in range(HEIGHT) if r % 2 == 1)


def _parity_delta(my_threats: int, opp: int, mask: int, to_move_is_p1: bool) -> int:
    """(my_favourable_threat_count - opp_favourable_threat_count), bit-native.
    `my_threats` is PASSED IN (net14's leaf already computed it as `my_win`
    for the free-exact-resolution check -- reused here at zero extra cost).
    `opp_threats` is computed fresh (the one genuinely new
    `_compute_winning_position` call this module adds). Calls through the
    `_HEBB`/`_SOLV` MODULE objects (not re-imported bare names) so a test's
    monkeypatch of e.g. `heuristic_eval_bb._and` or
    `solver._compute_winning_position` is observed here too -- the same
    indirection `heuristic_eval_bb.py` itself relies on for its own
    instrumentation."""
    opp_threats = _SOLV._compute_winning_position(opp, mask)
    my_fav_mask = ODD_ROW_MASK if to_move_is_p1 else EVEN_ROW_MASK
    opp_fav_mask = EVEN_ROW_MASK if to_move_is_p1 else ODD_ROW_MASK
    my_fav = _HEBB._popcount(_HEBB._and(my_threats, my_fav_mask))
    opp_fav = _HEBB._popcount(_HEBB._and(opp_threats, opp_fav_mask))
    return my_fav - opp_fav


def evaluate_bb_parity(board, my_threats: int) -> int:
    """`heuristic_eval_bb.evaluate_bb()`'s score plus a zugzwang odd/even
    threat-parity term (see module docstring). Static, 0-param, board-only.
    `my_threats` MUST be the caller's already-computed
    `_compute_winning_position(board.cur, board.mask)` (net14's `my_win`) --
    passed in so it is not paid for twice."""
    base = _HEBB.evaluate_bb(board)
    opp = board.mask ^ board.cur
    to_move_is_p1 = board.player_to_move() == 1
    delta = _parity_delta(my_threats, opp, board.mask, to_move_is_p1)
    return base + PARITY_WEIGHT * delta
