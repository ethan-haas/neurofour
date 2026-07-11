"""Regression guard for the `/analyze` perf fix (app/main.py::_bounded_analyze).

`_bounded_analyze` used to re-run a fresh, unshared depth-9 negamax for EVERY
legal column of a near-empty (< EXACT_SOLVE_MIN_PLY stones) position -- up to
7 full independent tree searches per request, ~8-50s depending on the
position. The fix (`app/main.py::_negamax_tt`) shares one transposition
table across all 7 per-column root searches in a single request, which is a
pure SPEED optimization: it must return byte-identical `/analyze` output to
the old per-column-fresh-search code for every position, never a cheaper-but-
different answer (lower depth, sampled columns, cached approximations, etc.
are all explicitly forbidden by the task -- see the regression history note
in `_negamax_tt`'s docstring in app/main.py).

`analyze_regression_golden.json` is a captured snapshot of `/analyze`'s FULL
JSON response for 34 distinct positions spanning plies 0..20 (42 records:
positions below ply 14 -- the `_bounded_analyze` path this fix touches -- in
"scored" mode only, since mode is ignored on that path; positions at/above
ply 14 -- the untouched exact-solver path -- in both "scored" and "value"
mode), captured against the pre-fix code (the per-column-fresh-search
`MinimaxAgent(9)._negamax` loop, git commit 4514190). This test re-runs the
SAME requests against the CURRENT code and asserts the response body is
exactly equal, position by position -- so any future change to the bounded
analysis path that alters even one column's score, `optimal_cols`, or
`best_col` for any of these positions fails loudly here.
"""
from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient

from app.main import app

_GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "analyze_regression_golden.json")

client = TestClient(app)


def _load_golden() -> list[dict]:
    with open(_GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_golden_fixture_has_enough_coverage():
    golden = _load_golden()
    positions = {tuple(rec["moves"]) for rec in golden}
    plies = {rec["ply"] for rec in golden}
    assert len(positions) >= 30, "need >=30 distinct positions for the regression proof"
    assert min(plies) == 0 and max(plies) >= 20, "must span plies 0..20"
    # the bounded (non-exact, TT-optimized) path this fix touches
    assert any(rec["ply"] < 14 for rec in golden)
    # the untouched exact-solver path, for breadth
    assert any(rec["ply"] >= 14 for rec in golden)


def test_analyze_output_unchanged_for_golden_positions():
    golden = _load_golden()
    assert len(golden) >= 30
    mismatches = []
    for rec in golden:
        r = client.post("/analyze", json={"board": rec["moves"], "mode": rec["mode"]})
        assert r.status_code == rec["status"], (rec["moves"], rec["mode"], r.status_code)
        body = r.json()
        if body != rec["body"]:
            mismatches.append((rec["ply"], rec["moves"], rec["mode"]))
    assert not mismatches, f"/analyze output changed for {len(mismatches)} golden position(s): {mismatches}"
