"""Self-test for `tests/_oracle_scan.py` (gen-9 T1). Proves the transitive
scan can actually FAIL -- the whole point of replacing the old single-file
scan, which could never trip on the laundering pattern it exists to catch.

Three scenarios, each plants temporary `.py` files under `app/agents/`
(deleted in a `finally`, never imported via Python's import machinery -- the
scanner only `ast.parse`s source text, so no bytecode/`sys.modules` cleanup
is needed):

1. A helper module that re-exports `Solver`/`solve` from `app.solver.solver`,
   and an agent module that imports ONLY the helper (never mentions
   "solver" anywhere in its own source). The OLD single-file scan (`"solver"
   not in <agent's own import names>`) would have PASSED this -- reproduced
   here inline to prove the gap. The NEW transitive scan must FAIL it.
2. The real, currently-registered/experimental nets (`net13`, `net14`,
   `net4`) must PASS cleanly.
3. A planted module that reads `bench_data/sealed.jsonl` via `open()` at
   inference time must FAIL.
"""
import os

import pytest

from tests._oracle_scan import (
    assert_no_oracle_access,
    transitive_findings,
    OracleAccessError,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENTS_DIR = os.path.join(_ROOT, "app", "agents")

_HELPER_NAME = "_tmp_oracle_selftest_helper"
_AGENT_NAME = "_tmp_oracle_selftest_agent"
_BENCH_NAME = "_tmp_oracle_selftest_benchread"

_HELPER_SRC = '''"""Planted by test_oracle_scan_selftest.py -- NOT a real agent.
Re-exports the oracle behind an innocuously-named helper, the exact
laundering pattern the transitive scan exists to catch."""
from app.solver.solver import Solver, solve


def make_solver():
    return Solver()
'''

_AGENT_SRC = '''"""Planted by test_oracle_scan_selftest.py -- NOT a real agent.
Never mentions "solver" anywhere in ITS OWN source -- only imports a
helper module that does. The single-file scan this replaces would have
passed this file."""
from app.agents._tmp_oracle_selftest_helper import make_solver


class TmpLaunderedAgent:
    name = "tmp-laundered"

    def select_move(self, board):
        s = make_solver()
        sol = s._negamax(board.cur, board.mask, board.n, -1000, 1000)
        return 0 if sol else 0
'''

_BENCH_SRC = '''"""Planted by test_oracle_scan_selftest.py -- NOT a real agent.
Reads the sealed label set directly at inference time."""


def cheat_peek():
    with open("bench_data/sealed.jsonl", "r", encoding="utf-8") as f:
        return f.read()
'''


def _legacy_single_file_scan_would_pass(agent_src: str) -> bool:
    """Reproduces the OLD (pre-gen-9) `test_source_has_no_solver_or_oracle_
    access` body EXACTLY: single-file ast.parse, "solver" substring check
    over ONLY this file's own import statements. Returns True if the old
    check would have let `agent_src` through."""
    import ast as _ast
    tree = _ast.parse(agent_src)
    try:
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    assert "solver" not in alias.name
            if isinstance(node, _ast.ImportFrom):
                mod = node.module or ""
                assert "solver" not in mod
        return True
    except AssertionError:
        return False


@pytest.fixture
def _planted_modules():
    written = []

    def _plant(name, src):
        path = os.path.join(_AGENTS_DIR, f"{name}.py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
        written.append(path)
        return path

    yield _plant

    for path in written:
        if os.path.exists(path):
            os.remove(path)


def test_transitive_scan_fails_on_laundered_solver_import(_planted_modules):
    """The core gap this whole module closes: an agent importing the oracle
    through an intermediate helper module."""
    _planted_modules(_HELPER_NAME, _HELPER_SRC)
    _planted_modules(_AGENT_NAME, _AGENT_SRC)

    # 1. the OLD single-file scan, applied to the planted agent's OWN
    #    source only, would have PASSED it (the gap this replaces).
    assert _legacy_single_file_scan_would_pass(_AGENT_SRC), (
        "test setup invariant broken: the planted agent's own source should "
        "read clean under the old single-file 'solver' substring check -- "
        "if this assertion fails, the test no longer demonstrates the gap")

    # 2. the NEW transitive scan must FAIL it.
    findings = transitive_findings(f"app.agents.{_AGENT_NAME}")
    assert findings, (
        "transitive scan did not catch the laundered Solver import routed "
        "through a helper module -- the exact regression this test exists "
        "to prevent")
    assert any("Solver" in f or "solve" in f or "_negamax" in f
               for f in findings), findings

    with pytest.raises(OracleAccessError):
        assert_no_oracle_access(f"app.agents.{_AGENT_NAME}")


def test_transitive_scan_passes_on_real_registered_and_experimental_nets():
    """net13 (registered), net14 (experimental, gen-9 T3 candidate), and
    net4 (registered) must all read clean: their only `app.solver` symbols,
    anywhere in the transitive closure, are the two allowlisted pure
    board-math helpers (`_compute_winning_position`, `_possible`), reached
    via `app.agents.encode`."""
    for mod in ("app.agents.net13", "app.agents.net14", "app.agents.net4"):
        findings = transitive_findings(mod)
        assert findings == [], f"{mod}: unexpected findings: {findings}"
        assert_no_oracle_access(mod)   # must not raise


def test_transitive_scan_fails_on_planted_bench_data_read(_planted_modules):
    _planted_modules(_BENCH_NAME, _BENCH_SRC)

    findings = transitive_findings(f"app.agents.{_BENCH_NAME}")
    assert findings, "did not catch a planted open('bench_data/...') read"
    assert any("bench_data" in f or "label" in f for f in findings), findings

    with pytest.raises(OracleAccessError):
        assert_no_oracle_access(f"app.agents.{_BENCH_NAME}")


def test_planted_modules_are_cleaned_up_after_each_test():
    """Belt-and-suspenders: confirm no planted file survives past its own
    test (the fixture's `finally`-equivalent teardown already guarantees
    this; this test just asserts the directory is clean at collection-
    adjacent time so a future test run can't accidentally inherit stale
    planted modules)."""
    for name in (_HELPER_NAME, _AGENT_NAME, _BENCH_NAME):
        path = os.path.join(_AGENTS_DIR, f"{name}.py")
        assert not os.path.exists(path), f"leftover planted module: {path}"
