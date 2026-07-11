"""Transitive oracle/solver-access + label-access scanner (gen-9 T1).

**Why this exists.** Every pre-gen-9 `test_source_has_no_solver_or_oracle_access`
test did a SINGLE-FILE `ast.parse` of ONE agent module and asserted
`"solver" not in <imported module name>`. That is not transitive: every
learned net (`net0`/`net1`/`net2`/`net4`/`net5`/.../`net13`/`net14`) imports
`app.agents.encode`, and `encode.py` itself does
`from app.solver.solver import _compute_winning_position, _possible` --
legitimate pure board-math helpers (win-detection / possible-moves bitboard
ops), not oracle access. Because every agent module's OWN source is clean of
the literal substring "solver", the old single-file scan was *already*
routed around by the repo's own established pattern (route the solver import
through a shared helper module) -- a future agent could hide a genuine
`Solver.solve()` call behind ANY new helper module (not just `encode.py`) and
the old check would stay green forever. This module closes that hole by
walking the full `app.**` import graph reachable from an agent module.

**Design.**
  - Starting from an agent's dotted module name (e.g. ``"app.agents.net13"``),
    resolve each module to its source file under the repo root and `ast.parse`
    it. Recurse into every `app.**` module reached via `import` / `from ...
    import ...`; stop at the boundary (stdlib / numpy / anything not under
    `app.`) -- we do not need to audit numpy's own source.
  - **Allowlist** exactly two symbols as pure board math, importable from
    `app.solver.solver` (or re-exported through `app.agents.encode`) anywhere
    in the closure: `_compute_winning_position`, `_possible`. Importing ANY
    other name from `app.solver.**` anywhere in the transitive closure is a
    FAILURE.
  - **Forbidden names**, anywhere in the closure (import alias, `as`-binding,
    or bare `Name`/`Attribute` reference): `Solver`, `solve`, `solve_best`,
    `_negamax`, `_weak`, `_weak_move`, `get_solver`, `_book_lookup`,
    `_load_book`. **Exception, deliberate and necessary**: a bare `Name`/
    `Attribute` reference is exempted iff the SAME module also locally
    DEFINES a function/method of that exact name (e.g. `net13.py`/`net14.py`/
    `net2.py`/`net0d.py`/`net11.py`/`net12.py` each implement their OWN
    from-scratch alpha-beta search whose recursive helper is, coincidentally,
    also named `_negamax` -- `self._negamax(...)` there calls the module's
    own method, not the solver's). This causes **zero coverage loss**: the
    only way to *reach* a real `Solver._negamax`/`._weak`/`._weak_move`/
    `.solve_best`/`._book_lookup`/`._load_book` at all is to first obtain a
    `Solver` instance or `get_solver()` handle, which requires an
    `ImportFrom`/`Import` of `Solver`/`get_solver`/`app.solver.*` SOMEWHERE
    in the closure -- and that import is flagged unconditionally, with no
    locally-defined-name exemption, by the ImportFrom/Import checks above
    (alias-proof: `import ... as X` and `from ... import Solver as X` are
    both checked against the ORIGINAL imported name, not the local binding).
    A local decoy `def _negamax(...):  pass` cannot launder an
    ELSEWHERE-real `from app.solver.solver import Solver` line -- that line
    still trips on its own, in whichever module it appears, without
    consulting local defs at all. Also flags **dynamic** solver imports
    (`importlib.import_module("app.solver...")`, `__import__("app.solver
    ...")`) which bypass static `Import`/`ImportFrom` node matching
    entirely.
  - **Label-file access**: any `open()`/`Path(...).read_text()`/`.read_bytes()`
    call whose argument contains a string literal referencing `bench_data/`
    or ending in `.jsonl` is a FAILURE (agents must never read the labelled
    position sets at inference time). This is call-site scoped (not a blanket
    string-literal ban) so it does not false-positive on module docstrings
    that legitimately *talk about* `bench_data/sealed.jsonl` in prose (nearly
    every agent module's docstring does, to describe how it was measured).
  - Also preserves the legacy per-module CODE-TOKEN ban (docstrings/comments
    excluded) on the literal tokens `optimal_cols`, `best_col`, `scored`,
    `solve`, `.jsonl` (plus any caller-supplied extra tokens, e.g. net14's
    `.npz`/`load_npz`), for continuity with the tests this replaces.

Raises `OracleAccessError` (an `AssertionError` subclass) listing every
violation found, so a failing test's assertion message is self-explanatory.
"""
from __future__ import annotations

import ast
import io
import os
import tokenize

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALLOWED_SOLVER_SYMBOLS = frozenset({"_compute_winning_position", "_possible"})

FORBIDDEN_NAMES = frozenset({
    "Solver", "solve", "solve_best", "_negamax", "_weak", "_weak_move",
    "get_solver", "_book_lookup", "_load_book",
})

# call-site identifiers that indicate "this Call node touches the
# filesystem for a file-like read" -- used to scope the bench_data/.jsonl
# string-literal check to actual I/O call arguments, not prose.
_FILE_IO_IDENTIFIERS = frozenset({"open", "read_text", "read_bytes", "Path"})

LEGACY_BANNED_TOKENS = ("optimal_cols", "best_col", "scored", "solve", ".jsonl")


class OracleAccessError(AssertionError):
    """Raised when the transitive scan finds a real (or planted) violation."""


def _module_to_file(mod_name: str) -> str | None:
    parts = mod_name.split(".")
    as_module = os.path.join(_ROOT, *parts) + ".py"
    if os.path.isfile(as_module):
        return as_module
    as_pkg = os.path.join(_ROOT, *parts, "__init__.py")
    if os.path.isfile(as_pkg):
        return as_pkg
    return None


def _call_touches_file_io(call: ast.Call) -> bool:
    for n in ast.walk(call.func):
        if isinstance(n, ast.Name) and n.id in _FILE_IO_IDENTIFIERS:
            return True
        if isinstance(n, ast.Attribute) and n.attr in _FILE_IO_IDENTIFIERS:
            return True
    return False


def _string_touches_label_data(s: str) -> bool:
    return ("bench_data" in s) or s.endswith(".jsonl")


def _collect_local_defs(tree: ast.AST) -> set[str]:
    """All function/method/class names DEFINED anywhere in this module (any
    nesting level) -- used to exempt legitimate same-named local
    reimplementations (e.g. a module's own `_negamax` method) from the
    forbidden-name reference check. See module docstring for why this loses
    zero real coverage."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _is_dynamic_solver_import(call: ast.Call) -> bool:
    func = call.func
    is_dunder_import = isinstance(func, ast.Name) and func.id == "__import__"
    is_import_module = isinstance(func, ast.Attribute) and func.attr == "import_module"
    if not (is_dunder_import or is_import_module):
        return False
    for arg in list(call.args) + [kw.value for kw in call.keywords]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if "solver" in arg.value.split("."):
                return True
    return False


def _scan_one_module(mod_name: str, src: str, findings: list[str],
                      to_recurse: list[str]) -> None:
    tree = ast.parse(src, filename=mod_name)
    local_defs = _collect_local_defs(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "app.solver" or mod.startswith("app.solver."):
                for alias in node.names:
                    if alias.name not in ALLOWED_SOLVER_SYMBOLS:
                        findings.append(
                            f"{mod_name}: imports {alias.name!r} from {mod!r} "
                            f"-- only {sorted(ALLOWED_SOLVER_SYMBOLS)} are "
                            f"allowlisted pure board-math helpers")
                continue
            if mod.startswith("app."):
                to_recurse.append(mod)
            for alias in node.names:
                bound = alias.asname or alias.name
                if alias.name in FORBIDDEN_NAMES or bound in FORBIDDEN_NAMES:
                    findings.append(
                        f"{mod_name}: imports forbidden symbol "
                        f"{alias.name!r} from {mod!r}")

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "solver" in alias.name.split("."):
                    findings.append(
                        f"{mod_name}: raw `import {alias.name}` touches the "
                        f"solver package")
                if alias.name.startswith("app."):
                    to_recurse.append(alias.name)
                bound = alias.asname or alias.name.split(".")[0]
                if bound in FORBIDDEN_NAMES:
                    findings.append(
                        f"{mod_name}: imports forbidden symbol via "
                        f"`import {alias.name}`")

        elif isinstance(node, (ast.Name, ast.Attribute)):
            ident = node.id if isinstance(node, ast.Name) else node.attr
            if ident in FORBIDDEN_NAMES and ident not in local_defs:
                findings.append(
                    f"{mod_name}: references forbidden identifier {ident!r} "
                    f"(line {node.lineno}), not locally defined -- must come "
                    f"from an external (solver) object")

        elif isinstance(node, ast.Call):
            if _is_dynamic_solver_import(node):
                findings.append(
                    f"{mod_name}: dynamic import of the solver package "
                    f"(line {node.lineno})")
            if _call_touches_file_io(node):
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    for sub in ast.walk(arg):
                        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                            if _string_touches_label_data(sub.value):
                                findings.append(
                                    f"{mod_name}: file I/O touches label/eval "
                                    f"data {sub.value!r} (line {node.lineno})")


def _legacy_token_scan(mod_name: str, src: str, findings: list[str],
                        extra_banned_tokens: tuple[str, ...]) -> None:
    code_tokens = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL,
                        tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                        tokenize.ENCODING):
            continue
        code_tokens.append(tok.string)
    code_only = " ".join(code_tokens)
    for banned in LEGACY_BANNED_TOKENS + tuple(extra_banned_tokens):
        if banned in code_only:
            findings.append(
                f"{mod_name}: forbidden token {banned!r} in executable code")


def transitive_findings(entry_module: str,
                         extra_banned_tokens: tuple[str, ...] = ()) -> list[str]:
    """Walk the `app.**` import closure reachable from `entry_module` and
    return a list of human-readable violation strings (empty = clean)."""
    findings: list[str] = []
    visited: set[str] = set()
    queue = [entry_module]

    while queue:
        mod_name = queue.pop()
        if mod_name in visited:
            continue
        visited.add(mod_name)
        path = _module_to_file(mod_name)
        if path is None:
            continue  # not a local app.** module (stdlib/3rd-party) -- boundary
        src = open(path, "r", encoding="utf-8").read()

        to_recurse: list[str] = []
        _scan_one_module(mod_name, src, findings, to_recurse)

        # legacy code-token ban only applies to the entry module itself (the
        # per-net tests this replaces only ever scanned their own file).
        if mod_name == entry_module:
            _legacy_token_scan(mod_name, src, findings, extra_banned_tokens)

        for m in to_recurse:
            if m not in visited:
                queue.append(m)

    return findings


def assert_no_oracle_access(entry_module: str,
                             extra_banned_tokens: tuple[str, ...] = ()) -> None:
    """Assert `entry_module` (and everything it transitively imports under
    `app.**`) is free of solver/oracle access and label-file reads. Raises
    `OracleAccessError` with every violation listed if not."""
    findings = transitive_findings(entry_module, extra_banned_tokens)
    if findings:
        raise OracleAccessError(
            f"transitive oracle-access scan found {len(findings)} "
            f"violation(s) starting from {entry_module!r}:\n  " +
            "\n  ".join(findings))
