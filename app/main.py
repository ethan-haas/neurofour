"""NeuroFour FastAPI backend (SPEC §5).

Importable as `app.main:app`. In-memory game store. The solver is reachable only
through `/analyze` (server-side analysis); agents never receive a solver handle.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from typing import Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError
from starlette.responses import JSONResponse

from app.engine.board import Board, WIDTH, CENTER_ORDER, IllegalMove
from app.agents import registry
from app.agents.display import display_info
from app.agents.heuristic_eval import evaluate as _heuristic_evaluate, WIN_SCORE
from app.solver.solver import Solver
from app.neurogolf.config import EXACT_SOLVE_MIN_PLY

app = FastAPI(title="NeuroFour", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _sanitize_non_finite(obj):
    """Recursively replace non-finite floats (NaN/Infinity/-Infinity) with
    ``None`` so the object is safe to pass to ``json.dumps(..., allow_nan=
    False)`` -- the serializer Starlette's ``JSONResponse`` always uses.
    Leaves every other value (ints, finite floats, str, bool, None, nested
    dict/list) completely unchanged."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    """SPEC.md sec.5: a malformed body must always get a 4xx, never a 5xx.

    Root cause this closes (class-wide, not per-endpoint): Python's stdlib
    ``json`` parser accepts the bare literals ``NaN``/``Infinity``/
    ``-Infinity`` as valid tokens (a non-standard extension), so a request
    body like ``{"board":[NaN]}`` parses fine and reaches pydantic. Pydantic
    then correctly REJECTS the value (wrong type for the field) and raises
    ``RequestValidationError`` -- so far, correct, on track for a 422. But
    FastAPI's *default* handler for that exception (the one this function
    replaces) echoes the offending raw input value verbatim into the error
    body via ``jsonable_encoder(exc.errors())``, and Starlette's
    ``JSONResponse`` serializes that body with
    ``json.dumps(..., allow_nan=False)`` -- which *raises* ``ValueError:
    Out of range float values are not JSON compliant`` on a NaN/Infinity
    float. That raise happens inside exception-handling itself, escapes
    uncaught, and Starlette's outermost ServerErrorMiddleware turns it into
    a bare 500. So the intended 422 silently becomes a 500, and the field-
    level type guard (StrictInt, see AnalyzeIn) never even gets a chance to
    matter -- the 500 happens one layer further out, in the error-reporting
    path itself, not in application code.

    This is registered once on the ``app`` (not per-route), so it covers
    the validation-error path for EVERY endpoint that takes a request body
    -- not just ``/analyze`` -- closing the whole class in one place: any
    field on any current or future endpoint that rejects a non-finite float
    is safe, because the value never reaches ``json.dumps`` un-sanitized.
    The behaviour for every *other* validation error (wrong type, missing
    field, extra field, out-of-range int, ...) is byte-for-byte identical
    to FastAPI's stock handler -- only a non-finite float's echoed value
    is swapped for ``None``; the error's ``type``/``loc``/``msg`` (i.e. the
    actionable part of the message) are untouched."""
    errors = jsonable_encoder(exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize_non_finite(errors)},
    )

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEADERBOARD_PATH = os.path.join(_ROOT, "bench_data", "leaderboard.json")

# ---- caches --------------------------------------------------------------- #
_AGENT_CACHE: dict = {}
_SOLVER = Solver()
_GAMES: dict[str, dict] = {}
_EVAL_CACHE: dict[str, dict] = {}

# stats merged onto each /agents row from bench_data/leaderboard.json (the
# same file /leaderboard serves) -- null for any of these when the agent has
# no leaderboard row (never invented, never crashes).
_LEADERBOARD_STAT_KEYS = (
    "optimality", "elo", "latency_ms", "neurogolf_score", "tier", "pareto",
    "over_budget",
)

# Loaded ONCE (lazily, on first use, then cached) rather than per-request --
# see A2. `None` means "not loaded yet"; `{}` is a valid loaded-but-empty result
# (leaderboard.json missing/unbuilt), distinguished by the sentinel below.
_LEADERBOARD_STATS_CACHE: Optional[dict[str, dict]] = None


def _leaderboard_stats_by_name() -> dict[str, dict]:
    """`{agent_name: {optimality, elo, latency_ms, neurogolf_score, tier,
    pareto, over_budget}}`, read from `bench_data/leaderboard.json` exactly
    once per process (module-level cache) and reused for every subsequent
    `/agents` call. Returns `{}` (never raises) if the file doesn't exist or
    fails to parse -- that just means every agent's stats come back `null`,
    not a broken endpoint."""
    global _LEADERBOARD_STATS_CACHE
    if _LEADERBOARD_STATS_CACHE is None:
        stats: dict[str, dict] = {}
        try:
            with open(LEADERBOARD_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for row in data.get("agents", []):
                name = row.get("name")
                if not name:
                    continue
                stats[name] = {k: row.get(k) for k in _LEADERBOARD_STAT_KEYS}
        except (OSError, ValueError):
            stats = {}
        _LEADERBOARD_STATS_CACHE = stats
    return _LEADERBOARD_STATS_CACHE


def _agent(name: str):
    if name not in _AGENT_CACHE:
        try:
            _AGENT_CACHE[name] = registry.make_agent(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown agent: {name}")
    return _AGENT_CACHE[name]


def _winning_line(board: Board):
    """Return list of [row,col] forming the winning 4 (or None)."""
    w = board.winner()
    if w == 0:
        return None
    grid = board.cells()
    from app.engine.board import HEIGHT
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if grid[r][c] != w:
                continue
            for dr, dc in dirs:
                cells = []
                for k in range(4):
                    rr, cc = r + dr * k, c + dc * k
                    if 0 <= rr < HEIGHT and 0 <= cc < WIDTH and grid[rr][cc] == w:
                        cells.append([rr, cc])
                    else:
                        break
                if len(cells) == 4:
                    return cells
    return None


def _state(game_id: str) -> dict:
    g = _GAMES[game_id]
    b: Board = g["board"]
    winner = b.winner()
    if winner:
        status = "won"
    elif b.is_draw():
        status = "draw"
    else:
        status = "in_progress"
    ptm = b.player_to_move()
    side_agent = g["first_agent"] if ptm == 1 else g["second_agent"]
    return {
        "id": game_id,
        "board": b.cells(),
        "key": b.to_key(),
        "moves": g["history"],
        "legal_moves": b.legal_moves(),
        "player_to_move": ptm,
        "to_move_is_agent": side_agent is not None,
        "to_move_agent": side_agent,
        "first_agent": g["first_agent"],
        "second_agent": g["second_agent"],
        "status": status,
        "winner": winner,
        "winning_line": _winning_line(b),
        "num_moves": b.n,
    }


def _parse_board(spec) -> Board:
    """Accept a move-seq string ('3,2,4'), a list of ints, or a key 'mask:cur'."""
    try:
        if isinstance(spec, list):
            return Board.from_moves([int(c) for c in spec])
        if isinstance(spec, str):
            if ":" in spec:
                return Board.from_key(spec)
            return Board.from_moves(spec)
        raise ValueError("unsupported board spec")
    except (IllegalMove, ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"invalid board: {e}")


# ---- schemas -------------------------------------------------------------- #
# Every request model below sets `extra="forbid"` and uses `Strict*` scalar
# types for the same reason `AnalyzeIn` originally did (ESCAPE 2 fix): SPEC.md
# sec.5 requires a malformed body to be rejected 4xx, and an unexpected extra
# field or a silently-coerced scalar (bool->int, numeric-string->int,
# whole-number-float->int) is a malformed body that pydantic's *lax* mode
# would otherwise let through untouched. Consistency matters here specifically
# because the auditor's finding was that two endpoints (`/analyze`'s `board`
# and `/game/{id}/move`'s `col`) disagreed about what counts as a valid
# integer column -- fixing one and leaving the others loose would just move
# the inconsistency rather than close it.
class NewGame(BaseModel):
    # `first_agent`/`second_agent` are plain `str` (pydantic v2's non-strict
    # `str` validator already rejects bool/int/float -- see MoveIn.col's
    # docstring below for why `int` needs `Strict` and `str` does not), and
    # any unrecognised name (including a coerced-looking one) is already
    # validated against `registry.agent_names()` in `game_new()` and 404s --
    # so there is no silent-acceptance hole here to close with `Strict`.
    # `extra="forbid"` is still added for surface-wide consistency: a stray
    # unexpected field should 422 here exactly like it does on /analyze.
    model_config = ConfigDict(extra="forbid")
    first_agent: Optional[str] = None
    second_agent: Optional[str] = None


class MoveIn(BaseModel):
    # DEFECT root cause: `col: int` used pydantic's *lax* int validator, which
    # silently COERCES rather than rejects: `bool` is an `int` subclass in
    # Python (`isinstance(True, int) is True`), so lax-mode `int` accepts
    # `True`/`False` outright (-> 1/0); a numeric string ("3") and a
    # whole-number float (3.0) are likewise lax-mode int coercions. `/analyze`
    # already used `StrictInt` for its board's column elements (ESCAPE 2 fix)
    # and correctly rejects a bool element there -- so the two endpoints
    # disagreed about what an integer column is. `StrictInt` closes the same
    # hole here: only a genuine Python `int` (not `bool`, `str`, or `float`)
    # is accepted; `ge=0, le=WIDTH-1` range-checking is unchanged.
    model_config = ConfigDict(extra="forbid")
    col: StrictInt = Field(..., ge=0, le=WIDTH - 1)


class AnalyzeIn(BaseModel):
    # ESCAPE 2 root-cause fix: `board: object` accepted literally anything,
    # so a nested list ([[3]]), a null element ([null]), or a mixed shape
    # ([1,2,[3]]) sailed through pydantic untouched and only blew up later
    # as an uncaught TypeError inside `int(c)` in `_parse_board` (a 500, not
    # the 4xx SPEC.md sec.5 requires). `Union[str, list[StrictInt]]` rejects
    # those shapes AT VALIDATION (422) before `_parse_board` ever runs.
    # `StrictInt` (not plain `int`) also closes the silent-coercion hole
    # where a float column ([3.5]) was accepted and truncated to `3` instead
    # of being rejected -- pydantic's non-strict `int` coerces float/bool/
    # numeric-string to int, which is exactly the "looseness" the audit
    # flagged. `extra="forbid"` rejects unexpected top-level fields for the
    # same reason: fail closed at validation, not deep inside board-parsing
    # code. A move-seq string ("3,2,4") and a canonical key string
    # ("mask:cur") both still match the `str` branch and are otherwise
    # unaffected -- only the list-of-elements shape got stricter.
    model_config = ConfigDict(extra="forbid")
    board: Union[str, list[StrictInt]] = ""
    # "scored" (mate-distance-aware) is the oracle used everywhere else (sealed
    # labels, the `perfect` agent) -- default to it so /analyze matches. "value"
    # remains selectable to get a sign-only (+1/0/-1) `value`/`per_col` display,
    # but never changes which move(s) are reported as optimal (see analyze()).
    mode: str = "scored"


class EvaluateIn(BaseModel):
    # `agent: str` needs no `Strict` wrapper for the same reason as
    # `NewGame`'s fields (plain `str` already rejects bool/int/float in
    # pydantic v2 lax mode, and any coerced-looking name still 404s against
    # `registry.agent_names()` in `evaluate()`). `extra="forbid"` added for
    # surface-wide consistency with the other three models.
    model_config = ConfigDict(extra="forbid")
    agent: str


# ---- endpoints ------------------------------------------------------------ #
@app.get("/health")
def health():
    return {"status": "ok", "agents": registry.agent_names(),
            "leaderboard": os.path.exists(LEADERBOARD_PATH)}


@app.get("/agents")
def agents():
    stats_by_name = _leaderboard_stats_by_name()
    out = []
    for name in registry.agent_names():
        m = _agent(name).manifest().to_dict()
        display_name, subtitle = display_info(name)
        m["display_name"] = display_name
        m["subtitle"] = subtitle
        # merge strength/cost stats from the leaderboard; null (never
        # invented/crashed) when this agent has no leaderboard row.
        row_stats = stats_by_name.get(name)
        for key in _LEADERBOARD_STAT_KEYS:
            m[key] = row_stats.get(key) if row_stats is not None else None
        out.append(m)
    return {"agents": out}


@app.post("/game/new")
def game_new(body: NewGame):
    for who in (body.first_agent, body.second_agent):
        if who is not None and who not in registry.agent_names():
            raise HTTPException(status_code=404, detail=f"unknown agent: {who}")
    gid = uuid.uuid4().hex[:12]
    _GAMES[gid] = {
        "board": Board.empty(),
        "first_agent": body.first_agent,
        "second_agent": body.second_agent,
        "history": [],
    }
    return _state(gid)


@app.get("/game/{game_id}")
def game_get(game_id: str):
    if game_id not in _GAMES:
        raise HTTPException(status_code=404, detail="unknown game")
    return _state(game_id)


@app.post("/game/{game_id}/move")
def game_move(game_id: str, body: MoveIn):
    if game_id not in _GAMES:
        raise HTTPException(status_code=404, detail="unknown game")
    g = _GAMES[game_id]
    b: Board = g["board"]
    if b.is_terminal():
        raise HTTPException(status_code=400, detail="game is over")
    ptm = b.player_to_move()
    side_agent = g["first_agent"] if ptm == 1 else g["second_agent"]
    if side_agent is not None:
        raise HTTPException(status_code=400,
                            detail="it is an agent's turn; call /agent-move")
    if not b.can_play(body.col):
        raise HTTPException(status_code=400, detail=f"illegal move: column {body.col}")
    g["board"] = b.play(body.col)
    g["history"].append(body.col)
    return _state(game_id)


@app.post("/game/{game_id}/agent-move")
def game_agent_move(game_id: str):
    if game_id not in _GAMES:
        raise HTTPException(status_code=404, detail="unknown game")
    g = _GAMES[game_id]
    b: Board = g["board"]
    if b.is_terminal():
        raise HTTPException(status_code=400, detail="game is over")
    ptm = b.player_to_move()
    side_agent = g["first_agent"] if ptm == 1 else g["second_agent"]
    if side_agent is None:
        raise HTTPException(status_code=400, detail="it is a human's turn")
    col = _agent(side_agent).select_move(b)
    if not b.can_play(col):
        raise HTTPException(status_code=500, detail="agent produced an illegal move")
    g["board"] = b.play(col)
    g["history"].append(col)
    st = _state(game_id)
    st["agent_move"] = col
    st["agent"] = side_agent
    return st


# exact-solve is fast for positions with enough stones; near-empty off-book
# positions are analysed with a depth-limited search so the endpoint never hangs.
# Shared with PerfectAgent.FAST_PLY (app/agents/baselines.py) -- both must stay
# equal so the exact-solve floor is one single source of truth (config.py).
ANALYZE_EXACT_PLY = EXACT_SOLVE_MIN_PLY


BOUNDED_ANALYZE_DEPTH = 9  # same fixed depth MinimaxAgent(9) used before this fix
_CENTER_RANK = {c: i for i, c in enumerate(CENTER_ORDER)}


def _negamax_tt(board, depth: int, alpha: int, beta: int, tt: dict) -> int:
    """Same recursion, terminal/base cases, static eval and CENTER_ORDER move
    order as ``MinimaxAgent._negamax`` (app/agents/baselines.py) -- byte-for-
    byte the same algorithm -- plus a transposition table shared across all
    per-column root searches issued by one ``_bounded_analyze`` call.

    Why this is safe (preserves the exact same return value as the
    un-cached recursion, for every call): ``tt`` uses the standard fail-soft
    alpha-beta bookkeeping (flag 0 = exact, -1 = upper bound, +1 = lower
    bound) keyed by ``(board.to_key(), depth)`` -- the identical scheme
    ``app/solver/solver.py::Solver._negamax`` already uses for the EXACT
    solver (see there for the same pattern proven correct). A cached entry
    only ever narrows the caller's [alpha, beta) window or short-circuits
    when that narrowed window is already empty; it never fabricates a value
    the plain recursion wouldn't also have produced. ``board.to_key()`` is
    the mirror-normalised key, which is valid here because ``evaluate()``
    (app/agents/heuristic_eval.py) sums over the full, mirror-symmetric set
    of 4-in-a-row windows plus a center-column bonus on the mirror-fixed
    center column -- i.e. ``evaluate(b) == evaluate(mirror(b))`` always, so
    folding mirror images together in the TT cannot merge two positions
    that could ever score differently.

    Why it's fast: connect-4 has massive move-order transposition (playing
    the same multiset of columns in a different order reaches the same
    bitboard whenever no intermediate move was itself a win), and the
    ORIGINAL per-column loop searched each of up to 7 columns from a fresh,
    unshared search -- throwing away every one of those shared subtrees.
    Sharing one ``tt`` dict across the whole request lets column 2's search
    reuse work already done while searching column 1, etc.
    """
    w = board.winner()
    if w != 0:
        return -WIN_SCORE - depth        # prefer slower losses / faster wins
    if board.n >= 42:
        return 0
    if depth == 0:
        return _heuristic_evaluate(board)

    key = (board.to_key(), depth)
    entry = tt.get(key)
    alpha0 = alpha
    if entry is not None:
        val, flag = entry
        if flag == 0:
            return val
        elif flag == 1:       # lower bound
            if val > alpha:
                alpha = val
        else:                  # upper bound
            if val < beta:
                beta = val
        if alpha >= beta:
            return val

    best = -(WIN_SCORE * 10)
    for c in CENTER_ORDER:
        if not board.can_play(c):
            continue
        child = board.play(c)
        score = -_negamax_tt(child, depth - 1, -beta, -alpha, tt)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if best <= alpha0:
        tt[key] = (best, -1)      # upper bound
    elif best >= beta:
        tt[key] = (best, 1)       # lower bound
    else:
        tt[key] = (best, 0)       # exact
    return best


def _bounded_analyze(board) -> dict:
    """Depth-limited (non-exact) analysis for near-empty off-book positions.

    Not the audited exact path, but already best-effort mate-distance-aware:
    the search scores a terminal win/loss as ``-WIN_SCORE - remaining_depth``,
    so a win/loss found with more depth budget left (i.e. discovered sooner)
    scores strictly better than one found deeper -- the same "prefer fastest
    mate / slowest loss" ordering the exact scored solver uses, just
    heuristic-bounded to this search's horizon rather than exact. A true
    exact fix here would mean running the full solver on every near-empty
    position, defeating the point of this fast path (SPEC requires the exact
    solve stay off near-empty positions to avoid a multi-minute pure-Python
    opening solve).

    Performance: one shared transposition table (``_negamax_tt``) serves all
    7 per-column root searches in this single request instead of each column
    re-exploring the whole depth-9 tree from scratch -- see `_negamax_tt`'s
    docstring for why this returns byte-identical values to the old
    per-column-fresh-search version, just far fewer redundant node visits.
    """
    tt: dict = {}
    per = {}
    for c in board.legal_moves():
        if board.winning_move(c):
            per[c] = 999999
            continue
        child = board.play(c)
        per[c] = -_negamax_tt(child, BOUNDED_ANALYZE_DEPTH - 1, -(10 ** 9), 10 ** 9, tt)
    best = max(per.values())
    optimal = sorted(c for c, v in per.items() if v == best)
    best_col = min(optimal, key=lambda c: _CENTER_RANK[c])
    return {"per_col": {str(k): int(v) for k, v in per.items()},
            "optimal_cols": optimal, "best_col": best_col}


@app.post("/analyze")
def analyze(body: AnalyzeIn):
    board = _parse_board(body.board)
    if board.is_terminal():
        return {
            "terminal": True,
            "winner": board.winner(),
            "is_draw": board.is_draw(),
            "value": None, "optimal_cols": [], "best_col": None, "per_col": {},
        }
    mode = body.mode if body.mode in ("value", "scored") else "scored"
    from app.solver.solver import _load_book
    booked = board.to_key() in _load_book()
    if board.n < ANALYZE_EXACT_PLY and not booked:
        # avoid a multi-minute pure-Python opening solve
        approx = _bounded_analyze(board)
        return {
            "terminal": False,
            "player_to_move": board.player_to_move(),
            "value": None,
            "optimal_cols": approx["optimal_cols"],
            "best_col": approx["best_col"],
            "per_col": approx["per_col"],
            "mode": "depth-limited",
            "exact": False,
        }
    sol = _SOLVER.solve(board, mode=mode)
    # `optimal_cols`/`best_col` MUST always be the scored (mate-distance-aware)
    # argmax -- the same oracle used by the sealed labels and the `perfect`
    # agent -- regardless of which `mode` the caller asked for. A win position
    # then lists only the fastest-mate winning move(s); a losing position lists
    # only the slowest-loss move(s); a draw lists the drawing move(s). The
    # `value`/`per_col` readout still honours the requested `mode` (sign-only
    # for "value", scored ints for "scored") -- that's a *display* choice and
    # doesn't affect which move(s) are reported as optimal.
    scored_sol = sol if mode == "scored" else _SOLVER.solve(board, mode="scored")
    return {
        "terminal": False,
        "player_to_move": board.player_to_move(),
        "value": sol.value,
        "optimal_cols": scored_sol.optimal_cols,
        "best_col": scored_sol.best_col,
        "per_col": {str(k): v for k, v in sol.per_col.items()},
        "mode": mode,
        "exact": True,
    }


@app.get("/leaderboard")
def leaderboard():
    if not os.path.exists(LEADERBOARD_PATH):
        raise HTTPException(status_code=404,
                            detail="leaderboard not built; run scripts/run_bench.py")
    with open(LEADERBOARD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Inject display_name/subtitle at serve time -- the committed
    # bench_data/leaderboard.json file (a derived artifact checked by
    # run_bench.py --check) is never rewritten on disk; this only decorates
    # the in-memory response for each request.
    for row in data.get("agents", []):
        name = row.get("name")
        if not name:
            continue
        display_name, subtitle = display_info(name)
        row["display_name"] = display_name
        row["subtitle"] = subtitle
    return data


@app.post("/evaluate")
def evaluate(body: EvaluateIn):
    name = body.agent
    if name not in registry.agent_names():
        raise HTTPException(status_code=404, detail=f"unknown agent: {name}")
    if name in _EVAL_CACHE:
        return _EVAL_CACHE[name]
    from app.neurogolf import strength, cost, positions
    sealed_path = os.path.join(_ROOT, "bench_data", "sealed.jsonl")
    if not os.path.exists(sealed_path):
        raise HTTPException(status_code=404, detail="sealed set not generated")
    rows = positions.load_set(sealed_path)
    ag = _agent(name)
    sc = strength.score(ag, rows)
    co = cost.measure(ag, rows)
    card = {"agent": name, "strength": sc.to_dict(), "cost": co.to_dict(),
            "manifest": ag.manifest().to_dict()}
    _EVAL_CACHE[name] = card
    return card
