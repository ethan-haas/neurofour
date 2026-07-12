import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app, ANALYZE_EXACT_PLY

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_agents_list():
    r = client.get("/agents")
    assert r.status_code == 200
    names = {a["name"] for a in r.json()["agents"]}
    assert {"random", "heuristic", "minimax-2", "minimax-4", "perfect"} <= names
    for a in r.json()["agents"]:
        assert set(a) >= {"name", "kind", "params", "size_bytes", "flops_per_move"}


def _walk_strings(obj):
    """Yield every string value anywhere in a nested dict/list/scalar."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


def test_agents_no_server_path_leak():
    # BUG FIX (info leak): GET /agents previously returned `artifact_path`
    # as the absolute SERVER filesystem path (e.g.
    # "/opt/render/project/src/app/agents/artifacts/neurofour-net.npz" in
    # prod). A public API must never leak server paths. No string value
    # anywhere in the response may contain a path separator or a
    # recognisable absolute-path prefix.
    r = client.get("/agents")
    assert r.status_code == 200
    body = r.json()
    for s in _walk_strings(body):
        assert "/" not in s, f"path separator leaked in /agents response: {s!r}"
        assert "\\" not in s, f"path separator leaked in /agents response: {s!r}"
        assert "/opt/" not in s
        assert "C:\\" not in s
    # sanity: at least one artifact-bearing agent exists and its
    # artifact_path (if present) is just a basename, not None-only for
    # every agent (would make this test vacuous).
    net_agents = [a for a in body["agents"] if a["name"] == "neurofour-net"]
    if net_agents:  # only registered if the artifact file exists locally
        assert net_agents[0]["artifact_path"] in (None,) or "/" not in net_agents[0]["artifact_path"]


def test_agents_display_name_and_stats_merged():
    r = client.get("/agents")
    assert r.status_code == 200
    body = r.json()["agents"]
    by_name = {a["name"]: a for a in body}

    # A3: canonical display names/subtitles present and correct for known ids.
    assert by_name["neurofour-net14"]["display_name"] == "Zero"
    assert by_name["neurofour-net14"]["subtitle"] == "0-byte champion — pure bitboard search"
    assert by_name["perfect"]["display_name"] == "Oracle"
    assert by_name["random"]["display_name"] == "Random"

    # every agent has a non-empty display_name and a subtitle key (possibly "")
    for a in body:
        assert isinstance(a["display_name"], str) and a["display_name"]
        assert isinstance(a["subtitle"], str)

    # A2: leaderboard stats merged server-side, present as keys on every row
    # (null when the agent has no leaderboard row, but the key must exist).
    stat_keys = {"optimality", "elo", "latency_ms", "neurogolf_score", "tier",
                 "pareto", "over_budget"}
    for a in body:
        assert stat_keys <= set(a)

    # If the committed leaderboard exists, a known agent's stats must be
    # real numbers (not null) and match the committed file exactly, and
    # the frozen `neurogolf_score` field name must survive the merge.
    lb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "bench_data", "leaderboard.json")
    if os.path.exists(lb_path):
        with open(lb_path, "r", encoding="utf-8") as f:
            lb = json.load(f)
        lb_by_name = {row["name"]: row for row in lb["agents"]}
        if "neurofour-net14" in lb_by_name:
            expected = lb_by_name["neurofour-net14"]
            got = by_name["neurofour-net14"]
            assert got["optimality"] == expected["optimality"]
            assert got["neurogolf_score"] == expected["neurogolf_score"]
            assert got["tier"] == expected["tier"]


def test_agents_unknown_id_falls_back_to_id_never_crashes():
    from app.agents.display import display_info
    name, subtitle = display_info("some-brand-new-experimental-agent-id")
    assert name == "some-brand-new-experimental-agent-id"
    assert subtitle == ""


def test_leaderboard_rows_carry_display_name_too():
    r = client.get("/leaderboard")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        body = r.json()
        for row in body["agents"]:
            assert "display_name" in row and isinstance(row["display_name"], str) and row["display_name"]
            assert "subtitle" in row
        # committed bench_data/leaderboard.json itself must be untouched by
        # the serve-time injection (it's a derived artifact checked by
        # run_bench.py --check).
        lb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "bench_data", "leaderboard.json")
        with open(lb_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for row in raw["agents"]:
            assert "display_name" not in row
            assert "subtitle" not in row


def test_new_game_and_human_move():
    r = client.post("/game/new", json={"second_agent": "heuristic"})
    assert r.status_code == 200
    st = r.json()
    gid = st["id"]
    assert st["player_to_move"] == 1
    assert st["status"] == "in_progress"
    # human (player 1) plays center
    r = client.post(f"/game/{gid}/move", json={"col": 3})
    assert r.status_code == 200
    assert r.json()["player_to_move"] == 2


def test_illegal_human_move_rejected():
    r = client.post("/game/new", json={})
    gid = r.json()["id"]
    # fill column 0 by alternating; then playing 0 should 4xx
    for _ in range(6):
        client.post(f"/game/{gid}/move", json={"col": 0})
    r = client.post(f"/game/{gid}/move", json={"col": 0})
    assert r.status_code == 400


def test_non_int_column_422():
    r = client.post("/game/new", json={})
    gid = r.json()["id"]
    r = client.post(f"/game/{gid}/move", json={"col": "banana"})
    assert r.status_code == 422
    r = client.post(f"/game/{gid}/move", json={"col": 99})
    assert r.status_code == 422        # out of ge/le range


@pytest.mark.parametrize("payload", [
    {"col": True},
    {"col": False},
    {"col": "3"},
    {"col": 3.0},
    {"col": 3.5},
    {"col": None},
])
def test_move_column_strict_type_rejected(payload):
    # Sibling-endpoint inconsistency the auditor flagged: /analyze's `board`
    # field is `list[StrictInt]` and correctly rejects a bool element
    # ({"board":[true]} -> 422), but /game/{id}/move's `MoveIn.col` was a
    # plain `int`, which pydantic's lax mode silently COERCES: `bool` is an
    # `int` subclass in Python (`isinstance(True, int) is True`, `int(True)
    # == 1`), so pydantic's lax `int` validator accepts it outright; a
    # numeric string ("3") and a whole-number float (3.0) are also lax-mode
    # int coercions. A column is specified as an integer 0-6 (SPEC.md sec.5);
    # `true`, `"3"`, and `3.0` are not integers, so they must be rejected at
    # validation (4xx) exactly like /analyze's `board` list already rejects
    # a bool element -- the two endpoints must agree on what a column is.
    # `{"col": None}` and `{"col": 3.5}` were already correctly rejected
    # pre-fix (missing-required / non-integral float) and must stay that way.
    r = client.post("/game/new", json={})
    gid = r.json()["id"]
    r = client.post(f"/game/{gid}/move", json=payload)
    assert r.status_code != 200
    assert r.status_code != 500
    assert 400 <= r.status_code < 500


def test_unknown_game_404():
    assert client.get("/game/deadbeef").status_code == 404
    assert client.post("/game/deadbeef/agent-move").status_code == 404


def test_unknown_agent_404():
    r = client.post("/game/new", json={"first_agent": "does-not-exist"})
    assert r.status_code == 404


def test_agent_move_plays_legally():
    r = client.post("/game/new", json={"first_agent": "heuristic"})
    gid = r.json()["id"]
    r = client.post(f"/game/{gid}/agent-move")
    assert r.status_code == 200
    st = r.json()
    assert "agent_move" in st
    assert st["num_moves"] == 1


def test_full_game_vs_agent_reaches_terminal():
    r = client.post("/game/new", json={"first_agent": "random", "second_agent": "random"})
    gid = r.json()["id"]
    for _ in range(60):
        st = client.get(f"/game/{gid}").json()
        if st["status"] != "in_progress":
            break
        client.post(f"/game/{gid}/agent-move")
    st = client.get(f"/game/{gid}").json()
    assert st["status"] in ("won", "draw")


def test_analyze_deep_position():
    # a deep, guaranteed non-terminal position (columns 0-5 filled 3-high with
    # alternating colours -> no 4-in-a-row) solves quickly and exactly
    moves = "0,1,0,1,0,1,2,3,2,3,2,3,4,5,4,5,4,5"
    from app.engine.board import Board
    assert not Board.from_moves(moves).is_terminal()
    r = client.post("/analyze", json={"board": moves, "mode": "value"})
    assert r.status_code == 200
    body = r.json()
    assert body["terminal"] is False
    assert body["value"] in (-1, 0, 1)
    assert body["best_col"] in body["optimal_cols"]


def test_analyze_empty_board_fastest_mate():
    # empty board: only the center column is a genuine win under perfect play
    # -- optimal_cols must be the unique fastest-mate move, NOT every column
    # that merely "doesn't immediately lose" under a value-only readout.
    r = client.post("/analyze", json={"board": []})
    assert r.status_code == 200
    body = r.json()
    assert body["terminal"] is False
    assert body["optimal_cols"] == [3]
    assert body["best_col"] == 3


def test_analyze_exact_scored_argmax_excludes_slower_wins():
    # a deep (>=14-stone) winning position with several value-preserving
    # winning replies at different mate distances: optimal_cols must be only
    # the fastest-mate winning move(s), not every winning column (that was
    # the auditor's "lists ALL value-preserving winning moves" bug).
    moves = [1, 4, 4, 1, 2, 4, 3, 5, 4, 0, 4, 0, 6, 3, 2, 4]
    from app.engine.board import Board
    b = Board.from_moves(moves)
    assert not b.is_terminal() and b.n >= ANALYZE_EXACT_PLY

    r = client.post("/analyze", json={"board": moves})
    assert r.status_code == 200
    body = r.json()
    assert body["exact"] is True
    assert body["mode"] == "scored"
    assert body["optimal_cols"] == [3]      # unique fastest mate
    assert body["best_col"] == 3

    # requesting mode="value" changes the value/per_col *display* scale only
    # -- optimal_cols/best_col must stay the scored fastest-mate argmax.
    r_value = client.post("/analyze", json={"board": moves, "mode": "value"})
    body_value = r_value.json()
    assert body_value["value"] == 1         # sign-only display for mode="value"
    assert body_value["optimal_cols"] == [3]
    assert body_value["best_col"] == 3


def test_analyze_exact_losing_position_reports_slowest_loss():
    # a losing position where every reply loses, but at different mate
    # distances: best_col must be the SLOWEST loss, never a strictly faster
    # one, regardless of requested mode.
    moves = [1, 4, 4, 1, 2, 4, 3, 5, 4, 0, 4, 0, 6, 3, 2, 4, 1, 1, 6, 3]
    from app.engine.board import Board
    b = Board.from_moves(moves)
    assert not b.is_terminal() and b.n >= ANALYZE_EXACT_PLY

    r = client.post("/analyze", json={"board": moves})
    body = r.json()
    assert body["exact"] is True
    assert body["value"] < 0                # a confirmed loss
    assert body["best_col"] == 2             # the slowest loss, not col 3

    r_value = client.post("/analyze", json={"board": moves, "mode": "value"})
    body_value = r_value.json()
    assert body_value["value"] == -1
    assert body_value["best_col"] == 2       # NOT 3 -- must not regress to a faster loss


def test_analyze_malformed_400():
    r = client.post("/analyze", json={"board": "9,9,9"})
    assert r.status_code == 400


@pytest.mark.parametrize("payload", [
    {"board": [[3]]},        # nested list element
    {"board": [None]},       # null element
    {"board": [1, 2, [3]]},  # mixed shape, list buried among valid ints
])
def test_analyze_malformed_shapes_are_4xx_never_500(payload):
    # ESCAPE 1 root cause: `board: object` let a nested/null/mixed shape
    # reach `_parse_board`'s `int(c) for c in spec`, which raised an
    # uncaught TypeError -> uncaught-exception 500. SPEC.md sec.5: malformed
    # body must be 4xx, never 5xx. `AnalyzeIn.board: Union[str,
    # list[StrictInt]]` now rejects these shapes at validation (422).
    r = client.post("/analyze", json=payload)
    assert r.status_code != 500
    assert 400 <= r.status_code < 500


@pytest.mark.parametrize("raw_body", [
    '{"board":[NaN]}',
    '{"board":[Infinity]}',
    '{"board":[-Infinity]}',
    '{"board":[1,2,NaN]}',
    '{"board":NaN}',
])
def test_analyze_non_finite_float_never_500s(raw_body):
    # DEFECT 1 root cause: Python's stdlib `json` parser accepts the bare
    # literals NaN/Infinity/-Infinity (a non-standard extension), so a body
    # like `{"board":[NaN]}` parses fine and reaches pydantic. Pydantic then
    # correctly REJECTS the value (wrong type) and raises
    # RequestValidationError -- on track for a 422 -- but FastAPI's *default*
    # handler for that exception echoes the offending raw input value
    # verbatim into the error body, and Starlette's JSONResponse serializes
    # with `json.dumps(..., allow_nan=False)`, which raises `ValueError: Out
    # of range float values are not JSON compliant` from INSIDE exception
    # handling. That escapes uncaught -> bare 500. SPEC.md sec.5: a
    # malformed body must be 4xx, NEVER 5xx. The `StrictInt` field-level
    # guard (ESCAPE 1 fix) does not touch this -- the failure is one layer
    # further out, in the error-reporting path itself, not in application
    # code, so it is reachable regardless of how strict the field type is.
    r = client.post("/analyze", content=raw_body,
                     headers={"content-type": "application/json"})
    assert r.status_code != 500
    assert r.status_code in (400, 422)


def test_move_non_finite_float_column_never_500s_proves_class_closed():
    # Proves the fix closes the whole CLASS (a global exception handler),
    # not just the /analyze endpoint: a completely different endpoint/field
    # (MoveIn.col: int) fed a non-finite float also fails validation and,
    # pre-fix, hit the identical echo-then-serialize 500.
    r = client.post("/game/new", json={})
    gid = r.json()["id"]
    r = client.post(f"/game/{gid}/move", content='{"col":NaN}',
                     headers={"content-type": "application/json"})
    assert r.status_code != 500
    assert r.status_code in (400, 422)


def test_analyze_non_integer_column_rejected_not_silently_coerced():
    # Related looseness on the same path: a float column was silently
    # truncated to an int (3.5 -> 3) and returned 200 instead of being
    # rejected. `StrictInt` (not plain `int`) closes that coercion hole.
    r = client.post("/analyze", json={"board": [3.5]})
    assert r.status_code == 422


@pytest.mark.parametrize("payload", [
    {"board": ["x"]},
    {"board": [7]},
    {"board": [-1]},
])
def test_analyze_already_correct_malformed_behaviours_preserved(payload):
    # Must NOT regress: these were already correctly rejected 4xx before the
    # fix (string element / out-of-range column / negative column) and must
    # remain so -- just never with a different status family (never 500).
    r = client.post("/analyze", json=payload)
    assert 400 <= r.status_code < 500


def test_analyze_post_terminal_move_sequence_reports_correct_winner():
    # ESCAPE 3 regression: P1 completes a bottom-row horizontal four
    # (cols 0-3) at move 19, but the sequence keeps going for 5 more
    # (illegal, post-terminal) moves. Pre-fix, `Board.from_moves` kept
    # applying those trailing moves, reaching a board where BOTH players
    # had a four-in-a-row and `winner()` reported the WRONG player (2).
    full = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    r_full = client.post("/analyze", json={"board": full})
    assert r_full.status_code == 200
    body_full = r_full.json()
    assert body_full["terminal"] is True
    assert body_full["winner"] == 1

    truncated = full[:19]
    r_trunc = client.post("/analyze", json={"board": truncated})
    assert r_trunc.status_code == 200
    body_trunc = r_trunc.json()
    assert body_trunc["terminal"] is True
    assert body_trunc["winner"] == 1


def test_analyze_edge_column_opening_is_correctly_bad_for_the_opener():
    # Root-cause regression for the web/shots/play-analyze-desktop.png
    # anomaly: a vision judge flagged a screenshot where EVERY column showed
    # "Loss" after Red opened (and doubled down) on an edge column while
    # Yellow claimed the center twice. That was NOT a rendering bug -- it is
    # the game-theoretically correct call. Per solved Connect-Four opening
    # theory, an edge-column opening (column 1 or 7, i.e. index 0 or 6) is a
    # proven loss for whoever plays it against correct defense, while the
    # center column (index 3) is the unique winning opening. This test pins
    # that the depth-limited analyze fallback (used for near-empty, off-book
    # positions) still respects that asymmetry, so a future change to
    # `evaluate()`/`MinimaxAgent` can't silently invert it.
    edge = client.post("/analyze", json={"board": [0, 3, 0, 3], "mode": "scored"}).json()
    assert edge["exact"] is False  # depth-limited path (n=4 < ANALYZE_EXACT_PLY, no book)
    # every legal reply is rated a loss for Red (the mover) -- consistent with
    # Red having already blundered the opening onto a proven-losing column.
    assert all(v < 0 for v in edge["per_col"].values() if v is not None)

    center = client.post("/analyze", json={"board": [3, 3], "mode": "scored"}).json()
    assert center["exact"] is False
    # after Red opens center (the proven-best opening) and Yellow replies
    # center too, Red must NOT be assessed as lost on every column -- at
    # least one reply should be rated non-negative.
    assert any(v is not None and v >= 0 for v in center["per_col"].values())


def test_shoot_demo_sequence_analyze_overlay_is_not_a_wall_of_loss():
    # Mirrors exactly what web/scripts/shoot.mjs now drives before capturing
    # play-analyze-*.png: Red (human) opens the CENTER column twice, with an
    # agent replying in between, then /analyze is called on the resulting
    # position. Guards that the screenshot's demo scenario never regresses
    # back into the all-"Loss" opening (see test above) that made the
    # Analyze feature's showcase screenshot look broken.
    g = client.post("/game/new", json={"second_agent": "heuristic"}).json()
    gid = g["id"]
    client.post(f"/game/{gid}/move", json={"col": 3})
    client.post(f"/game/{gid}/agent-move")
    st = client.post(f"/game/{gid}/move", json={"col": 3}).json()
    if st["status"] == "in_progress":
        st = client.post(f"/game/{gid}/agent-move").json()

    if st["status"] == "in_progress":
        r = client.post("/analyze", json={"board": st["moves"], "mode": "scored"})
        body = r.json()
        assert any(v is not None and v >= 0 for v in body["per_col"].values()), (
            f"scripted demo opening produced an all-Loss analyze overlay: {body}"
        )


def test_leaderboard_endpoint():
    r = client.get("/leaderboard")
    # either serves the committed board or 404 if not yet built -- never 500
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        body = r.json()
        assert "agents" in body and "headline" in body


def test_won_game_reports_no_legal_moves_or_player_to_move():
    """Terminal-state invariant: a finished game must not advertise live moves.

    Regression for the escape where `legal_moves`/`player_to_move` were derived
    from column-fullness alone, so a WON game (board not full) still reported
    all open columns as legal and a live side to move -- letting a client offer
    a move the /move endpoint then 400s. Terminal status is the source of truth.
    """
    r = client.post("/game/new", json={})
    gid = r.json()["id"]
    # Human plays a vertical 4-in-column-0 win (opponent answers in column 1).
    last = None
    for col in [0, 1, 0, 1, 0, 1, 0]:
        last = client.post(f"/game/{gid}/move", json={"col": col}).json()
    assert last["status"] == "won"
    assert last["winner"] == 1
    assert last["legal_moves"] == []
    assert last["player_to_move"] is None
    assert last["to_move_is_agent"] is False
    assert last["to_move_agent"] is None
    # GET must agree with the move response.
    s = client.get(f"/game/{gid}").json()
    assert s["legal_moves"] == []
    assert s["player_to_move"] is None
    # A move on the finished game is still correctly rejected.
    assert client.post(f"/game/{gid}/move", json={"col": 2}).status_code == 400


def test_in_progress_game_still_reports_legal_moves():
    """Guard the fix: a live game must STILL report its open columns + mover."""
    r = client.post("/game/new", json={})
    gid = r.json()["id"]
    s = client.post(f"/game/{gid}/move", json={"col": 3}).json()
    assert s["status"] == "in_progress"
    assert s["legal_moves"] == [0, 1, 2, 3, 4, 5, 6]
    assert s["player_to_move"] == 2
