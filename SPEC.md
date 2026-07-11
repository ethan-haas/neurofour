# NeuroFour — build spec (BUILD it; independent audits judge it)

Build a benchmark + arena for **Connect 4 agents scored on strength × computational cheapness**
("NeuroGolf for Connect 4"). Backend (engine, exact solver, agent framework, scorer, API) **and** a
web UI (play vs any agent + a Pareto-frontier leaderboard). It is a large build — build it as
COMPONENTS, smoke-test each, then wire the SEAMS, then the UI. **Build from THIS spec and `METRIC.md`
only; never read `AUDIT.md` or any `audit`/`oracle`/`answer` path, and do not inject invariants from
memory.**

Language: Python 3.11 backend (stdlib + `numpy` + `fastapi` + `uvicorn` + `pydantic`; `hypothesis`
for tests). Frontend: your choice of Next.js or a self-contained SPA in `web/`. No GPU, no internet at
runtime, no external services.

---

## 1. The game (`app/engine`)

Standard Connect 4: **7 columns × 6 rows**, two players (`1` = first/red, `2` = second/yellow).
Players alternate dropping a disc into a non-full column; the disc falls to the lowest empty cell. A
player wins by making **4 in a row** horizontally, vertically, or diagonally. If the board fills with
no winner it is a **draw**.

Provide a `Board` abstraction with at least:
- construction from empty, from a move sequence (list of columns), and from a compact string.
- `legal_moves() -> list[int]` (columns 0–6 that are not full).
- `play(col) -> Board` (returns the resulting position; reject illegal columns with a clear error).
- `winner() -> 0|1|2` and `is_draw()`, `is_terminal()`.
- `to_key()` — a canonical, hashable encoding (use it for transposition + symmetry).
- `player_to_move() -> 1|2`.

**Use bitboards** (two 64-bit integers, one per player, in the standard 7×7-with-sentinel layout) for
the engine and solver — a naive nested-list board will not meet the solver's speed target.

## 2. The perfect solver (`app/solver`) — the ground-truth ORACLE

An **exact** Connect 4 solver. Given any non-terminal position it returns the game-theoretic value for
the side to move and the set of optimal moves.

- `solve(board) -> Solved` where `Solved` carries: `value` (see below), `optimal_cols: list[int]`
  (every column whose resulting position preserves the best achievable value), and `best_col`.
- **Value convention (mandatory, exact):** a signed integer score. A win is positive, a loss negative,
  a draw `0`; a *faster* win scores higher than a slower win and a *slower* loss scores higher than a
  faster loss. Concretely, score a win as `(REMAINING_CELLS_AFTER_WIN + 1)` for the winner's side and
  the negation for a loss, so mate-in-fewer is strictly preferred. Draw = 0. The exact scale is up to
  you but MUST be internally consistent so that `optimal_cols` = every move achieving `max` value.
- Implement **negamax with alpha-beta pruning, a transposition table, move ordering (center-first),
  and board symmetry** (horizontal mirror). It must solve the empty board and arbitrary midgame
  positions **exactly** (Connect 4 is a first-player win: the empty board's optimal move is the center
  column). Target: solve any position in well under a second after warmup; cache/iterative-deepen as
  needed.
- Expose a **weak/value-only mode** (win/draw/loss = `+1/0/-1`) in addition to the full scored mode;
  the metric uses value-only to define "did the agent preserve the outcome?" and scored mode to define
  the optimal-move set.

The solver is used **only** to (a) label positions offline for the benchmark and (b) act as the
top-of-ladder opponent. **Agents must never invoke the solver at inference time** (the agent framework
must not give agents access to it).

## 3. Agents (`app/agents`) — the models under test

A uniform interface so any entrant is scored identically. Base class `Agent`:

```python
class Agent:
    name: str
    def select_move(self, board: Board) -> int: ...   # returns a legal column; ONLY input is the board
    def manifest(self) -> AgentManifest: ...           # cost self-report (verified by the bench)
```

`AgentManifest` fields: `name`, `kind` ("table" | "nn" | "search" | "heuristic" | "random"),
`params: int`, `size_bytes: int` (serialized artifact size — see METRIC), `flops_per_move: int`
(declared inference cost), and an optional `artifact_path`. A registry (`app/agents/registry.py`)
lists all agents by name and constructs them.

Ship at least these agents so the frontier is populated:
- `random` — uniform legal move (floor baseline; tiny).
- `heuristic` — a hand-written static evaluation (center preference, counts of open 2s/3s, immediate
  win/block) picking the best 1-ply move. Small, no learned params.
- `minimax-k` for a couple of depths `k` (e.g. 2 and 4) using the engine + a simple eval — a *search*
  agent whose cost is FLOPs/latency, not bytes.
- `perfect` — wraps the solver at full strength; it is the **strength ceiling and cost ceiling**
  reference (it is intentionally NOT cheap — it exists to anchor the frontier's top-left).
- **`neurofour-net`** — the flagship **NeuroGolf entrant**: a *tiny* neural network (a small numpy MLP
  over a bitboard/feature encoding of the position) trained by supervised distillation on
  solver-labelled positions. This is the agent that should sit far out on the frontier: high
  optimality at a few KB. Provide its trainer (`app/agents/train_net.py`) and commit the trained
  weights artifact. Keep it genuinely small (target: strong play well under 32 KB of weights).

Agents that require a learned artifact must serialize/deserialize it deterministically; `size_bytes`
is the real byte length of that artifact on disk.

## 4. The NeuroGolf benchmark (`app/bench`) — scoring

Implement exactly the math in `METRIC.md`. Components:
- **Position generator** (`positions.py`): produce reproducible position sets by self-play/random play
  to varied depths, deduplicated, then label each with the solver (value + optimal_cols). Emit three
  disjoint sets — `train`, `dev`, `sealed` — as JSONL, with a fixed seed. Trainers may use `train`
  (+`dev`); scoring strength uses `sealed` (agents must not have trained on it).
- **Strength scoring** (`strength.py`): over a position set, compute `optimality` (fraction where the
  agent's move is in `optimal_cols`), `blunder_rate` (fraction where the agent's move strictly worsens
  the value-only outcome vs the position's best), and per-outcome breakdowns.
- **Ladder + Elo** (`ladder.py`): a round-robin of paired games (each pair plays both colors from a
  fixed set of opening plies) among registered agents + fixed reference opponents; compute Elo.
- **Cost measurement** (`cost.py`): read `size_bytes` from the artifact, verify `flops_per_move` is
  plausible against a declared cap, and measure `latency_ms` (p50 over the sealed set, warmup
  excluded).
- **Composite** (`score.py`): compute `neurogolf_score`, assign each agent to a size **budget tier**,
  and compute the **Pareto frontier** (non-dominated on the strength↑ / cost↓ plane). Write
  `leaderboard.json`.

**Anti-cheat requirements the bench MUST enforce (see METRIC + AUDIT):**
- Agents receive **only** the `Board` in `select_move` — no solver handle, no test-set handle, no I/O.
- Strength is measured on the **sealed** set (and the bench must support re-labeling a *freshly
  generated* position set at audit time, so memorizing positions cannot win).
- `size_bytes` is the artifact's real size; a giant lookup table pays for its bytes.
- An inference **compute ceiling**: an agent exceeding a per-move latency/FLOP cap is flagged and
  scored as `disqualified` for the cheap tiers (documented in METRIC).

## 5. Backend API (`app/main.py`, importable `app.main:app`)

FastAPI, JSON. Deterministic where noted; read config from env (`NEUROFOUR_DB_PATH` if you persist,
`NEUROFOUR_SEED`). Endpoints (names/shapes are yours to finalize but must cover):

- `GET /health` → ok.
- `GET /agents` → list of registered agents with their manifests (name, kind, params, size_bytes,
  flops_per_move).
- `POST /game/new` `{first_agent?, second_agent?}` → create a game (human vs agent, or agent vs agent);
  returns a game id + board state.
- `GET /game/{id}` → current board, legal moves, player to move, status (in_progress/won/draw), winner.
- `POST /game/{id}/move` `{col}` → apply a **human** move (must be legal; reject illegal with 4xx),
  returns updated state.
- `POST /game/{id}/agent-move` → have the side-to-move **agent** play one move; returns the move and
  updated state.
- `POST /analyze` `{board}` → solver analysis of a position: value, optimal_cols, best_col, and (for
  UI) a per-column value readout. (This uses the solver server-side; it is an *analysis* endpoint, not
  something agents can call.)
- `GET /leaderboard` → the current `leaderboard.json` (agents with strength, cost, neurogolf_score,
  tier, and a `pareto: bool` flag), plus the frontier points for plotting.
- `POST /evaluate` `{agent}` (optional, may be async/cached) → run the bench for one agent and return
  its scorecard.

Errors: malformed body → 4xx (never 500); illegal move → 4xx with reason; unknown game/agent → 404.

## 6. Frontend (`web/`) — the arena

A polished, responsive single app. Screens/flow:
1. **Play** — an interactive 7×6 Connect 4 board. Choose your opponent (any registered agent) and who
   goes first. Click a column to drop; the agent replies. Show whose turn, legal columns, win/draw
   result with the winning line highlighted, and a "new game" / "watch agent vs agent" mode.
2. **Analyze** (toggle) — overlay the solver's per-column evaluation (which columns win/draw/lose, best
   move) so a human can see how optimal the agent's move was. Make it clearly an *analysis aid*.
3. **Leaderboard** — a table of agents (strength/optimality, size, FLOPs, latency, neurogolf_score,
   tier) **and** a **Pareto-frontier scatter plot**: x = cost (log size or FLOPs), y = strength
   (optimality or Elo), with the non-dominated frontier drawn as a line and each agent as a point;
   highlight `neurofour-net` as the flagship. Hover/focus shows the agent's card.

Quality bar (independent vision judge + scriptable gate): premium cohesive design; responsive at 375px
and 1440px with no horizontal page scroll; real loading/empty/error states; accessible (keyboard-
playable board, visible focus, WCAG AA contrast, aria live region announcing moves/results, zero axe
critical/serious); production build clean; zero console errors.

---

## Deliverables checklist
- `app/engine`, `app/solver`, `app/agents` (incl. trained `neurofour-net` + trainer + committed
  weights), `app/bench`, `app/main.py`.
- `bench_data/` generated sets + `leaderboard.json` produced by `scripts/run_bench.py`.
- `web/` app; `scripts/shoot.mjs` for screenshots.
- `requirements.txt`, tests under `tests/` (engine correctness, solver-vs-known values, metric
  determinism, API happy/edge paths), and a top-level `make`/README run recipe.
- A verify command that runs the test suite green.

Build both halves to production quality. Independent auditors test the running backend black-box and
screenshot the running frontend. Do not read `AUDIT.md`.
