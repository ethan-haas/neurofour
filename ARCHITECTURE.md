# NeuroFour — architecture & interfaces

## Layers (bottom-up)

```
┌──────────────────────────────────────────────────────────────────┐
│ web/  (Next.js/SPA)   Play · Analyze · Leaderboard(Pareto plot)   │
└───────────────▲──────────────────────────────────────────────────┘
                │ HTTP JSON
┌───────────────┴──────────────────────────────────────────────────┐
│ app/main.py  (FastAPI)  /agents /game /analyze /leaderboard ...   │
└──────┬───────────────┬───────────────┬───────────────┬────────────┘
       │               │               │               │
┌──────▼─────┐  ┌──────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
│ engine/    │  │ solver/     │  │ agents/    │  │ bench/      │
│ Board      │  │ solve()     │  │ Agent base │  │ positions   │
│ bitboards  │  │ negamax+αβ  │  │ registry   │  │ strength    │
│ win detect │  │ TT + symm   │  │ net/heur.. │  │ ladder/Elo  │
└────────────┘  └─────────────┘  └────────────┘  │ cost/score  │
                                                  └─────────────┘
```

## Key interfaces (contracts the components agree on)

### `engine.Board`
Immutable-ish value object. `legal_moves()`, `play(col)->Board`, `winner()`, `is_draw()`,
`is_terminal()`, `player_to_move()`, `to_key()` (canonical incl. mirror-normal form for TT/symmetry),
`from_moves(seq)`, `from_key(s)`, `__str__` for the UI. Bitboard internals; expose a `cells()` grid
accessor for rendering/encoding.

### `solver.solve(board, mode="scored") -> Solved`
`Solved(value:int, optimal_cols:list[int], best_col:int, per_col:dict[int,int])`. `mode="value"` uses
the ±1/0 convention. Negamax + alpha-beta + transposition table (Zobrist or key-hash) + center-first
move ordering + horizontal-mirror symmetry. Pure, deterministic, no global mutable state that leaks
across calls except the TT cache (which must be correctness-preserving).

### `agents.Agent`
`select_move(board)->int` (ONLY input is the board — enforced: the framework passes nothing else),
`manifest()->AgentManifest`. Learned agents load their artifact from `agents/artifacts/<name>.npz`
(or similar); `size_bytes` = `os.path.getsize(artifact)`. The **encoder** (board → feature vector) is
shared code in `agents/encode.py` so trainer and inference agree exactly.

### `bench`
- `positions.generate(seed, n_per_set) -> writes bench_data/*.jsonl` using `solver` for labels.
- `strength.score(agent, positions) -> StrengthCard(optimality, blunder_rate, soundness, per_outcome)`.
- `ladder.run(agents, openings) -> {elo, results}`.
- `cost.measure(agent, positions) -> CostCard(size_bytes, params, flops_per_move, latency_ms, over_budget)`.
- `score.build_leaderboard(cards) -> leaderboard.json` (neurogolf_score, tier, pareto flags, frontier).

## Data flow: producing the leaderboard
1. `gen_positions.py` → solver labels → `bench_data/{train,dev,sealed}.jsonl`.
2. `train_net.py` → reads `train`/`dev` → fits tiny MLP → `agents/artifacts/neurofour-net.npz`.
3. `run_bench.py` → for each registered agent: `strength.score(sealed)` + `cost.measure` → `ladder.run`
   → `score.build_leaderboard` → `leaderboard.json`.
4. API `/leaderboard` serves that JSON; `web/` plots the frontier.

## Determinism boundaries
- Solver, encoder, strength, score: **deterministic** (pure functions of inputs + seed).
- `latency_ms`: measured, **excluded from pass/fail** (report-only + coarse tiebreak bucket).
- Game API: in-memory game store keyed by id; a new game with fixed agents + fixed human moves is
  reproducible.

## The flagship `neurofour-net` (reference design, maker may improve)
Encoder: the 6×7 board as two bitplanes (side-to-move discs, opponent discs) → 84-dim {0,1} vector,
optionally plus a few cheap engineered features (column heights, immediate-win/immediate-loss flags).
Model: 1 hidden layer MLP (e.g. 84→H→7) with a legal-move mask on the output; argmax over legal
columns. Trained by cross-entropy against the solver's `optimal_cols` (multi-hot) or best_col, on
`train`, early-stopped on `dev`. With `H` in the low tens this is a few thousand params → a few KB in
`float16`/`int8` → fits `micro` (≤32KB), often `nano` (≤4KB) with quantization. Goal: maximize
`optimality` on `sealed` at the smallest artifact.
