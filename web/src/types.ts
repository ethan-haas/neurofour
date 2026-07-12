// Types mirroring the NeuroFour backend API contract, verified against the
// REAL running FastAPI app (app/main.py) — not the originally-documented
// spec, which the real backend diverges from in several places (agents
// wrapper, player_to_move naming, winning_line as [row,col] pairs, analyze
// taking a move-history not a board grid, leaderboard top-level shape).

export type Cell = 0 | 1 | 2;
/** 6 rows x 7 cols. row 0 = top row, row 5 = bottom row. 0 = empty, 1/2 = player disc. */
export type BoardGrid = Cell[][];

export type PlayerId = 1 | 2;

export type AgentKind = 'table' | 'nn' | 'search' | 'heuristic' | 'random';

export type BudgetTier = 'nano' | 'micro' | 'mini' | 'small' | 'open';

/** GET /agents now enriches the manifest server-side (app/main.py::agents)
 * with a display name/subtitle (app/agents/display.py) and, when the agent
 * has a row in bench_data/leaderboard.json, its strength/cost stats. Any
 * agent with no leaderboard row gets `null` for every stat field below --
 * never fabricated. */
export interface AgentManifest {
  name: string;
  kind: AgentKind;
  params: number;
  size_bytes: number;
  flops_per_move: number;
  artifact_path?: string | null;
  display_name: string;
  subtitle: string;
  optimality: number | null;
  elo: number | null;
  latency_ms: number | null;
  /** Frozen wire field name (run_bench.py --check contract) -- never
   * rendered as the literal string "NeuroGolf" anywhere in the UI. */
  neurogolf_score: number | null;
  tier: BudgetTier | null;
  pareto: boolean | null;
  over_budget: boolean | null;
}

/** Raw wire shape of GET /agents. */
export interface AgentsResponse {
  agents: AgentManifest[];
}

export type GameStatus = 'in_progress' | 'won' | 'draw';

export interface WinCell {
  row: number;
  col: number;
}

/** The backend does not return the last move; the frontend derives it from
 * `board` + `moves` (see lib/format.ts#lastMoveFromGame). */
export interface LastMove {
  col: number;
  row: number;
  player: PlayerId;
}

/** Raw wire shape returned by /game/new, GET /game/{id}, /game/{id}/move,
 * and /game/{id}/agent-move (the last two also add agent_move/agent). */
export interface GameState {
  id: string;
  board: BoardGrid;
  key: string;
  moves: number[];
  /** Empty once the game is terminal (won/draw) — the server derives this
   * from the terminal status, not column-fullness, so a finished game never
   * advertises a "legal" move. */
  legal_moves: number[];
  /** Null once the game is terminal — no side is to move on a finished game. */
  player_to_move: PlayerId | null;
  to_move_is_agent: boolean;
  to_move_agent: string | null;
  first_agent: string | null;
  second_agent: string | null;
  status: GameStatus;
  winner: 0 | PlayerId;
  /** [row, col] pairs forming the winning four, or null. */
  winning_line: [number, number][] | null;
  num_moves: number;
  agent_move?: number;
  agent?: string;
}

export interface NewGameRequest {
  first_agent?: string | null;
  second_agent?: string | null;
}

export interface MoveRequest {
  col: number;
}

export type AnalyzeMode = 'value' | 'scored';

/** /analyze accepts a move-history (column ints), a "mask:cur" key string, or
 * a comma-separated move string — never a full board grid. */
export interface AnalyzeRequest {
  board: number[] | string;
  mode?: AnalyzeMode;
}

export interface AnalyzeResult {
  terminal: boolean;
  player_to_move: PlayerId | null;
  value: number | null;
  optimal_cols: number[];
  best_col: number | null;
  per_col: Record<string, number | null>;
  mode?: string;
  exact?: boolean;
  winner?: 0 | PlayerId | null;
  is_draw?: boolean;
}

export interface LeaderboardAgent {
  name: string;
  kind: AgentKind;
  params: number;
  size_bytes: number;
  flops_per_move: number;
  flops_plausible?: boolean;
  latency_ms: number;
  optimality: number;
  soundness: number;
  blunder_rate: number;
  elo: number;
  neurogolf_score: number;
  tier: BudgetTier;
  pareto: boolean;
  over_budget: boolean;
  qualifies_micro?: boolean;
  per_outcome?: Record<string, { n: number; optimality: number }>;
  /** Injected at serve time by GET /leaderboard (app/main.py) from
   * app/agents/display.py -- never persisted into the committed
   * bench_data/leaderboard.json artifact. */
  display_name?: string;
  subtitle?: string;
}

export interface FrontierBySizePoint {
  name: string;
  size_bytes: number;
  optimality: number;
  elo: number;
}

export interface FrontierByFlopsPoint {
  name: string;
  flops_per_move: number;
  optimality: number;
  elo: number;
}

export interface Headline {
  metric: string;
  value: number;
  agent: string | null;
  tier: BudgetTier | null;
}

export interface TierBest {
  name: string;
  optimality: number;
  size_bytes: number;
  neurogolf_score: number;
}

/** Raw wire shape of GET /leaderboard (bench_data/leaderboard.json, served as-is). */
export interface LeaderboardResponse {
  seed: number;
  headline: Headline;
  auc_strength_logsize: number;
  tiers: Record<string, TierBest | null>;
  agents: LeaderboardAgent[];
  frontier: {
    by_size: FrontierBySizePoint[];
    by_flops: FrontierByFlopsPoint[];
  };
  ladder?: {
    elo: Record<string, number>;
    games: Record<string, number>;
    scores: Record<string, number>;
  };
}

export interface EvaluateResponse {
  agent: string;
  optimality: number;
  soundness: number;
  blunder_rate: number;
  neurogolf_score: number;
  tier: BudgetTier;
}
