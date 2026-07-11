// Plain-JS mock backend for NeuroFour's frontend: screenshots (shoot.mjs) and
// Playwright e2e both drive the same in-memory Connect-4 engine + fixture
// data through page.route(), for use when the real backend isn't reachable.
// Wire shapes here mirror the REAL running FastAPI backend (app/main.py),
// verified against it directly — not the originally-documented spec, which
// diverges from what the backend actually returns.

export const ROWS = 6;
export const COLS = 7;

export const MOCK_AGENTS = [
  { name: 'random', kind: 'random', params: 0, size_bytes: 0, flops_per_move: 7, artifact_path: null },
  { name: 'heuristic', kind: 'heuristic', params: 0, size_bytes: 0, flops_per_move: 490, artifact_path: null },
  { name: 'minimax-2', kind: 'search', params: 0, size_bytes: 0, flops_per_move: 3430, artifact_path: null },
  { name: 'minimax-4', kind: 'search', params: 0, size_bytes: 0, flops_per_move: 168070, artifact_path: null },
  { name: 'neurofour-net', kind: 'nn', params: 33031, size_bytes: 25603, flops_per_move: 66251, artifact_path: 'app/agents/artifacts/neurofour-net.npz' },
  { name: 'perfect', kind: 'search', params: 0, size_bytes: 0, flops_per_move: 50_000_000, artifact_path: null },
];

const MOCK_LEADERBOARD_AGENTS = [
  { name: 'perfect', kind: 'search', optimality: 1.0, blunder_rate: 0.0, soundness: 1.0, size_bytes: 0, params: 0, flops_per_move: 50_000_000, flops_plausible: true, latency_ms: 0.003, over_budget: true, tier: 'nano', qualifies_micro: false, neurogolf_score: 100.0, elo: 1213, pareto: true },
  { name: 'minimax-4', kind: 'search', optimality: 0.936667, blunder_rate: 0.02, soundness: 0.98, size_bytes: 0, params: 0, flops_per_move: 168070, flops_plausible: true, latency_ms: 0.525, over_budget: false, tier: 'nano', qualifies_micro: true, neurogolf_score: 94.317, elo: 1064, pareto: true },
  { name: 'minimax-2', kind: 'search', optimality: 0.903333, blunder_rate: 0.03, soundness: 0.97, size_bytes: 0, params: 0, flops_per_move: 3430, flops_plausible: true, latency_ms: 0.05, over_budget: false, tier: 'nano', qualifies_micro: true, neurogolf_score: 91.234, elo: 936, pareto: true },
  { name: 'heuristic', kind: 'heuristic', optimality: 0.9, blunder_rate: 0.03, soundness: 0.96, size_bytes: 0, params: 0, flops_per_move: 490, flops_plausible: true, latency_ms: 0.01, over_budget: false, tier: 'nano', qualifies_micro: true, neurogolf_score: 90.1, elo: 808, pareto: true },
  // optimality is deliberately ABOVE minimax-4's (the free/0-byte best) so
  // this agent is a genuine higher-cost frontier vertex (paying bytes for
  // more strength) — mirrors the real backend's net1-vs-minimax-4 shape and
  // exercises the honest multi-tier staircase, not just a same-cost tie.
  { name: 'neurofour-net', kind: 'nn', optimality: 0.95, blunder_rate: 0.06, soundness: 0.92, size_bytes: 25603, params: 33031, flops_per_move: 66251, flops_plausible: true, latency_ms: 0.11, over_budget: false, tier: 'micro', qualifies_micro: true, neurogolf_score: 88.6, elo: 1180, pareto: true },
  { name: 'random', kind: 'random', optimality: 0.263333, blunder_rate: 0.74, soundness: 0.3, size_bytes: 0, params: 0, flops_per_move: 7, flops_plausible: true, latency_ms: 0.001, over_budget: false, tier: 'nano', qualifies_micro: true, neurogolf_score: 24.0, elo: 0, pareto: true },
];

export const MOCK_LEADERBOARD = {
  seed: 4,
  headline: { metric: 'optimality', value: 0.936667, agent: 'minimax-4', tier: 'nano' },
  auc_strength_logsize: 0.0,
  tiers: {
    nano: { name: 'minimax-4', optimality: 0.936667, size_bytes: 0, neurogolf_score: 94.317 },
    micro: { name: 'minimax-4', optimality: 0.936667, size_bytes: 0, neurogolf_score: 94.317 },
    mini: { name: 'minimax-4', optimality: 0.936667, size_bytes: 0, neurogolf_score: 94.317 },
    small: { name: 'minimax-4', optimality: 0.936667, size_bytes: 0, neurogolf_score: 94.317 },
    open: { name: 'perfect', optimality: 1.0, size_bytes: 0, neurogolf_score: 100.0 },
  },
  agents: MOCK_LEADERBOARD_AGENTS,
  frontier: {
    by_size: MOCK_LEADERBOARD_AGENTS
      .filter((a) => a.pareto)
      .sort((a, b) => a.size_bytes - b.size_bytes)
      .map((a) => ({ name: a.name, size_bytes: a.size_bytes, optimality: a.optimality, elo: a.elo })),
    by_flops: MOCK_LEADERBOARD_AGENTS
      .filter((a) => a.pareto)
      .sort((a, b) => a.flops_per_move - b.flops_per_move)
      .map((a) => ({ name: a.name, flops_per_move: a.flops_per_move, optimality: a.optimality, elo: a.elo })),
  },
  ladder: {
    elo: Object.fromEntries(MOCK_LEADERBOARD_AGENTS.map((a) => [a.name, a.elo])),
    games: Object.fromEntries(MOCK_LEADERBOARD_AGENTS.map((a) => [a.name, 100])),
    scores: Object.fromEntries(MOCK_LEADERBOARD_AGENTS.map((a) => [a.name, a.elo / 20])),
  },
};

export function emptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(0));
}

// Row convention mirrors the REAL backend exactly (app/engine/board.py:
// "row 0 = bottom, row 5 = top playable") — NOT the naive row-0-is-top guess
// the frontend originally shipped with (which rendered a real game upside
// down). The frontend flips this to its own row-0-is-top convention at the
// api.ts boundary, uniformly for both the real backend and this mock.
function lowestEmptyRow(board, col) {
  for (let r = 0; r < ROWS; r += 1) {
    if (board[r][col] === 0) return r;
  }
  return -1;
}

function legalMoves(board) {
  const moves = [];
  for (let c = 0; c < COLS; c += 1) if (board[ROWS - 1][c] === 0) moves.push(c);
  return moves;
}

const DIRECTIONS = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

/** Returns [row,col] pairs (matching the real backend's wire shape), or null. */
function checkWin(board, row, col, player) {
  for (const [dr, dc] of DIRECTIONS) {
    const line = [[row, col]];
    for (const sign of [1, -1]) {
      let r = row + dr * sign;
      let c = col + dc * sign;
      while (r >= 0 && r < ROWS && c >= 0 && c < COLS && board[r][c] === player) {
        line.push([r, c]);
        r += dr * sign;
        c += dc * sign;
      }
    }
    if (line.length >= 4) {
      return line.slice(0, 4);
    }
  }
  return null;
}

function boardFromMoves(moves) {
  const board = emptyBoard();
  let player = 1;
  for (const col of moves) {
    const row = lowestEmptyRow(board, col);
    board[row][col] = player;
    player = player === 1 ? 2 : 1;
  }
  return board;
}

/** In-memory Connect-4 game store keyed by id, mirroring the REAL /game
 * wire shape (player_to_move / to_move_is_agent / moves / winning_line as
 * [row,col] pairs — no last_move; the frontend derives that itself). */
export class MockGameStore {
  constructor() {
    this.games = new Map();
    this.counter = 0;
  }

  create(firstAgent, secondAgent) {
    this.counter += 1;
    const id = `mock-${this.counter}`;
    const state = {
      id,
      board: emptyBoard(),
      key: '0:0',
      moves: [],
      legal_moves: legalMoves(emptyBoard()),
      player_to_move: 1,
      to_move_is_agent: Boolean(firstAgent),
      to_move_agent: firstAgent ?? null,
      first_agent: firstAgent ?? null,
      second_agent: secondAgent ?? null,
      status: 'in_progress',
      winner: 0,
      winning_line: null,
      num_moves: 0,
    };
    this.games.set(id, state);
    return state;
  }

  get(id) {
    const s = this.games.get(id);
    if (!s) throw new Error('unknown game');
    return s;
  }

  applyMove(id, col) {
    const s = this.get(id);
    if (s.status !== 'in_progress') throw new Error('game already finished');
    if (!legalMoves(s.board).includes(col)) throw new Error('illegal column');
    const row = lowestEmptyRow(s.board, col);
    const player = s.player_to_move;
    s.board[row][col] = player;
    s.moves.push(col);
    s.num_moves = s.moves.length;
    const line = checkWin(s.board, row, col, player);
    if (line) {
      s.status = 'won';
      s.winner = player;
      s.winning_line = line;
    } else if (legalMoves(s.board).length === 0) {
      s.status = 'draw';
      s.winner = 0;
    } else {
      s.player_to_move = player === 1 ? 2 : 1;
    }
    s.legal_moves = legalMoves(s.board);
    const sideAgent = s.player_to_move === 1 ? s.first_agent : s.second_agent;
    s.to_move_is_agent = s.status === 'in_progress' && Boolean(sideAgent);
    s.to_move_agent = s.status === 'in_progress' ? (sideAgent ?? null) : null;
    return s;
  }

  /** Deterministic "agent": center-first column preference among legal moves. */
  agentMove(id) {
    const s = this.get(id);
    const mover = s.player_to_move;
    const moverAgent = mover === 1 ? s.first_agent : s.second_agent;
    const order = [3, 2, 4, 1, 5, 0, 6];
    const legal = legalMoves(s.board);
    const col = order.find((c) => legal.includes(c)) ?? legal[0];
    this.applyMove(id, col);
    const out = { ...this.get(id) };
    out.agent_move = col;
    out.agent = moverAgent;
    return out;
  }
}

/** Fake but structurally valid /analyze response for a move-history board
 * spec (array of column ints — the real backend never accepts a full grid). */
export function mockAnalyze(boardSpec, _mode) {
  const moves = Array.isArray(boardSpec) ? boardSpec : [];
  const board = boardFromMoves(moves);
  const legal = legalMoves(board);
  const order = [3, 2, 4, 1, 5, 0, 6];
  const best = order.find((c) => legal.includes(c)) ?? legal[0];
  const worst = [...legal].reverse().find((c) => c !== best) ?? null;
  const per_col = {};
  for (let c = 0; c < COLS; c += 1) {
    if (!legal.includes(c)) {
      per_col[String(c)] = null;
    } else if (c === best) {
      per_col[String(c)] = 18;
    } else if (c === worst) {
      per_col[String(c)] = -12;
    } else {
      per_col[String(c)] = 0;
    }
  }
  return {
    terminal: false,
    player_to_move: moves.length % 2 === 0 ? 1 : 2,
    value: per_col[String(best)],
    optimal_cols: [best],
    best_col: best,
    per_col,
    mode: 'depth-limited',
    exact: false,
  };
}
