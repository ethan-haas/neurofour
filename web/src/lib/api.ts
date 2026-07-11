import type {
  AgentManifest,
  AgentsResponse,
  AnalyzeRequest,
  AnalyzeResult,
  GameState,
  LeaderboardResponse,
  NewGameRequest,
} from '../types';

export const API_BASE: string = (() => {
  const env = import.meta.env as Record<string, string | undefined>;
  return (env.VITE_API_BASE ?? env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000').replace(
    /\/$/,
    '',
  );
})();

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
    });
  } catch (err) {
    // An intentional cancellation (AbortController.abort()) is not a
    // network failure -- let it propagate as-is so callers can tell "the
    // caller cancelled this" apart from "the backend is unreachable".
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    throw new ApiError('Cannot reach the NeuroFour backend. Is it running?', 0);
  }

  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      detail = body?.detail ?? body?.message ?? JSON.stringify(body);
    } catch {
      detail = res.statusText;
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** The backend's board/winning_line use "row 0 = bottom" (see
 * app/engine/board.py). The rest of this app renders with "row 0 = top"
 * (a plain top-to-bottom CSS grid). Flip once here, at the wire boundary,
 * so every consumer downstream can assume row 0 is the top row. Without
 * this the board renders upside down against the real backend (discs
 * appear to fall upward) — it only ever looked right against the mock,
 * which happened to guess the opposite (wrong) convention. */
function normalizeGame(raw: GameState): GameState {
  const height = raw.board.length;
  const board = [...raw.board].reverse();
  const winning_line = raw.winning_line
    ? raw.winning_line.map(([r, c]): [number, number] => [height - 1 - r, c])
    : null;
  return { ...raw, board, winning_line };
}

export const api = {
  health: () => request<{ status: string } | { ok: boolean }>('/health'),

  agents: () => request<AgentsResponse>('/agents').then((r) => r.agents as AgentManifest[]),

  newGame: (body: NewGameRequest) =>
    request<GameState>('/game/new', { method: 'POST', body: JSON.stringify(body) }).then(
      normalizeGame,
    ),

  getGame: (id: string) =>
    request<GameState>(`/game/${encodeURIComponent(id)}`).then(normalizeGame),

  move: (id: string, col: number) =>
    request<GameState>(`/game/${encodeURIComponent(id)}/move`, {
      method: 'POST',
      body: JSON.stringify({ col }),
    }).then(normalizeGame),

  agentMove: (id: string) =>
    request<GameState>(`/game/${encodeURIComponent(id)}/agent-move`, { method: 'POST' }).then(
      normalizeGame,
    ),

  analyze: (body: AnalyzeRequest, opts?: { signal?: AbortSignal }) =>
    request<AnalyzeResult>('/analyze', { method: 'POST', body: JSON.stringify(body), signal: opts?.signal }),

  leaderboard: () => request<LeaderboardResponse>('/leaderboard'),
};
