import type { BudgetTier } from '../types';

export const TIER_LABEL: Record<BudgetTier, string> = {
  nano: 'Nano ≤4KB',
  micro: 'Micro ≤32KB',
  mini: 'Mini ≤256KB',
  small: 'Small ≤2MB',
  open: 'Open',
};

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb % 1 === 0 ? kb : kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb % 1 === 0 ? mb : mb.toFixed(2)} MB`;
}

/** Format a positive number to 3 significant figures (e.g. 50 -> "50.0",
 * 5 -> "5.00", 168.07 -> "168"). Used below to give every K/M-scaled FLOP
 * count the SAME precision rule, instead of each magnitude band picking its
 * own ad-hoc decimal count (the old rule was "0 decimals if an exact
 * multiple of the unit, else a fixed 1 or 2 decimals" -- which produced
 * "50M", "5.00M", "168.1K" side by side with no consistent rule a reader
 * could learn). */
function sig3(v: number): string {
  if (v === 0) return '0';
  const digits = Math.floor(Math.log10(Math.abs(v))) + 1;
  const decimals = Math.max(0, 3 - digits);
  return v.toFixed(decimals);
}

export function formatFlops(flops: number): string {
  if (flops < 1000) return `${Math.round(flops)}`;
  if (flops < 999_500) return `${sig3(flops / 1000)}K`;
  if (flops < 999_500_000) return `${sig3(flops / 1_000_000)}M`;
  return `${sig3(flops / 1_000_000_000)}B`;
}

export function formatPct(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
}

export function formatLatency(ms: number): string {
  return `${ms.toFixed(ms < 10 ? 2 : 1)} ms`;
}

export function formatScore(score: number): string {
  return score.toFixed(3);
}

export const EMPTY_BOARD_ROWS = 6;
export const EMPTY_BOARD_COLS = 7;

export function emptyBoard(): number[][] {
  return Array.from({ length: EMPTY_BOARD_ROWS }, () => Array(EMPTY_BOARD_COLS).fill(0));
}

/** Given a board grid, find lowest empty row (highest row index) in a column, or -1 if full. */
export function lowestEmptyRow(board: number[][], col: number): number {
  for (let r = EMPTY_BOARD_ROWS - 1; r >= 0; r -= 1) {
    if (board[r][col] === 0) return r;
  }
  return -1;
}

export function columnLegal(board: number[][], col: number): boolean {
  return board[0][col] === 0;
}

/** The backend doesn't return a "last move" field; derive it from the board
 * + column-move-history it does return. The most recently played column's
 * disc is always the topmost occupied cell in that column (discs in a
 * column only ever stack bottom-up, and nothing has touched that column
 * since), so this is exact, not a heuristic. */
export function lastMoveFromGame(
  board: number[][],
  moves: number[],
): { row: number; col: number; player: 1 | 2 } | null {
  if (moves.length === 0) return null;
  const col = moves[moves.length - 1];
  for (let r = 0; r < board.length; r += 1) {
    const cell = board[r][col];
    if (cell !== 0) return { row: r, col, player: cell as 1 | 2 };
  }
  return null;
}

/** The human-facing name for an agent. The API's enriched `/agents` and
 * `/leaderboard` serve `display_name` ("Zero"); `name` is the internal
 * registry id ("neurofour-net14"). Rendering `display_name` directly breaks
 * against any backend response that omits it (e.g. a not-yet-redeployed
 * server), showing a blank/dash where a name should be -- so every surface
 * derives the label through here and falls back to the id. */
export function displayName(agent: { name: string; display_name?: string | null }): string {
  return agent.display_name ?? agent.name;
}
