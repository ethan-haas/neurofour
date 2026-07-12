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

export function formatFlops(flops: number): string {
  if (flops < 1000) return `${flops}`;
  if (flops < 1_000_000) return `${(flops / 1000).toFixed(flops % 1000 === 0 ? 0 : 1)}K`;
  return `${(flops / 1_000_000).toFixed(flops % 1_000_000 === 0 ? 0 : 2)}M`;
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
