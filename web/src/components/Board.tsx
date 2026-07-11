import { useEffect, useMemo, useRef, useState } from 'react';
import type { AnalyzeResult, BoardGrid, GameStatus, LastMove, PlayerId } from '../types';
import { EMPTY_BOARD_COLS, EMPTY_BOARD_ROWS } from '../lib/format';

interface BoardProps {
  board: BoardGrid;
  legalMoves: number[];
  toMove: PlayerId;
  status: GameStatus;
  winner: 0 | PlayerId;
  /** [row, col] pairs, as returned by the backend. */
  winningLine?: [number, number][] | null;
  lastMove?: LastMove | null;
  interactive: boolean;
  busy?: boolean;
  onDrop: (col: number) => void;
  analysis?: AnalyzeResult | null;
  /** True while a fresh `/analyze` request is in flight. When `analysis` is
   * already populated (a previous position's result, kept on-screen so the
   * badge row doesn't flicker back to bare arrows -- see PlayScreen's
   * effect), this only needs to add a subtle "refreshing" affordance. When
   * `analysis` is null (first analyze of a game, nothing to hold onto yet),
   * this drives a skeleton placeholder instead of bare arrows, so the very
   * first several-second wait doesn't read as a dead/hung control either. */
  analysisPending?: boolean;
  reducedMotion?: boolean;
}

const discClass: Record<PlayerId, string> = {
  1: 'bg-[var(--disc-1)] ring-[var(--disc-1-ring)]',
  2: 'bg-[var(--disc-2)] ring-[var(--disc-2-ring)]',
};

function playerName(p: 0 | PlayerId): string {
  if (p === 1) return 'Red';
  if (p === 2) return 'Yellow';
  return 'nobody';
}

export function Board({
  board,
  legalMoves,
  toMove,
  status,
  winner,
  winningLine,
  lastMove,
  interactive,
  busy,
  onDrop,
  analysis,
  analysisPending,
  reducedMotion,
}: BoardProps) {
  const legalSet = useMemo(() => new Set(legalMoves), [legalMoves]);
  const winSet = useMemo(() => {
    const s = new Set<string>();
    (winningLine ?? []).forEach(([row, col]) => s.add(`${row},${col}`));
    return s;
  }, [winningLine]);

  const buttonRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const [focusedCol, setFocusedCol] = useState<number>(() => legalMoves[0] ?? 0);

  useEffect(() => {
    if (!legalSet.has(focusedCol)) {
      const next = legalMoves[0];
      if (next !== undefined) setFocusedCol(next);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [legalMoves.join(',')]);

  const focusCol = (col: number) => {
    setFocusedCol(col);
    buttonRefs.current[col]?.focus();
  };

  const moveFocus = (delta: 1 | -1) => {
    if (legalMoves.length === 0) return;
    let idx = focusedCol;
    for (let i = 0; i < EMPTY_BOARD_COLS; i += 1) {
      idx = (idx + delta + EMPTY_BOARD_COLS) % EMPTY_BOARD_COLS;
      if (legalSet.has(idx)) {
        focusCol(idx);
        return;
      }
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!interactive) return;
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      moveFocus(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      moveFocus(1);
    } else if (e.key === 'Home') {
      e.preventDefault();
      const first = legalMoves[0];
      if (first !== undefined) focusCol(first);
    } else if (e.key === 'End') {
      e.preventDefault();
      const last = legalMoves[legalMoves.length - 1];
      if (last !== undefined) focusCol(last);
    }
  };

  // Total cells on the board (42 on the standard 7x6 grid) -- needed to turn a
  // raw scored value back into a plies-to-mate count (see the docstring in
  // app/solver/solver.py: a win placed as the k-th stone overall scores
  // SIZE - (k - 1), so k = SIZE - value + 1, and the number of plies from
  // *now* (this board has `stonesOnBoard` already down) until that stone is
  // placed is k - stonesOnBoard).
  const SIZE = EMPTY_BOARD_COLS * EMPTY_BOARD_ROWS;
  const stonesOnBoard = useMemo(() => board.reduce((n, row) => n + row.filter((c) => c !== 0).length, 0), [board]);

  const analysisFor = (col: number) => {
    if (!analysis) return null;
    const v = analysis.per_col[String(col)];
    if (v === undefined || v === null) return null;
    const isBest = analysis.best_col === col;
    let tone: 'good' | 'draw' | 'bad' = 'draw';
    if (v > 0) tone = 'good';
    else if (v < 0) tone = 'bad';
    // `analysis.exact` (see AnalyzeResult / app/main.py::analyze) is only
    // ever true for the audited exact solve, where `v` is a real mate-aware
    // score and a plies-to-mate count can be derived from it directly. Off
    // -book near-empty positions ("exact": false / mode "depth-limited") use
    // a depth-9 bounded static-eval search instead: `v` there is a raw,
    // heuristic alpha-beta score with no calibrated units, so deriving a
    // "mate in N" from it would just be a confident-looking guess -- flag it
    // as an estimate instead of fabricating a distance.
    const mateIn = tone !== 'draw' && analysis.exact ? SIZE - Math.abs(v) + 1 - stonesOnBoard : null;
    return { v, isBest, tone, mateIn, isEstimate: analysis.exact === false };
  };

  const showSkeleton = Boolean(analysisPending) && !analysis;

  return (
    <div className="w-full max-w-[min(92vw,560px)] lg:max-w-[640px] mx-auto select-none">
      {analysis || showSkeleton ? (
        <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 mb-2 text-xs text-[var(--ink-2)]">
          <span className="inline-flex items-center gap-1">
            <span aria-hidden="true" className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: 'var(--good)' }} />
            Winning move
          </span>
          <span className="inline-flex items-center gap-1">
            <span aria-hidden="true" className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: 'var(--ink-muted)' }} />
            Drawing move
          </span>
          <span className="inline-flex items-center gap-1">
            <span aria-hidden="true" className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: 'var(--critical)' }} />
            Losing move
          </span>
          <span className="inline-flex items-center gap-1">
            <span aria-hidden="true">★</span> Solver&apos;s best move
          </span>
          {/* "est." used to repeat on every one of the 7 per-column badges
              below (vision review: "'est.' is repeated verbatim on all seven
              analyze badges") -- said once here in the legend instead; the
              per-badge label now only ever carries the mate count, never the
              estimate caveat (see the badge render loop). Screen-reader users
              still get the caveat per-column too, in each button's own
              aria-label, since that's read in isolation without this legend
              as context. */}
          {analysis?.exact === false ? (
            <span className="inline-flex items-center gap-1" style={{ color: 'var(--ink-muted-text)' }}>
              (fast estimate, not a proven result)
            </span>
          ) : null}
          {analysisPending ? (
            <span role="status" className="inline-flex items-center gap-1 font-medium" style={{ color: 'var(--ink-2)' }}>
              <svg className="inline-block h-3 w-3 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
                <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              </svg>
              {analysis ? 'Updating…' : 'Analyzing…'}
            </span>
          ) : null}
        </div>
      ) : null}
      {/* Single shared card for the analysis toolbar row AND the board grid
          below it -- both are direct children of this one container, so they
          inherit the identical horizontal inset (this card's own `p-2`) and
          both grids use the identical `gap-1.5` column gap. That pixel-for
          -pixel match is what makes badge N sit truly, visually directly
          over column N (previously the toolbar had no horizontal inset and a
          `gap-1` column gap while the board grid below had its own `p-2`
          inset and a `gap-1.5` gap -- two independent mismatches that pushed
          every badge slightly out of register with the column beneath it).
          Wrapping both in one bordered/backgrounded card also stops the
          toolbar from reading as a separate floating strip detached above
          the board: it's now visually inside the same panel. */}
      <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-2">
      <div
        role="toolbar"
        aria-label="Choose a column to drop your disc"
        aria-orientation="horizontal"
        onKeyDown={onKeyDown}
        className="grid gap-1.5"
        style={{ gridTemplateColumns: `repeat(${EMPTY_BOARD_COLS}, minmax(0,1fr))` }}
      >
        {Array.from({ length: EMPTY_BOARD_COLS }, (_, col) => {
          const legal = legalSet.has(col) && interactive && !busy;
          const a = analysisFor(col);
          const tabIndex = col === focusedCol ? 0 : -1;
          return (
            <button
              key={col}
              ref={(el) => {
                buttonRefs.current[col] = el;
              }}
              type="button"
              tabIndex={interactive ? tabIndex : -1}
              disabled={!legal}
              aria-disabled={!legal}
              aria-label={
                legalSet.has(col)
                  ? `Drop disc in column ${col + 1}${
                      a
                        ? `, solver says ${a.tone === 'good' ? 'winning' : a.tone === 'bad' ? 'losing' : 'drawing'}${a.isBest ? ', best move' : ''}${
                            a.mateIn ? ` in ${a.mateIn} ${a.mateIn === 1 ? 'move' : 'moves'}` : ''
                          }${a.isEstimate ? ' (fast estimate, not a proven result)' : ''}`
                        : ''
                    }`
                  : `Column ${col + 1}, full`
              }
              onClick={() => legal && onDrop(col)}
              onFocus={() => setFocusedCol(col)}
              className="group flex flex-col items-center gap-1.5 rounded-md py-2 text-xs font-medium
                bg-transparent border border-[var(--border)] text-[var(--ink-2)]
                enabled:hover:bg-[var(--surface-2)] enabled:hover:border-[var(--border-strong)]
                disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer transition-colors"
            >
              {a ? (
                <>
                  <span
                    className="inline-flex items-center gap-0.5 rounded px-1 tabular"
                    style={{
                      color:
                        a.tone === 'good'
                          ? 'var(--good-text)'
                          : a.tone === 'bad'
                            ? 'var(--critical)'
                            : 'var(--ink-muted-text)',
                    }}
                    title={`solver score ${a.v}`}
                  >
                    {a.isBest ? '★ ' : ''}
                    {a.tone === 'good' ? 'Win' : a.tone === 'bad' ? 'Loss' : 'Draw'}
                  </span>
                  {/* Disambiguates the badge above from the plain column
                      number below it (a starred "Loss" sitting directly over
                      a numeral like "4" can misread as "loss in 4 moves").
                      When the result is exact, show the real plies-to-mate.
                      The depth-limited-estimate caveat used to be repeated
                      here too ("est.", on all seven badges verbatim --
                      flagged by vision review as noise); it now appears once
                      in the legend above instead. It is still carried in
                      full in this button's own aria-label below, since that
                      is read standalone (without the legend as context) by
                      screen readers. */}
                  <span className="text-[10px] font-medium leading-tight" style={{ color: 'var(--ink-muted-text)' }}>
                    {a.mateIn ? `in ${a.mateIn}` : ' '}
                  </span>
                </>
              ) : showSkeleton ? (
                <span aria-hidden="true" className="flex flex-col items-center gap-1 py-[3px]">
                  <span className="h-[13px] w-8 animate-pulse rounded" style={{ backgroundColor: 'var(--ink-muted)', opacity: 0.35 }} />
                  <span className="h-[10px] w-5 animate-pulse rounded" style={{ backgroundColor: 'var(--ink-muted)', opacity: 0.25 }} />
                </span>
              ) : (
                <span aria-hidden="true">&darr;</span>
              )}
              <span className="tabular text-[var(--ink-muted-text)]">col {col + 1}</span>
            </button>
          );
        })}
      </div>

      {/* No own padding here (unlike the pre-fix version) -- it must share
          EXACTLY the outer card's p-2 inset with the toolbar row above (not
          add a second, larger inset of its own), or column N's disc cells
          drift out of pixel alignment with column N's badge again. */}
      <div
        className="mt-1.5 grid gap-1.5 rounded-xl"
        style={{
          gridTemplateColumns: `repeat(${EMPTY_BOARD_COLS}, minmax(0,1fr))`,
          backgroundColor: 'var(--board-frame)',
        }}
        aria-hidden="true"
      >
        {board.map((row, r) =>
          row.map((cell, c) => {
            const isWin = winSet.has(`${r},${c}`);
            const isLast = lastMove && lastMove.row === r && lastMove.col === c;
            return (
              <div
                key={`${r}-${c}`}
                className="relative aspect-square rounded-full"
                style={{ backgroundColor: cell === 0 ? 'var(--disc-empty)' : undefined }}
              >
                {cell !== 0 ? (
                  <div
                    className={`absolute inset-0 rounded-full ring-2 ${discClass[cell]} ${
                      isLast && !reducedMotion ? 'disc-drop' : ''
                    } ${isWin ? 'win-cell' : ''}`}
                    style={isLast ? ({ '--drop-from': '-320px' } as React.CSSProperties) : undefined}
                  />
                ) : null}
              </div>
            );
          }),
        )}
      </div>
      </div>

      {/* Accessible textual board summary for assistive tech that wants row/column
          detail. This is deliberately NOT a <table> element: a <table> lays
          itself out to its content's intrinsic width regardless of the
          sr-only rule's `width:1px` (table auto-layout sizes from cell
          content, not from any CSS width on the table box), so a real
          <table> here expanded to ~1229px and caused genuine horizontal page
          scroll at every viewport whenever a board was present. ARIA
          role="table"/"row"/"cell" on plain <div>s preserves the identical
          screen-reader semantics (announced the same way as a native table)
          without that intrinsic-content-width behavior, so the sr-only
          1x1px clip actually constrains it. */}
      <div className="sr-only" role="table" aria-label={`Connect Four board, row 1 is the top, row ${EMPTY_BOARD_ROWS} is the bottom. Turn: ${
        status === 'in_progress' ? `${playerName(toMove)} to move` : status === 'draw' ? 'draw' : `${playerName(winner)} won`
      }.`}
      >
        {board.map((row, r) => (
          <div role="row" key={r}>
            {row.map((cell, c) => (
              <div role="cell" key={c}>
                Row {r + 1}, column {c + 1}: {cell === 0 ? 'empty' : `${playerName(cell)} disc`}
                {winSet.has(`${r},${c}`) ? ' (part of winning line)' : ''}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
