import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useAgents } from '../hooks/useAgents';
import { useGame } from '../hooks/useGame';
import { api, ApiError } from '../lib/api';
import { lastMoveFromGame } from '../lib/format';
import { usePrefersReducedMotion } from '../lib/theme';
import { Board } from './Board';
import { NewGamePanel } from './NewGamePanel';
import { StatusBanner } from './StatusBanner';
import type { AgentManifest, AnalyzeResult, GameState } from '../types';

/** The registry id (e.g. "neurofour-net14") must never be shown to a user --
 * the rest of the app (Agents cards, Leaderboard table/chart) always shows
 * `display_name` ("Zero"), so the Play status line has to agree with it too
 * (was leaking the raw id verbatim: "Yellow to move (neurofour-net14)").
 * Falls back to the id only if the agents list hasn't loaded yet / doesn't
 * contain it, never to a blank. */
function playerLabel(game: GameState, seat: 1 | 2, agents: AgentManifest[]): string {
  const agentId = seat === 1 ? game.first_agent : game.second_agent;
  if (!agentId) return 'You';
  return agents.find((a) => a.name === agentId)?.display_name ?? agentId;
}

const AGENT_MOVE_DELAY_MS = 550;

/** Small inline spinner -- respects `prefers-reduced-motion` for free via the
 * global `animation-duration: 0.001ms !important` override in index.css, so
 * it never needs its own reduced-motion branch. Used as a visible "this is
 * genuinely working, not stalled" affordance next to the Analyze pending
 * text and on the board's own badge row (see Board.tsx). */
function Spinner({ label }: { label?: string }) {
  return (
    <svg
      className="inline-block h-3 w-3 animate-spin"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden={label ? undefined : true}
      role={label ? 'img' : undefined}
      aria-label={label}
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

/** Turn a game's raw column-history into "Red -> col 3 / Yellow -> col 5"
 * rows, oldest first. Red always moves first (ply 0, 2, 4, ...). Genuine,
 * data-driven content for the left column -- not filler -- particularly in
 * agent-vs-agent watch mode, where moves happen on a timer and a spectator
 * has no other way to review what already happened. */
function buildMoveHistory(moves: number[]): { ply: number; player: 1 | 2; col: number }[] {
  return moves.map((col, i) => ({ ply: i + 1, player: (i % 2 === 0 ? 1 : 2) as 1 | 2, col }));
}

interface PlayScreenProps {
  /** Preselect this agent as the opponent -- set by the Agents screen's
   * "Play" button (via App.tsx), cleared automatically once the user
   * navigates to any screen other than Play (see App.tsx::handleScreen). */
  presetOpponent?: string | null;
}

export function PlayScreen({ presetOpponent }: PlayScreenProps = {}) {
  const agentsState = useAgents();
  const { game, loading, busy, error, newGame, playMove, requestAgentMove, clearError } = useGame();
  const reducedMotion = usePrefersReducedMotion();

  const [analyzeOn, setAnalyzeOn] = useState(false);
  const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [analysisPending, setAnalysisPending] = useState(false);
  const [watching, setWatching] = useState(false);
  const [liveMessage, setLiveMessage] = useState('');

  const agentTimerRef = useRef<number | null>(null);
  const lastAnnouncedRef = useRef<string>('');
  const lastAnalyzedGameIdRef = useRef<string | null>(null);

  const isWatchMode = Boolean(game?.first_agent) && Boolean(game?.second_agent);
  const agentsList = agentsState.status === 'success' ? agentsState.data : [];

  // The backend doesn't return a "last move" field — derive it from board + moves.
  const lastMove = useMemo(() => (game ? lastMoveFromGame(game.board, game.moves) : null), [game]);
  const moveHistory = useMemo(() => (game ? buildMoveHistory(game.moves) : []), [game]);

  // Announce moves / results via aria-live.
  useEffect(() => {
    if (!game) return;
    const key = `${game.id}:${lastMove ? `${lastMove.player}-${lastMove.col}` : 'start'}:${game.status}`;
    if (lastAnnouncedRef.current === key) return;
    lastAnnouncedRef.current = key;

    if (!lastMove) {
      setLiveMessage(
        `New game started. Red: ${playerLabel(game, 1, agentsList)}. Yellow: ${playerLabel(game, 2, agentsList)}. ${playerLabel(game, 1, agentsList)} moves first.`,
      );
      return;
    }
    const mover = lastMove.player === 1 ? 'Red' : 'Yellow';
    let msg = `${mover} dropped in column ${lastMove.col + 1}.`;
    if (game.status === 'won') {
      msg += ` ${game.winner === 1 ? 'Red' : 'Yellow'} wins!`;
    } else if (game.status === 'draw') {
      msg += ' The game is a draw.';
    }
    setLiveMessage(msg);
  }, [game, lastMove, agentsList]);

  // Auto-advance agent turns (human-vs-agent auto-reply, and agent-vs-agent watch mode).
  useEffect(() => {
    if (agentTimerRef.current) {
      window.clearTimeout(agentTimerRef.current);
      agentTimerRef.current = null;
    }
    if (!game || game.status !== 'in_progress' || busy) return;
    if (!game.to_move_is_agent) return;
    if (isWatchMode && !watching) return;

    agentTimerRef.current = window.setTimeout(() => {
      void requestAgentMove();
    }, AGENT_MOVE_DELAY_MS);

    return () => {
      if (agentTimerRef.current) window.clearTimeout(agentTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [game?.id, game?.status, game?.player_to_move, game?.to_move_is_agent, busy, isWatchMode, watching]);

  // Fetch solver analysis for the position to move, when Analyze is toggled
  // on. This effect is the ONLY place `/analyze` is called; every dependency
  // it reads live from `game` (id, moves.length, status) is listed so the
  // browser refetches on every position change, not just when the checkbox
  // itself flips.
  //
  // ROOT CAUSE (reproduced against the real backend, not the fixture mock):
  // the depth-limited analysis the backend runs for near-empty/off-book
  // positions (app/main.py::_bounded_analyze — a depth-9 pure-Python search
  // per legal column) genuinely takes several seconds. The previous version
  // of this effect fetched correctly but gave the user ZERO feedback while
  // waiting — no pending state, no spinner, nothing — so during that
  // multi-second window the toggle looked and screenshotted as a dead
  // control (pixel-identical ON vs OFF) even though a request really was in
  // flight. It also only ever *ignored* a stale/superseded response instead
  // of actually cancelling the underlying fetch, so rapidly re-toggling
  // could leave multiple slow requests running concurrently in the
  // background indefinitely.
  useEffect(() => {
    if (!analyzeOn || !game || game.status !== 'in_progress') {
      setAnalysis(null);
      setAnalysisError(null);
      setAnalysisPending(false);
      lastAnalyzedGameIdRef.current = null;
      return;
    }
    // Deliberately do NOT clear `analysis` here on every dependency change --
    // keeping the previous render's overlay in place while the new request
    // is in flight is what stops the badge row from flickering back to bare
    // "↓" arrows on every move (see DEFECT 2 in the vision pass). The one
    // exception is a genuinely DIFFERENT game (a brand-new board, e.g. after
    // "New game" with Analyze still checked): that stale overlay belongs to
    // a different position entirely, not just an older one, so showing it
    // even briefly would be actively misleading rather than merely dated.
    if (lastAnalyzedGameIdRef.current !== game.id) {
      setAnalysis(null);
    }
    lastAnalyzedGameIdRef.current = game.id;
    const controller = new AbortController();
    setAnalysisError(null);
    setAnalysisPending(true);
    api
      // "scored" is the mate-distance-aware oracle the backend now always uses
      // for optimal_cols/best_col regardless of mode; request it explicitly
      // so a future backend change to per_col's display scale stays truthful
      // too (per_col's sign, which is all this UI reads, is unaffected).
      .analyze({ board: game.moves, mode: 'scored' }, { signal: controller.signal })
      .then((res) => {
        // Defensive: never render a broken overlay for a terminal/no-move
        // response even if one slips through (this effect already gates on
        // `status === 'in_progress'` before firing, so the backend
        // shouldn't return `terminal: true` here, but per_col is the thing
        // every column button reads, so guard its shape too).
        if (res.terminal || !res.per_col || Object.keys(res.per_col).length === 0) {
          setAnalysis(null);
        } else {
          setAnalysis(res);
        }
        setAnalysisPending(false);
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === 'AbortError') return; // superseded, not a real error
        setAnalysisError(err instanceof ApiError ? err.message : 'Analysis unavailable.');
        setAnalysis(null);
        setAnalysisPending(false);
      });
    return () => {
      // Actually cancel the in-flight request (not just ignore its result)
      // so toggling off / changing position doesn't leave slow requests
      // running in the background.
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analyzeOn, game?.id, game?.moves.length, game?.status]);

  const handleStart = useCallback(
    (first: string | null, second: string | null) => {
      setWatching(false);
      lastAnnouncedRef.current = '';
      void newGame(first, second);
    },
    [newGame],
  );

  const handleDrop = useCallback(
    (col: number) => {
      void playMove(col);
    },
    [playMove],
  );

  // "Play again" on the result card re-starts with the SAME two
  // participants the just-finished game had (not whatever the sidebar
  // selects currently show -- those are independent controlled state and
  // may have been changed mid-game, e.g. via a preset-opponent navigation).
  const handlePlayAgain = useCallback(() => {
    if (!game) return;
    handleStart(game.first_agent, game.second_agent);
  }, [game, handleStart]);

  const humanCanPlay = Boolean(game) && game!.status === 'in_progress' && !game!.to_move_is_agent && !busy;

  // NOTE: `presetOpponent` is intentionally NOT cleared here. NewGamePanel's
  // own agents-loading gate (the StatusBanner branch below) can mount well
  // after this component's first commit -- clearing the preset eagerly on
  // PlayScreen's own mount raced NewGamePanel's later mount and cleared the
  // preset before it was ever read. App.tsx's `handleScreen` already clears
  // `presetOpponent` whenever the user navigates to any screen OTHER than
  // Play, which is the only time it needs to stop applying.

  return (
    // Desktop (lg): the row's two columns are laid out as a `flex-row` with
    // BOTH columns sized to their own content (`flex-none`, not a
    // `320px_minmax(0,1fr)` grid) and the pair is centered as ONE unit via
    // `lg:justify-center` on a full-width row -- see the horizontal-layout
    // history below. Vertically, the row top-aligns its children
    // (`lg:items-start`), not centers them.
    //
    // Why not centered: the left column (opponent picker + Status card) is
    // ~330px tall; the board card is ~640-700px. `items-center` (the
    // previous value) centered that short column against the tall one,
    // which put HALF the height difference above the left column and half
    // below it -- a symmetric void above AND below a card that otherwise
    // reads as sitting at a totally arbitrary vertical offset, independently
    // flagged by vision review as "reads like a centering bug, not a
    // decision" (worst in dark, where the empty bands show up as flat black
    // gaps with nothing to anchor them to). Top-aligning both columns to the
    // board's own top edge kills the upper void outright and gives the
    // layout an actual anchor (both columns start together, like a
    // side-panel next to a document). The Move history card below (which
    // grows with the game) then uses the column's own remaining height for
    // real content instead of the layout depending on a leftover void being
    // "fine" -- see its own comment.
    //
    // Horizontal composition (unchanged from the previous pass): the
    // previous `320px + minmax(0,1fr)` grid gave the right track literally
    // all the remaining space up to `max-w-6xl` (1152px), but the board card
    // inside it was `w-fit`/self-centered at ~560-640px -- so that oversized
    // 1fr track showed up on screen as dead space on BOTH sides of the
    // board: a gutter between the left panel and the board, and a second,
    // larger gap between the board and the container's right edge
    // (independently confirmed via screenshot at 1440: ~140px gutter +
    // ~260px right margin, i.e. the right third of the viewport unused).
    // Sizing both columns to their content and centering the resulting
    // (left-panel + gap + board-card) block as a whole removes the
    // mismatched track: the empty space that remains is symmetric outer
    // margin (intentional framing), not an internal gutter or a one-sided
    // stranded third. Purely a desktop change -- no `lg:` prefix touches the
    // base mobile flex-col layout, which stays exactly as it was.
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-6 lg:min-h-[calc(100vh-6rem)] lg:max-w-none lg:flex-row lg:items-start lg:justify-center lg:gap-10 lg:py-10">
      <div aria-live="polite" role="status" className="sr-only">
        {liveMessage}
      </div>

      <div className="flex w-full flex-col gap-4 lg:w-[340px] lg:flex-none">
        {agentsState.status === 'error' ? (
          <StatusBanner kind="error" title="Could not load agents" detail={agentsState.message} onRetry={agentsState.reload} />
        ) : agentsState.status === 'loading' || agentsState.status === 'idle' ? (
          <StatusBanner kind="loading" title="Loading agents…" />
        ) : agentsState.data.length === 0 ? (
          <StatusBanner kind="empty" title="No agents registered yet" onRetry={agentsState.reload} />
        ) : (
          <NewGamePanel
            agents={agentsState.data}
            onStart={handleStart}
            busy={loading}
            presetOpponent={presetOpponent}
          />
        )}

        {game ? (
          <div className="flex flex-col gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 text-sm">
            <div className="flex items-center justify-between">
              <span className="font-medium text-[var(--ink)]">Status</span>
              <span className="font-medium text-[var(--ink-2)]">
                {game.status === 'in_progress'
                  ? `${game.player_to_move === 1 ? 'Red' : 'Yellow'} to move (${playerLabel(game, game.player_to_move ?? 1, agentsList)})`
                  : game.status === 'draw'
                    ? 'Draw'
                    : `${game.winner === 1 ? 'Red' : 'Yellow'} won`}
              </span>
            </div>

            <label className="mt-2 flex items-center justify-between gap-2">
              <span className="font-medium text-[var(--ink)]">Analyze</span>
              <input
                type="checkbox"
                checked={analyzeOn}
                onChange={(e) => setAnalyzeOn(e.target.checked)}
                aria-describedby="analyze-hint"
              />
            </label>
            <p id="analyze-hint" className="text-xs font-medium text-[var(--ink-2)]">
              Overlays the solver&apos;s per-column evaluation as an analysis aid — it does not play for you.
            </p>
            {analysisPending ? (
              <p role="status" className="flex items-center gap-1.5 text-xs font-medium text-[var(--ink-2)]">
                <Spinner />
                Analyzing position…
              </p>
            ) : null}
            {analysisError ? <p role="alert" className="text-xs" style={{ color: 'var(--critical)' }}>{analysisError}</p> : null}

            {isWatchMode && game.status === 'in_progress' ? (
              <button
                type="button"
                onClick={() => setWatching((w) => !w)}
                className="mt-2 rounded-md border border-[var(--border-strong)] px-3 py-1.5 text-sm font-medium cursor-pointer
                  hover:bg-[var(--surface-2)]"
              >
                {watching ? 'Pause watch' : 'Watch agent vs agent'}
              </button>
            ) : null}
          </div>
        ) : null}

        {/* Move history -- genuine content, not filler: once the row
            top-aligns (see the layout comment above), the short left column
            has real leftover height next to the taller board, and a
            scrollable move log is something a spectator (especially in
            watch mode, where moves play on a timer with no other record)
            can actually use. Only rendered once there's at least one move,
            so the empty-board state stays exactly as clean as it was.
            `flex-1` + `overflow-y-auto` on the list lets it grow with the
            column instead of the column growing unboundedly with a long
            game. */}
        {game && moveHistory.length > 0 ? (
          <div className="flex min-h-0 flex-1 flex-col gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 text-sm lg:flex-1">
            <span className="font-medium text-[var(--ink)]">Moves</span>
            <ol className="flex-1 min-h-0 overflow-y-auto pr-1 text-xs text-[var(--ink-2)] lg:max-h-[420px]">
              {moveHistory.map((m) => (
                <li
                  key={m.ply}
                  className="flex items-center justify-between gap-2 border-b border-[var(--border)] py-1 last:border-b-0"
                >
                  <span className="tabular text-[var(--ink-muted-text)]">{m.ply}.</span>
                  <span className="flex-1 font-medium text-[var(--ink)]">
                    {m.player === 1 ? 'Red' : 'Yellow'}
                  </span>
                  <span className="tabular text-[var(--ink-2)]">col {m.col + 1}</span>
                </li>
              ))}
            </ol>
          </div>
        ) : null}
      </div>

      {/* Fixed to an explicit width matching the board's own `lg:max-w`
          (640px) plus this card's `p-8` inset (2 x 2rem = 64px), NOT
          `w-fit`/shrink-to-fit -- the Board component's root has `w-full`
          (100%), and inside a shrink-to-fit flex item a percentage-width
          descendant doesn't contribute to the parent's intrinsic size (a
          well-known CSS circularity), which was collapsing the whole grid
          toward its min-content (a few px per column) instead of filling
          the card. An explicit width breaks that circularity while still
          sizing the card to "hug the board" in effect. Paired with the
          outer row's `lg:justify-center`, this is what makes the whole
          (left panel + board card) block center as one unit instead of the
          board card floating, off-balance, inside an oversized right
          track. */}
      {/* Mobile (base): render the board FIRST (`order-first`) so the core
          interactive surface is at the top of the scroll instead of buried
          below the opponent picker, Status card, and an unboundedly-growing
          Moves list (each move otherwise pushed the board further down until
          only the column headers peeked above the fold). Desktop restores the
          natural DOM order (`lg:order-none`), where the two columns sit
          side-by-side and ordering is irrelevant. */}
      <div className="order-first flex w-full flex-1 flex-col items-center gap-4 lg:order-none lg:flex-none lg:w-[704px] lg:max-w-full lg:rounded-2xl lg:border lg:border-[var(--border)] lg:bg-[var(--surface)] lg:p-8">
        {error ? (
          <div role="alert" className="w-full max-w-[640px] rounded-md border px-3 py-2 text-sm" style={{ borderColor: 'var(--critical)', color: 'var(--critical)' }}>
            {error}
            <button type="button" onClick={clearError} className="ml-2 underline cursor-pointer">
              dismiss
            </button>
          </div>
        ) : null}

        {loading && !game ? (
          <StatusBanner kind="loading" title="Starting game…" bare />
        ) : !game ? (
          <StatusBanner kind="empty" title="No game yet" detail="Choose your opponents, then start a new game." bare />
        ) : (
          <>
            {game.status !== 'in_progress' ? (
              // A real result card (M9), not a thin one-line banner: large
              // type, a winner-tinted background + border (not the analyze
              // legend's green -- see --win-ring's own comment for why that
              // collided with the winning-disc ring color), and a primary
              // "Play again" action so finishing a game isn't a dead end
              // that requires reaching back into the sidebar form.
              <div
                role="status"
                className="flex w-full max-w-[640px] flex-col items-center gap-3 rounded-xl border-2 px-6 py-6 text-center"
                style={{
                  borderColor:
                    game.status === 'draw'
                      ? 'var(--border-strong)'
                      : game.winner === 1
                        ? 'var(--disc-1-ring)'
                        : 'var(--disc-2-ring)',
                  backgroundColor:
                    game.status === 'draw'
                      ? 'var(--surface-2)'
                      : `color-mix(in srgb, ${game.winner === 1 ? 'var(--disc-1-ring)' : 'var(--disc-2-ring)'} 12%, var(--surface))`,
                }}
              >
                <div className="flex items-center gap-2.5">
                  {game.status !== 'draw' ? (
                    <span
                      aria-hidden="true"
                      className="inline-block h-5 w-5 shrink-0 rounded-full border-2"
                      style={{
                        backgroundColor: game.winner === 1 ? 'var(--disc-1)' : 'var(--disc-2)',
                        borderColor: game.winner === 1 ? 'var(--disc-1-ring)' : 'var(--disc-2-ring)',
                      }}
                    />
                  ) : null}
                  <p className="text-2xl font-bold text-[var(--ink)]">
                    {game.status === 'draw' ? "It's a draw." : `${game.winner === 1 ? 'Red' : 'Yellow'} wins!`}
                  </p>
                </div>
                <p className="text-xs text-[var(--ink-muted-text)]">
                  {playerLabel(game, 1, agentsList)} vs {playerLabel(game, 2, agentsList)}
                </p>
                <button
                  type="button"
                  onClick={handlePlayAgain}
                  className="mt-1 rounded-md px-4 py-2 text-sm font-semibold cursor-pointer"
                  style={{ backgroundColor: 'var(--accent-solid)', color: 'var(--accent-solid-ink)' }}
                >
                  Play again
                </button>
              </div>
            ) : null}

            <Board
              board={game.board}
              legalMoves={game.legal_moves}
              toMove={game.player_to_move}
              status={game.status}
              winner={game.winner}
              winningLine={game.winning_line}
              lastMove={lastMove}
              interactive={humanCanPlay}
              busy={busy}
              onDrop={handleDrop}
              analysis={analyzeOn ? analysis : null}
              analysisPending={analyzeOn ? analysisPending : false}
              reducedMotion={reducedMotion}
            />
          </>
        )}
      </div>
    </div>
  );
}
