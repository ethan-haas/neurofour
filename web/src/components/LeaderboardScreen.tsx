import { useState } from 'react';
import { useLeaderboard } from '../hooks/useLeaderboard';
import { StatusBanner } from './StatusBanner';
import { ParetoPlot, type CostAxis } from './ParetoPlot';
import { LeaderboardTable } from './LeaderboardTable';

export function LeaderboardScreen() {
  const state = useLeaderboard();
  const [axis, setAxis] = useState<CostAxis>('size');

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-6">
      <div>
        <h1 className="text-lg font-semibold text-[var(--ink)]">Leaderboard</h1>
        <p className="text-sm text-[var(--ink-2)]">
          Ranked by <span className="font-semibold text-[var(--ink)]">NeuroFour score</span> — strength per byte, not raw strength.
        </p>
      </div>

      {state.status === 'error' ? (
        <StatusBanner kind="error" title="Could not load the leaderboard" detail={state.message} onRetry={state.reload} />
      ) : state.status === 'loading' || state.status === 'idle' ? (
        <StatusBanner kind="loading" title="Loading leaderboard…" />
      ) : state.data.agents.length === 0 ? (
        <StatusBanner kind="empty" title="No scored agents yet" detail="Run the bench to populate leaderboard.json." onRetry={state.reload} />
      ) : (
        <>
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4">
            <ParetoPlot agents={state.data.agents} axis={axis} onAxisChange={setAxis} flagship={state.data.headline.agent} />
          </div>
          <LeaderboardTable agents={state.data.agents} flagship={state.data.headline.agent} />
        </>
      )}
    </div>
  );
}
