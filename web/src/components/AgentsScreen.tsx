import { useAgents } from '../hooks/useAgents';
import { useLeaderboard } from '../hooks/useLeaderboard';
import { StatusBanner } from './StatusBanner';
import type { AgentManifest } from '../types';
import { TIER_LABEL, formatBytes, formatFlops, formatLatency, formatPct } from '../lib/format';

const KIND_LABEL: Record<AgentManifest['kind'], string> = {
  table: 'Table',
  nn: 'Neural net',
  search: 'Search',
  heuristic: 'Heuristic',
  random: 'Random',
};

interface AgentsScreenProps {
  onPlay: (agentName: string) => void;
}

function sortAgents(agents: AgentManifest[]): AgentManifest[] {
  return [...agents].sort((a, b) => {
    const sa = a.neurogolf_score ?? -1;
    const sb = b.neurogolf_score ?? -1;
    return sb - sa;
  });
}

function AgentCard({ agent, isChampion, onPlay }: { agent: AgentManifest; isChampion: boolean; onPlay: (name: string) => void }) {
  return (
    <div className="flex flex-col gap-2.5 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 transition-colors hover:border-[var(--border-strong)]">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="truncate text-base font-semibold text-[var(--ink)]">{agent.display_name}</h3>
          <p className="truncate text-xs text-[var(--ink-muted-text)]">{agent.subtitle}</p>
        </div>
        <span
          className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold"
          style={{ backgroundColor: 'var(--surface-2)', color: 'var(--ink-2)' }}
        >
          {KIND_LABEL[agent.kind]}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {isChampion ? (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
            style={{ backgroundColor: 'var(--accent-solid)', color: 'var(--accent-solid-ink)' }}
          >
            champion
          </span>
        ) : null}
        {agent.pareto ? (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
            style={{ color: 'var(--good-text)', border: '1px solid var(--good)' }}
          >
            pareto
          </span>
        ) : null}
        {agent.over_budget ? (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
            style={{ color: 'var(--critical)', border: '1px solid var(--critical)' }}
          >
            over budget
          </span>
        ) : null}
        {agent.tier ? (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
            style={{ backgroundColor: 'var(--surface-2)', color: 'var(--ink-2)' }}
          >
            {TIER_LABEL[agent.tier]}
          </span>
        ) : null}
      </div>

      <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs text-[var(--ink-2)]">
        <div>
          <dt className="text-[var(--ink-muted-text)]">Optimality</dt>
          <dd className="tabular font-medium text-[var(--ink)]">
            {agent.optimality != null ? formatPct(agent.optimality) : '—'}
          </dd>
        </div>
        <div>
          <dt className="text-[var(--ink-muted-text)]">Size</dt>
          <dd className="tabular font-medium text-[var(--ink)]">{formatBytes(agent.size_bytes)}</dd>
        </div>
        <div>
          <dt className="text-[var(--ink-muted-text)]">FLOPs/move</dt>
          <dd className="tabular font-medium text-[var(--ink)]">{formatFlops(agent.flops_per_move)}</dd>
        </div>
        <div>
          <dt className="text-[var(--ink-muted-text)]">Latency</dt>
          <dd className="tabular font-medium text-[var(--ink)]">
            {agent.latency_ms != null ? formatLatency(agent.latency_ms) : '—'}
          </dd>
        </div>
        <div>
          <dt className="text-[var(--ink-muted-text)]">Elo</dt>
          <dd className="tabular font-medium text-[var(--ink)]">{agent.elo ?? '—'}</dd>
        </div>
        <div>
          <dt className="text-[var(--ink-muted-text)]">NeuroFour score</dt>
          <dd className="tabular font-medium text-[var(--ink)]">
            {agent.neurogolf_score != null ? agent.neurogolf_score.toFixed(3) : '—'}
          </dd>
        </div>
      </dl>

      <button
        type="button"
        onClick={() => onPlay(agent.name)}
        className="mt-1 self-start rounded-md border border-[var(--border-strong)] px-3 py-1.5 text-xs font-semibold
          cursor-pointer hover:bg-[var(--surface-2)]"
      >
        Play against {agent.display_name}
      </button>
    </div>
  );
}

export function AgentsScreen({ onPlay }: AgentsScreenProps) {
  const agentsState = useAgents();
  const leaderboardState = useLeaderboard();
  const champion = leaderboardState.status === 'success' ? leaderboardState.data.headline.agent : null;

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-6">
      <div>
        <h1 className="text-lg font-semibold text-[var(--ink)]">Agents</h1>
        <p className="text-sm text-[var(--ink-2)]">
          Every registered agent, ranked by <span className="font-semibold text-[var(--ink)]">NeuroFour score</span> (strength per
          byte and FLOP, not raw strength). Pick a card to play against it.
        </p>
      </div>

      {agentsState.status === 'error' ? (
        <StatusBanner kind="error" title="Could not load agents" detail={agentsState.message} onRetry={agentsState.reload} />
      ) : agentsState.status === 'loading' || agentsState.status === 'idle' ? (
        <StatusBanner kind="loading" title="Loading agents…" />
      ) : agentsState.data.length === 0 ? (
        <StatusBanner kind="empty" title="No agents registered yet" onRetry={agentsState.reload} />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {sortAgents(agentsState.data).map((a) => (
            <AgentCard key={a.name} agent={a} isChampion={a.name === champion} onPlay={onPlay} />
          ))}
        </div>
      )}
    </div>
  );
}
