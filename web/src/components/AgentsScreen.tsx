import { useAgents } from '../hooks/useAgents';
import { useLeaderboard } from '../hooks/useLeaderboard';
import { StatusBanner } from './StatusBanner';
import { AgentsGridSkeleton } from './Skeleton';
import { Badge } from './Badge';
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
      <div className="min-w-0">
        <h3 className="truncate text-base font-semibold text-[var(--ink)]">{agent.display_name}</h3>
        <p className="truncate text-xs text-[var(--ink-muted-text)]">{agent.subtitle}</p>
      </div>

      {/* ONE badge system, ONE row, ONE baseline (see Badge.tsx): status
          chips (champion/over budget/pareto) first, taxonomy chips
          (kind/tier) after -- no more separate top-corner "kind" chip
          floating at a different vertical rhythm than the rest. */}
      <div className="flex flex-wrap items-center gap-1.5">
        {isChampion ? <Badge variant="accent">champion</Badge> : null}
        {agent.over_budget ? <Badge variant="warning">over budget</Badge> : null}
        {agent.pareto ? <Badge variant="good">pareto</Badge> : null}
        <Badge variant="neutral">{KIND_LABEL[agent.kind]}</Badge>
        {agent.tier ? <Badge variant="neutral">{TIER_LABEL[agent.tier]}</Badge> : null}
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
        <AgentsGridSkeleton />
      ) : agentsState.data.length === 0 ? (
        <StatusBanner kind="empty" title="No agents registered yet" onRetry={agentsState.reload} />
      ) : (
        <>
          {/* Competing agents only, ranked by NeuroFour score -- an
              intentionally over-budget exact solver (Oracle) used to sort
              straight to card #1 by raw score even though the About page
              explicitly says it "never competes for the crown", burying the
              actual champion (Zero) at card #2 and losing the whole
              headline story. Oracle (and anything else marked over_budget)
              now gets its own clearly-labeled reference section below,
              never mixed into the ranked competing grid. */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {sortAgents(agentsState.data.filter((a) => !a.over_budget)).map((a) => (
              <AgentCard key={a.name} agent={a} isChampion={a.name === champion} onPlay={onPlay} />
            ))}
          </div>

          {agentsState.data.some((a) => a.over_budget) ? (
            <div className="flex flex-col gap-3 border-t border-[var(--border)] pt-6">
              <div>
                <h2 className="text-sm font-semibold text-[var(--ink)]">Reference — exact solver (not competing)</h2>
                <p className="text-xs text-[var(--ink-muted-text)]">
                  Runs well over the compute budget on purpose; it grades every other agent but never ranks for the
                  NeuroFour Score crown.
                </p>
              </div>
              {/* NOT `opacity-80` on the whole card -- CSS opacity dims TEXT
                  contrast along with everything else, and axe caught the
                  card's own already-AA-tuned `--ink-muted-text` labels
                  dropping to 3.68:1 under it. The section heading + border
                  above already signals "this is the de-emphasized reference
                  section"; the cards themselves stay at full, always-AA
                  contrast. */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {sortAgents(agentsState.data.filter((a) => a.over_budget)).map((a) => (
                  <AgentCard key={a.name} agent={a} isChampion={false} onPlay={onPlay} />
                ))}
              </div>
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
