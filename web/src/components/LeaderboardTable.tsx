import type { LeaderboardAgent } from '../types';
import { formatBytes, formatFlops, formatLatency, formatPct, formatScore } from '../lib/format';

interface LeaderboardTableProps {
  agents: LeaderboardAgent[];
  /** The headline/flagship agent name, read from `leaderboard.headline.agent`
   * by the caller — never hardcoded here. */
  flagship?: string | null;
}

const TIER_LABEL: Record<LeaderboardAgent['tier'], string> = {
  nano: 'Nano ≤4KB',
  micro: 'Micro ≤32KB',
  mini: 'Mini ≤256KB',
  small: 'Small ≤2MB',
  open: 'Open',
};

export function LeaderboardTable({ agents, flagship }: LeaderboardTableProps) {
  const sorted = [...agents].sort((a, b) => b.neurogolf_score - a.neurogolf_score);

  return (
    <div
      className="overflow-x-auto rounded-xl border border-[var(--border)]"
      tabIndex={0}
      role="region"
      aria-label="Leaderboard table, scrollable"
    >
      <table className="w-full min-w-[720px] text-left text-sm">
        <caption className="sr-only">NeuroFour leaderboard: agents ranked by NeuroFour score.</caption>
        <thead className="bg-[var(--surface-2)] text-xs uppercase tracking-wide text-[var(--ink-2)]">
          <tr>
            <th scope="col" className="px-3 py-2 font-medium">
              Agent
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Kind
            </th>
            <th scope="col" className="px-3 py-2 font-medium text-right">
              Optimality
            </th>
            <th scope="col" className="px-3 py-2 font-medium text-right">
              Size
            </th>
            <th scope="col" className="px-3 py-2 font-medium text-right">
              FLOPs/move
            </th>
            <th scope="col" className="px-3 py-2 font-medium text-right">
              Latency
            </th>
            <th scope="col" className="px-3 py-2 font-medium text-right">
              NeuroFour Score
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Tier
            </th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((a, i) => (
            <tr
              key={a.name}
              className="border-t border-[var(--border)]"
              style={{ backgroundColor: a.name === flagship ? 'var(--surface-2)' : undefined }}
            >
              <th scope="row" className="px-3 py-2 font-medium text-[var(--ink)]">
                <span className="tabular text-[var(--ink-muted-text)] mr-1.5">{i + 1}.</span>
                {a.name}
                {a.name === flagship ? (
                  <span className="ml-1.5 rounded px-1 py-0.5 text-[10px] font-semibold" style={{ backgroundColor: 'var(--accent-solid)', color: 'var(--accent-solid-ink)' }}>
                    flagship
                  </span>
                ) : null}
                {a.pareto ? (
                  <span className="ml-1.5 rounded px-1 py-0.5 text-[10px] font-semibold" style={{ color: 'var(--good-text)', border: '1px solid var(--good)' }}>
                    pareto
                  </span>
                ) : null}
                {a.over_budget ? (
                  <span className="ml-1.5 rounded px-1 py-0.5 text-[10px] font-semibold" style={{ color: 'var(--critical)', border: '1px solid var(--critical)' }}>
                    over budget
                  </span>
                ) : null}
              </th>
              <td className="px-3 py-2 text-[var(--ink-2)]">{a.kind}</td>
              <td className="px-3 py-2 text-right tabular text-[var(--ink)]">{formatPct(a.optimality)}</td>
              <td className="px-3 py-2 text-right tabular text-[var(--ink)]">{formatBytes(a.size_bytes)}</td>
              <td className="px-3 py-2 text-right tabular text-[var(--ink)]">{formatFlops(a.flops_per_move)}</td>
              <td className="px-3 py-2 text-right tabular text-[var(--ink)]">{formatLatency(a.latency_ms)}</td>
              <td className="px-3 py-2 text-right tabular font-semibold text-[var(--ink)]">{formatScore(a.neurogolf_score)}</td>
              <td className="px-3 py-2 text-[var(--ink-2)]">{TIER_LABEL[a.tier]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
