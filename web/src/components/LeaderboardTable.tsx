import type { LeaderboardAgent } from '../types';
import { Badge } from './Badge';
import { TIER_LABEL, formatBytes, formatFlops, formatLatency, formatPct, formatScore } from '../lib/format';

interface LeaderboardTableProps {
  agents: LeaderboardAgent[];
  /** The headline/flagship agent name, read from `leaderboard.headline.agent`
   * by the caller — never hardcoded here. */
  flagship?: string | null;
}

export function LeaderboardTable({ agents, flagship }: LeaderboardTableProps) {
  const sorted = [...agents].sort((a, b) => b.neurogolf_score - a.neurogolf_score);

  return (
    <div
      className="overflow-x-auto rounded-xl border border-[var(--border)]"
      tabIndex={0}
      role="region"
      aria-label="Leaderboard table, scrollable"
    >
      <table className="w-full min-w-[760px] text-left text-sm">
        <caption className="sr-only">NeuroFour leaderboard: agents ranked by NeuroFour score.</caption>
        {/* Explicit column widths (not left to auto-layout): the Agent
            column previously claimed ~360px of mostly-empty space while five
            numeric columns fought for what was left, and jammed together.
            Agent gets a fixed generous-but-bounded share; every numeric
            column gets an equal fixed share so values align in a clean
            right-aligned rail. */}
        <colgroup>
          <col style={{ width: '30%' }} />
          <col style={{ width: '9%' }} />
          <col style={{ width: '11%' }} />
          <col style={{ width: '10%' }} />
          <col style={{ width: '12%' }} />
          <col style={{ width: '10%' }} />
          <col style={{ width: '10%' }} />
          <col style={{ width: '8%' }} />
        </colgroup>
        <thead className="bg-[var(--surface-2)] text-xs uppercase tracking-wide text-[var(--ink-2)]">
          <tr>
            <th scope="col" className="whitespace-nowrap px-3 py-2 font-medium">
              Agent
            </th>
            <th scope="col" className="whitespace-nowrap px-3 py-2 font-medium">
              Kind
            </th>
            <th scope="col" className="whitespace-nowrap px-3 py-2 font-medium text-right">
              Optimality
            </th>
            <th scope="col" className="whitespace-nowrap px-3 py-2 font-medium text-right">
              Size
            </th>
            <th scope="col" className="whitespace-nowrap px-3 py-2 font-medium text-right">
              FLOPs/move
            </th>
            <th scope="col" className="whitespace-nowrap px-3 py-2 font-medium text-right">
              Latency
            </th>
            <th scope="col" className="whitespace-nowrap px-3 py-2 font-medium text-right">
              NeuroFour Score
            </th>
            <th scope="col" className="whitespace-nowrap px-3 py-2 font-medium">
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
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="tabular text-[var(--ink-muted-text)]">{i + 1}.</span>
                  <span>{a.display_name ?? a.name}</span>
                  {a.subtitle ? <span className="font-normal text-[var(--ink-muted-text)]">{a.subtitle}</span> : null}
                  {a.name === flagship ? <Badge variant="accent">flagship</Badge> : null}
                  {a.over_budget ? <Badge variant="warning">over budget</Badge> : null}
                  {a.pareto ? <Badge variant="good">pareto</Badge> : null}
                </div>
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
