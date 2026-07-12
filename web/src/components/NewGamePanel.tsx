import { useEffect, useState } from 'react';
import type { AgentManifest } from '../types';
import { AgentPicker, HUMAN } from './AgentPicker';
import { TIER_LABEL, formatBytes, formatFlops, formatLatency, formatPct } from '../lib/format';

/** Full-width strip of the selected agent's stats (C1). This used to live
 * INSIDE AgentPicker, crammed into that picker's own ~145px-wide select
 * column on desktop -- six stat cells stacked 2-per-row in a column that
 * narrow collided into unreadable text ("Optimality"/"Size" ran together
 * into "OptimalitSize", "FLOPs/move" overran the card's border). Rendered
 * here instead, spanning the FULL width of the config card below BOTH
 * selects, the same six cells get real room: `min-w-0` on each cell lets
 * long values shrink instead of overflowing, and the grid goes from 2 tight
 * columns (mobile) to all 6 across in one row once there's room (desktop). */
function AgentStatStrip({ label, agent }: { label: string; agent: AgentManifest }) {
  return (
    <div className="min-w-0 rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2">
      <p className="mb-1.5 truncate text-[11px] font-semibold uppercase tracking-wide text-[var(--ink-muted-text)]">
        {label}
      </p>
      {/* Fixed grid-cols-3 at every VIEWPORT size, deliberately not a
          `sm:grid-cols-6` breakpoint switch: Tailwind breakpoints key off
          viewport width, but this strip's actual available width is
          whatever the sidebar card happens to be (a fixed ~340px on
          desktop, from PlayScreen's own layout) -- a viewport-based
          6-column jump at desktop widths crammed six labels into ~50px
          each ("Opti...", "Late...", "Nan..." truncating everywhere) even
          though the CONTAINER never got any wider. 2 rows of 3 stays
          readable at the container's actual width. */}
      <dl className="grid grid-cols-3 gap-x-3 gap-y-2 text-xs text-[var(--ink-2)]">
        <div className="min-w-0">
          <dt className="truncate text-[var(--ink-muted-text)]">Optimality</dt>
          <dd className="tabular truncate font-medium text-[var(--ink)]">
            {agent.optimality != null ? formatPct(agent.optimality) : '—'}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="truncate text-[var(--ink-muted-text)]">Size</dt>
          <dd className="tabular truncate font-medium text-[var(--ink)]">{formatBytes(agent.size_bytes)}</dd>
        </div>
        <div className="min-w-0">
          <dt className="truncate text-[var(--ink-muted-text)]">FLOPs/move</dt>
          <dd className="tabular truncate font-medium text-[var(--ink)]">{formatFlops(agent.flops_per_move)}</dd>
        </div>
        <div className="min-w-0">
          <dt className="truncate text-[var(--ink-muted-text)]">Latency</dt>
          <dd className="tabular truncate font-medium text-[var(--ink)]">
            {agent.latency_ms != null ? formatLatency(agent.latency_ms) : '—'}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="truncate text-[var(--ink-muted-text)]">Elo</dt>
          <dd className="tabular truncate font-medium text-[var(--ink)]">{agent.elo ?? '—'}</dd>
        </div>
        <div className="min-w-0">
          <dt className="truncate text-[var(--ink-muted-text)]">Tier</dt>
          <dd className="truncate font-medium text-[var(--ink)]">{agent.tier ? TIER_LABEL[agent.tier] : '—'}</dd>
        </div>
      </dl>
    </div>
  );
}

interface NewGamePanelProps {
  agents: AgentManifest[];
  onStart: (firstAgent: string | null, secondAgent: string | null) => void;
  busy?: boolean;
  /** Preselect this agent as the opponent (Yellow) -- e.g. when the user
   * clicked "Play" on a card in the Agents screen. */
  presetOpponent?: string | null;
}

export function NewGamePanel({ agents, onStart, busy, presetOpponent }: NewGamePanelProps) {
  // Default opponent = the 0-byte champion ("Zero"), falling back to the
  // first registered agent if it isn't present (e.g. a stripped-down local
  // registry).
  const defaultOpponent =
    agents.find((a) => a.name === 'neurofour-net14')?.name ?? agents[0]?.name ?? '';
  const [first, setFirst] = useState<string>(HUMAN);
  const [second, setSecond] = useState<string>(presetOpponent ?? defaultOpponent);

  // A later "Play" click on an Agents-screen card re-preselects the
  // opponent even after this panel already mounted once.
  useEffect(() => {
    if (presetOpponent) setSecond(presetOpponent);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetOpponent]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    onStart(first === HUMAN ? null : first, second === HUMAN ? null : second);
  };

  const watchMode = first !== HUMAN && second !== HUMAN;
  const firstAgent = first !== HUMAN ? (agents.find((a) => a.name === first) ?? null) : null;
  const secondAgent = second !== HUMAN ? (agents.find((a) => a.name === second) ?? null) : null;

  return (
    <form
      onSubmit={submit}
      className="flex flex-col gap-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4"
    >
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <AgentPicker
          id="first-agent"
          label="Red (moves first)"
          agents={agents}
          value={first}
          onChange={setFirst}
        />
        <AgentPicker
          id="second-agent"
          label="Yellow (moves second)"
          agents={agents}
          value={second}
          onChange={setSecond}
        />
      </div>

      {firstAgent || secondAgent ? (
        <div className="flex flex-col gap-2">
          {firstAgent ? <AgentStatStrip label={`Red — ${firstAgent.display_name}`} agent={firstAgent} /> : null}
          {secondAgent ? <AgentStatStrip label={`Yellow — ${secondAgent.display_name}`} agent={secondAgent} /> : null}
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-medium text-[var(--ink-2)]">
          {watchMode
            ? 'Agent vs agent — start, then use Watch to auto-play.'
            : 'Pick who plays which color, then start.'}
        </p>
        <button
          type="submit"
          disabled={busy}
          className="shrink-0 whitespace-nowrap rounded-md px-3 py-2 text-sm font-semibold cursor-pointer disabled:opacity-50
            disabled:cursor-not-allowed"
          style={{ backgroundColor: 'var(--accent-solid)', color: 'var(--accent-solid-ink)' }}
        >
          {busy ? 'Starting…' : 'New game'}
        </button>
      </div>
    </form>
  );
}
