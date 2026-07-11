import { useState } from 'react';
import type { AgentManifest } from '../types';

interface NewGamePanelProps {
  agents: AgentManifest[];
  onStart: (firstAgent: string | null, secondAgent: string | null) => void;
  busy?: boolean;
}

const HUMAN = '__human__';

export function NewGamePanel({ agents, onStart, busy }: NewGamePanelProps) {
  const defaultOpponent = agents.find((a) => a.name === 'neurofour-net')?.name ?? agents[0]?.name ?? '';
  const [first, setFirst] = useState<string>(HUMAN);
  const [second, setSecond] = useState<string>(defaultOpponent);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    onStart(first === HUMAN ? null : first, second === HUMAN ? null : second);
  };

  const watchMode = first !== HUMAN && second !== HUMAN;

  return (
    <form
      onSubmit={submit}
      className="flex flex-col gap-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4"
    >
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="grid grid-rows-[2.75rem_auto] gap-1 text-sm">
          <span className="flex items-end font-medium text-[var(--ink)]">Red (moves first)</span>
          <select
            value={first}
            onChange={(e) => setFirst(e.target.value)}
            className="rounded-md border border-[var(--border-strong)] bg-[var(--surface)] px-2 py-1.5 text-[var(--ink)]"
          >
            <option value={HUMAN}>You (human)</option>
            {agents.map((a) => (
              <option key={a.name} value={a.name}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        <label className="grid grid-rows-[2.75rem_auto] gap-1 text-sm">
          <span className="flex items-end font-medium text-[var(--ink)]">Yellow (moves second)</span>
          <select
            value={second}
            onChange={(e) => setSecond(e.target.value)}
            className="rounded-md border border-[var(--border-strong)] bg-[var(--surface)] px-2 py-1.5 text-[var(--ink)]"
          >
            <option value={HUMAN}>You (human)</option>
            {agents.map((a) => (
              <option key={a.name} value={a.name}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
      </div>

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
