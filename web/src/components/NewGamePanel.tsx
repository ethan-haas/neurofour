import { useEffect, useState } from 'react';
import type { AgentManifest } from '../types';
import { AgentPicker, HUMAN } from './AgentPicker';

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
