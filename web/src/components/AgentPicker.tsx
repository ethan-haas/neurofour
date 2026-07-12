import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import type { AgentManifest } from '../types';
import { TIER_LABEL, formatBytes, formatFlops, formatLatency, formatPct } from '../lib/format';

export const HUMAN = '__human__';

interface AgentOption {
  value: string;
  agent: AgentManifest | null; // null = "You (human)"
}

interface AgentPickerProps {
  id: string;
  label: string;
  agents: AgentManifest[];
  value: string;
  onChange: (value: string) => void;
  /** The current headline/champion agent name (e.g. from the leaderboard),
   * so the champion badge tracks the backend's own verdict instead of a
   * hardcoded id. */
  championName?: string | null;
  disabled?: boolean;
}

function buildOptions(agents: AgentManifest[]): AgentOption[] {
  return [{ value: HUMAN, agent: null }, ...agents.map((a) => ({ value: a.name, agent: a }))];
}

/** Compact stat chips shown inline on each option row (and reused on the
 * trigger's collapsed summary). Renders nothing for a stat that's null
 * (no leaderboard row for this agent) rather than a fabricated placeholder. */
function OptionStats({ agent }: { agent: AgentManifest }) {
  return (
    <span className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-[var(--ink-muted-text)]">
      {agent.optimality != null ? <span className="tabular">{formatPct(agent.optimality)} opt</span> : null}
      <span className="tabular">{formatBytes(agent.size_bytes)}</span>
      {agent.tier ? (
        <span
          className="rounded px-1 py-0.5 text-[10px] font-semibold"
          style={{ backgroundColor: 'var(--surface-2)', color: 'var(--ink-2)' }}
        >
          {TIER_LABEL[agent.tier]}
        </span>
      ) : null}
    </span>
  );
}

/** Accessible rich agent picker: a collapsible listbox button (WAI-ARIA APG
 * "select only" combobox pattern) whose options show display name + subtitle
 * + compact stats, not a bare native `<select>`. The trigger retains DOM
 * focus the whole time (the popup itself is never focused); the currently
 * highlighted row is communicated via `aria-activedescendant` + a visual
 * highlight, exactly the pattern a screen reader expects from a listbox
 * button. Arrow keys preview/commit like a native `<select>` (selection
 * follows highlighting); Escape reverts to whatever was selected when the
 * popup opened and closes; clicking outside also closes without side effects
 * beyond whatever was already committed by keyboard/click selection. */
export function AgentPicker({ id, label, agents, value, onChange, championName, disabled }: AgentPickerProps) {
  const options = useMemo(() => buildOptions(agents), [agents]);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const openedValueRef = useRef(value);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const listRef = useRef<HTMLUListElement | null>(null);
  const reactId = useId();
  const labelId = `${id}-label-${reactId}`;
  const buttonId = `${id}-button-${reactId}`;
  const listboxId = `${id}-listbox-${reactId}`;
  const optionId = (i: number) => `${id}-option-${i}-${reactId}`;

  const selectedIndex = Math.max(
    0,
    options.findIndex((o) => o.value === value),
  );
  const selectedOption = options[selectedIndex] ?? options[0];

  const openList = useCallback(() => {
    openedValueRef.current = value;
    setActiveIndex(selectedIndex);
    setOpen(true);
  }, [selectedIndex, value]);

  const closeList = useCallback(() => {
    setOpen(false);
  }, []);

  const commit = useCallback(
    (index: number) => {
      const opt = options[index];
      if (opt) onChange(opt.value);
    },
    [onChange, options],
  );

  // Click-outside-to-close.
  useEffect(() => {
    if (!open) return;
    const onDocMouseDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        closeList();
      }
    };
    document.addEventListener('mousedown', onDocMouseDown);
    return () => document.removeEventListener('mousedown', onDocMouseDown);
  }, [open, closeList]);

  // Keep the active row scrolled into view as it changes via keyboard.
  useEffect(() => {
    if (!open) return;
    const el = listRef.current?.querySelector<HTMLElement>(`#${CSS.escape(optionId(activeIndex))}`);
    el?.scrollIntoView({ block: 'nearest' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, activeIndex]);

  const moveActive = (delta: number) => {
    setActiveIndex((i) => {
      const next = Math.max(0, Math.min(options.length - 1, i + delta));
      commit(next);
      return next;
    });
  };

  const onButtonKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;
    if (!open) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openList();
      }
      return;
    }
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        moveActive(1);
        break;
      case 'ArrowUp':
        e.preventDefault();
        moveActive(-1);
        break;
      case 'Home':
        e.preventDefault();
        setActiveIndex(0);
        commit(0);
        break;
      case 'End':
        e.preventDefault();
        setActiveIndex(options.length - 1);
        commit(options.length - 1);
        break;
      case 'Enter':
      case ' ':
      case 'Tab':
        if (e.key !== 'Tab') e.preventDefault();
        closeList();
        break;
      case 'Escape':
        e.preventDefault();
        onChange(openedValueRef.current);
        closeList();
        break;
      default:
        break;
    }
  };

  const summary = selectedOption.agent ? selectedOption.agent.display_name : 'You (human)';

  return (
    <div ref={rootRef} className="flex flex-col gap-1 text-sm">
      <span id={labelId} className="font-medium text-[var(--ink)]">
        {label}
      </span>
      <div className="relative">
        <button
          ref={buttonRef}
          id={buttonId}
          type="button"
          disabled={disabled}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-labelledby={`${labelId} ${buttonId}`}
          aria-controls={listboxId}
          aria-activedescendant={open ? optionId(activeIndex) : undefined}
          onClick={() => (open ? closeList() : openList())}
          onKeyDown={onButtonKeyDown}
          className="flex w-full items-center justify-between gap-2 rounded-md border border-[var(--border-strong)]
            bg-[var(--surface)] px-2.5 py-1.5 text-left text-[var(--ink)] cursor-pointer
            hover:border-[var(--border-strong)] disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span className="flex min-w-0 flex-col">
            <span className="truncate font-medium">{summary}</span>
            {selectedOption.agent ? (
              <span className="truncate text-[11px] text-[var(--ink-muted-text)]">{selectedOption.agent.subtitle}</span>
            ) : null}
          </span>
          <span aria-hidden="true" className="shrink-0 text-[var(--ink-muted-text)]">
            ▾
          </span>
        </button>

        {open ? (
          <ul
            ref={listRef}
            id={listboxId}
            role="listbox"
            aria-labelledby={labelId}
            tabIndex={-1}
            className="absolute z-30 mt-1 max-h-72 w-full min-w-[260px] overflow-y-auto rounded-md border
              border-[var(--border-strong)] bg-[var(--surface)] py-1 shadow-lg"
          >
            {options.map((opt, i) => {
              const isSelected = opt.value === value;
              const isActive = i === activeIndex;
              const isChampion = opt.agent != null && opt.agent.name === championName;
              return (
                <li
                  key={opt.value}
                  id={optionId(i)}
                  role="option"
                  aria-selected={isSelected}
                  onMouseEnter={() => setActiveIndex(i)}
                  onClick={() => {
                    setActiveIndex(i);
                    commit(i);
                    closeList();
                    buttonRef.current?.focus();
                  }}
                  className="flex cursor-pointer flex-col gap-0.5 px-2.5 py-1.5"
                  style={{ backgroundColor: isActive ? 'var(--surface-2)' : undefined }}
                >
                  <span className="flex items-center gap-1.5">
                    <span className="font-medium text-[var(--ink)]">
                      {opt.agent ? opt.agent.display_name : 'You (human)'}
                    </span>
                    {isChampion ? (
                      <span
                        className="rounded px-1 py-0.5 text-[9px] font-semibold"
                        style={{ backgroundColor: 'var(--accent-solid)', color: 'var(--accent-solid-ink)' }}
                      >
                        champion
                      </span>
                    ) : null}
                    {opt.agent?.pareto ? (
                      <span
                        className="rounded px-1 py-0.5 text-[9px] font-semibold"
                        style={{ color: 'var(--good-text)', border: '1px solid var(--good)' }}
                      >
                        pareto
                      </span>
                    ) : null}
                  </span>
                  {opt.agent ? (
                    <span className="text-[11px] text-[var(--ink-muted-text)]">{opt.agent.subtitle}</span>
                  ) : null}
                  {opt.agent ? <OptionStats agent={opt.agent} /> : null}
                </li>
              );
            })}
          </ul>
        ) : null}
      </div>

      {selectedOption.agent ? (
        <dl
          className="mt-1 grid grid-cols-3 gap-x-3 gap-y-1 rounded-md border border-[var(--border)]
            bg-[var(--surface-2)] px-2.5 py-2 text-[11px] text-[var(--ink-2)]"
        >
          <div>
            <dt className="text-[var(--ink-muted-text)]">Optimality</dt>
            <dd className="tabular font-medium text-[var(--ink)]">
              {selectedOption.agent.optimality != null ? formatPct(selectedOption.agent.optimality) : '—'}
            </dd>
          </div>
          <div>
            <dt className="text-[var(--ink-muted-text)]">Size</dt>
            <dd className="tabular font-medium text-[var(--ink)]">{formatBytes(selectedOption.agent.size_bytes)}</dd>
          </div>
          <div>
            <dt className="text-[var(--ink-muted-text)]">FLOPs/move</dt>
            <dd className="tabular font-medium text-[var(--ink)]">{formatFlops(selectedOption.agent.flops_per_move)}</dd>
          </div>
          <div>
            <dt className="text-[var(--ink-muted-text)]">Latency</dt>
            <dd className="tabular font-medium text-[var(--ink)]">
              {selectedOption.agent.latency_ms != null ? formatLatency(selectedOption.agent.latency_ms) : '—'}
            </dd>
          </div>
          <div>
            <dt className="text-[var(--ink-muted-text)]">Elo</dt>
            <dd className="tabular font-medium text-[var(--ink)]">{selectedOption.agent.elo ?? '—'}</dd>
          </div>
          <div>
            <dt className="text-[var(--ink-muted-text)]">Tier</dt>
            <dd className="font-medium text-[var(--ink)]">
              {selectedOption.agent.tier ? TIER_LABEL[selectedOption.agent.tier] : '—'}
            </dd>
          </div>
        </dl>
      ) : null}
    </div>
  );
}
