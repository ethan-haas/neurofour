import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import type { AgentManifest } from '../types';
import { Badge } from './Badge';
import { TIER_LABEL, formatBytes, formatPct, displayName } from '../lib/format';

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
  // The menu header states the list is "ranked by NeuroFour score", so it must
  // actually BE ranked -- the API returns agents in registry order, which put
  // the 0-byte champion ("Zero") in the middle of the list and made the header
  // a lie. Sort by neurogolf_score descending (the same score the leaderboard
  // ranks by); agents without a scored leaderboard row sort last, ties keep
  // their incoming order. "You (human)" always stays pinned first.
  const ranked = [...agents].sort((a, b) => (b.neurogolf_score ?? -Infinity) - (a.neurogolf_score ?? -Infinity));
  return [{ value: HUMAN, agent: null }, ...ranked.map((a) => ({ value: a.name, agent: a }))];
}

/** Stat "chips" shown on each option row. Previously these were a loose
 * inline run of spans at 11px, so nothing lined up down the list -- "3.2
 * KB", "24.1 KB", "0 B" each started at a different x, and the eye couldn't
 * scan the column. Fixed-width columns + tabular numerals + right-aligned
 * numbers turn it into an actual scannable table; the tier badge gets its
 * own trailing column so it never reflows into the number columns. Renders
 * an em dash (not a blank) for a stat that's genuinely absent (no
 * leaderboard row for this agent yet), so the column still holds its width. */
function OptionStats({ agent }: { agent: AgentManifest }) {
  return (
    <span className="flex items-center gap-x-3 text-xs text-[var(--ink-muted-text)]">
      <span className="tabular inline-block w-[60px] text-right">
        {agent.optimality != null ? `${formatPct(agent.optimality)} opt` : '—'}
      </span>
      <span className="tabular inline-block w-14 text-right">{formatBytes(agent.size_bytes)}</span>
      <span className="inline-block w-[70px]">
        {agent.tier ? (
          <span
            className="rounded px-1 py-0.5 text-[10px] font-semibold"
            style={{ backgroundColor: 'var(--surface-2)', color: 'var(--ink-2)' }}
          >
            {TIER_LABEL[agent.tier]}
          </span>
        ) : null}
      </span>
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

  const summary = selectedOption.agent ? displayName(selectedOption.agent) : 'You (human)';

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
          // Outer wrapper carries the border/shadow/positioning; the count
          // header is sticky at its top (outside the scroll region, so it
          // never scrolls away), and the `<ul>` itself is the ONLY scrolling
          // element. A visible (not auto-hiding-overlay) scrollbar --
          // `.picker-scroll` in index.css -- plus this header together
          // replace what used to be a scroll container with literally no
          // visible affordance that it scrolled at all, or how many options
          // existed (vision review: "reasonably concludes the arena has 4
          // agents"). max-h-64 deliberately ends mid-row, not on a row
          // boundary, so the cut-off row itself signals "more below".
          <div
            className="absolute z-30 mt-1 w-full min-w-[280px] overflow-hidden rounded-md border
              border-[var(--border-strong)] bg-[var(--surface)] shadow-lg"
          >
            <div
              className="border-b border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-1.5 text-[11px]
                font-semibold uppercase tracking-wide text-[var(--ink-2)]"
            >
              {options.length} options · ranked by NeuroFour score
            </div>
            <ul
              ref={listRef}
              id={listboxId}
              role="listbox"
              aria-labelledby={labelId}
              tabIndex={-1}
              className="picker-scroll max-h-64 overflow-y-auto py-1"
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
                    className="flex cursor-pointer flex-col gap-0.5 border-l-[3px] px-2 py-1.5"
                    style={{
                      backgroundColor: isActive ? 'var(--surface-2)' : isSelected ? 'var(--accent-tint)' : undefined,
                      borderLeftColor: isSelected ? 'var(--accent-solid)' : 'transparent',
                    }}
                  >
                    <span className="flex items-center gap-1.5">
                      {/* Leading check, not just a ~2% background tint, so
                          the selected row is unmistakable at a glance (the
                          old selected-row highlight was nearly invisible --
                          vision review measured it as ~2% lighter gray). */}
                      <span
                        aria-hidden="true"
                        className="inline-block w-3 shrink-0 text-center font-bold"
                        style={{ color: 'var(--accent-solid)' }}
                      >
                        {isSelected ? '✓' : ''}
                      </span>
                      <span className="font-medium text-[var(--ink)]">
                        {opt.agent ? displayName(opt.agent) : 'You (human)'}
                      </span>
                      {isChampion ? <Badge variant="accent">champion</Badge> : null}
                      {opt.agent?.pareto ? <Badge variant="good">pareto</Badge> : null}
                    </span>
                    {opt.agent ? (
                      <span className="pl-[18px] text-[11px] text-[var(--ink-muted-text)]">{opt.agent.subtitle}</span>
                    ) : null}
                    {opt.agent ? (
                      <span className="pl-[18px]">
                        <OptionStats agent={opt.agent} />
                      </span>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  );
}
